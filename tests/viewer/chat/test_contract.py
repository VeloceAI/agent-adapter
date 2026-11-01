"""The agent contract: str, async iterator, sync function, errors; history and state."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_adapter.core.errors import ToolTimeout
from agent_adapter.viewer.chat import ChatMessage, ChatSession
from tests.viewer.chat.helpers import chat_served, new_session, read_stream, send_turn


def texts(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "text")


async def async_text(message: ChatMessage, session: ChatSession) -> str:
    return f"echo {message.text}"


async def streamed(message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
    for word in ("one ", "two ", "three"):
        await asyncio.sleep(0)
        yield word


def sync_agent(message: ChatMessage, session: ChatSession) -> str:
    session.state["seen"] = session.state.get("seen", 0) + 1
    return f"{len(session.history)} earlier, said {message.text}"


class Streamer:
    async def __call__(self, message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
        yield "from "
        yield "an object"


def returns_iterator(message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
    return streamed(message, session)


@pytest.mark.parametrize(
    ("agent", "answer"),
    [
        (async_text, "echo hi"),
        (streamed, "one two three"),
        (Streamer(), "from an object"),
        (returns_iterator, "one two three"),
    ],
)
def test_every_reply_shape_streams_to_a_done_turn(
    tmp_path: Path, agent: object, answer: str
) -> None:
    with chat_served(tmp_path, agent=agent) as server:
        sid = new_session(server)
        turn = send_turn(server, sid, "hi")
        events = read_stream(server, turn["turn_id"])
    assert texts(events) == answer
    assert events[-1]["type"] == "done" and events[-1]["turn"]["answer"] == answer
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_a_sync_agent_runs_in_a_thread_and_sees_history_and_state(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=sync_agent) as server:
        sid = new_session(server, state={"seen": 0})
        first = read_stream(server, send_turn(server, sid, "a")["turn_id"])
        second = read_stream(server, send_turn(server, sid, "b")["turn_id"])
    assert texts(first) == "0 earlier, said a"
    assert texts(second) == "2 earlier, said b"
    assert second[-2] == {"seq": second[-2]["seq"], "type": "state", "state": {"seen": 2}}


@pytest.mark.parametrize(
    ("error", "kind", "message"),
    [
        (ToolTimeout("lookup", 3.0), "tool-timeout", "lookup"),
        (ValueError("secret detail"), "internal-error", "ValueError"),
    ],
)
def test_an_exception_fails_the_turn_with_a_safe_message(
    tmp_path: Path, error: Exception, kind: str, message: str
) -> None:
    async def failing(message_: ChatMessage, session: ChatSession) -> str:
        raise error

    with chat_served(tmp_path, agent=failing) as server:
        events = read_stream(server, send_turn(server, new_session(server), "x")["turn_id"])
    end = events[-1]
    assert (end["type"], end["status"], end["kind"]) == ("error", "failed", kind)
    assert message in end["message"] and "secret detail" not in str(events)


@pytest.mark.parametrize("reply", [42, ["not", "text"]])
def test_a_wrong_return_type_fails_the_turn(tmp_path: Path, reply: object) -> None:
    async def wrong(message: ChatMessage, session: ChatSession) -> object:
        return reply

    with chat_served(tmp_path, agent=wrong) as server:
        events = read_stream(server, send_turn(server, new_session(server), "x")["turn_id"])
    assert events[-1]["kind"] == "tool-usage-error" and "must return str" in events[-1]["message"]


def test_a_non_text_chunk_fails_the_turn(tmp_path: Path) -> None:
    async def bad_chunks(message: ChatMessage, session: ChatSession) -> AsyncIterator[object]:
        yield "fine "
        yield 7

    with chat_served(tmp_path, agent=bad_chunks) as server:
        events = read_stream(server, send_turn(server, new_session(server), "x")["turn_id"])
    assert texts(events) == "fine " and "str chunks" in events[-1]["message"]
    assert events[-1]["turn"]["answer"] == "fine "
