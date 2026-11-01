"""Host edge cases: start failures, history, stored failures, close, and small helpers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import LLMRequest
from agent_adapter.core.types import ApprovalRequest, Event
from agent_adapter.exec import Executor
from agent_adapter.viewer.chat import ChatMessage, ChatSession
from agent_adapter.viewer.chat import approvals as approvals_module
from agent_adapter.viewer.chat import builtin as builtin_module
from agent_adapter.viewer.chat import host as host_module
from agent_adapter.viewer.chat.approvals import ChatApprovals
from agent_adapter.viewer.chat.builtin import BuiltinAgent
from agent_adapter.viewer.chat.contract import Attachment
from agent_adapter.viewer.chat.evalcases import EvalCases
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.records import Refused, Turn
from agent_adapter.viewer.chat.watch import TurnWatch
from tests.fakes.models import text
from tests.viewer.chat.helpers import (
    chat,
    chat_served,
    get,
    new_session,
    post,
    read_stream,
    send_turn,
    tools,
)
from tests.viewer.chat.test_contract import async_text
from tests.viewer.helpers import request

calls_seen: list[int] = []


async def flaky(message: ChatMessage, session: ChatSession) -> str:
    calls_seen.append(len(session.history))
    if message.text == "fail":
        raise RuntimeError("boom")
    return "ok"


def test_failed_turns_are_left_out_of_history_and_restored_as_errors(tmp_path: Path) -> None:
    calls_seen.clear()
    with chat_served(tmp_path, agent=flaky) as server:
        sid = new_session(server)
        failed = send_turn(server, sid, "fail")
        read_stream(server, failed["turn_id"])
        read_stream(server, send_turn(server, sid, "next")["turn_id"])
    assert calls_seen == [0, 0]
    with chat_served(tmp_path, agent=flaky) as server:
        events = read_stream(server, failed["turn_id"])
    assert [e["type"] for e in events] == ["error"]
    assert (events[0]["status"], events[0]["kind"]) == ("failed", "internal-error")


def test_a_turn_that_cannot_start_is_stored_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        host = chat(server)

        def broken(*args: object, **kwargs: object) -> None:
            raise ToolUsageError("closed")

        monkeypatch.setattr(host._runner, "run", broken)
        sid = new_session(server)
        status, body = post(server, f"/api/chat/sessions/{sid}/turns", {"text": "x"})
        assert status == 503 and "did not start" in body["error"]
        detail = get(server, f"/api/chat/sessions/{sid}")[1]
    assert detail["turns"][0]["status"] == "failed"
    assert detail["session"]["running"] is None


def test_old_event_lists_are_dropped_and_read_back_from_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_module, "MAX_LIVE", 0)
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        first = send_turn(server, sid, "a")
        read_stream(server, first["turn_id"])
        read_stream(server, send_turn(server, sid, "b")["turn_id"])
        assert first["turn_id"] not in chat(server)._live
        events = read_stream(server, first["turn_id"])
    assert [e["type"] for e in events] == ["text", "done"]


def test_too_many_attachments_and_an_empty_post_body(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        ids = [f"fil_{n:016x}" for n in range(21)]
        status, body = post(
            server, f"/api/chat/sessions/{sid}/turns", {"text": "x", "attachment_ids": ids}
        )
        assert status == 400 and "at most 20" in body["error"]
        status, _ = post(server, "/api/chat/sessions")  # no body is {}
        assert status == 201
        assert post(server, "/api/chat/approvals/apr_1/elsewhere", {})[0] == 404
        turn = send_turn(server, sid, "y")
        read_stream(server, turn["turn_id"])
        status, raw, _ = request(server, "GET", f"/api/chat/turns/{turn['turn_id']}/stream?since=x")
        assert status == 200 and "event: done" in raw


def test_a_close_step_that_fails_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        host = chat(server)

        def broken() -> None:
            raise RuntimeError("nope")

        host.on_close.insert(0, broken)
        with caplog.at_level(logging.WARNING):
            host.close()
            host.close()  # twice is fine
    assert "a close step failed" in caplog.text


def test_a_failing_close_of_the_turn_still_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        host = chat(server)

        def broken(live: LiveTurn, status: str) -> list:
            raise RuntimeError("desk broke")

        monkeypatch.setattr(host.desk, "close_turn", broken)
        with caplog.at_level(logging.ERROR):
            events = read_stream(server, send_turn(server, new_session(server), "x")["turn_id"])
    assert events[-1]["type"] == "done" and "closing turn" in caplog.text


async def test_the_builtin_agent_without_stream_sends_one_chunk(tmp_path: Path) -> None:
    class Bare:
        provider, model = "bare", "m"

        async def generate(self, request: LLMRequest) -> object:
            return text("whole answer")

        def cache_key_fields(self) -> dict:
            return {}

    agent = BuiltinAgent(Bare(), tools)  # type: ignore[arg-type]
    executor = Executor(tools)
    session = ChatSession("ses_1", (), {}, executor, "run_1")
    chunks = [chunk async for chunk in agent(ChatMessage("hi"), session)]
    assert chunks == ["whole answer"]


def test_the_builtin_agent_refuses_big_text_attachments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtin_module, "MAX_INLINE", 2)
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    agent = BuiltinAgent(object(), tools)  # type: ignore[arg-type]
    item = Attachment("a.txt", "text/plain", 3, path)
    assert "may hold 2 bytes" in agent.refusal((item,))
    assert item.read_bytes() == b"abc" and item.read_text() == "abc"


async def test_an_async_iterator_without_aclose_is_read() -> None:
    from agent_adapter.viewer.chat.invoke import reply_chunks

    class Plain:
        def __init__(self) -> None:
            self.items = ["a", "b"]

        def __aiter__(self) -> Plain:
            return self

        async def __anext__(self) -> str:
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    async def agent(message: ChatMessage, session: ChatSession) -> Plain:
        return Plain()

    session = ChatSession("s", (), {}, Executor(tools), "r")
    assert [c async for c in reply_chunks(agent, ChatMessage("x"), session)] == ["a", "b"]
    assert session.context().run_id == "r"


async def test_decided_requests_are_forgotten_past_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(approvals_module, "MAX_KEPT", 1)
    watch = TurnWatch()
    watch.add(LiveTurn("turn_1", "ses_1", "run_1"))
    desk = ChatApprovals(watch, reviewer="local", timeout=0.01)
    for n in (1, 2):
        request = ApprovalRequest(approval_id=f"apr_{n}", run_id="run_1", call_id="c", tool="t")
        await desk.ask(request)
    with pytest.raises(Exception, match="apr_1"):
        desk.decide("apr_1", approved=True)


async def test_a_decision_that_lands_as_the_wait_ends_counts() -> None:
    watch = TurnWatch()
    watch.add(LiveTurn("turn_1", "ses_1", "run_1"))
    desk = ChatApprovals(watch, reviewer="ana", timeout=0.2)
    request = ApprovalRequest(approval_id="apr_1", run_id="run_1", call_id="c", tool="t")
    waiting = asyncio.ensure_future(desk.ask(request))
    await asyncio.sleep(0.05)
    item = desk._items["apr_1"]
    desk.decide("apr_1", approved=True)
    assert desk._expired(item).approved
    assert (await waiting).approved


def test_a_broken_event_route_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from agent_adapter.viewer.chat import watch as watch_module

    watch = TurnWatch()
    watch.add(LiveTurn("turn_1", "ses_1", "run_1"))

    def broken(live: LiveTurn, event: Event) -> None:
        raise RuntimeError("bad")

    monkeypatch.setattr(watch_module, "_route", broken)
    with caplog.at_level(logging.WARNING):
        watch.emit(Event("tool.started", run_id="run_1"))
    assert "could not show tool.started" in caplog.text


def test_a_case_that_cannot_be_written_is_a_500(tmp_path: Path) -> None:
    (tmp_path / "eval_cases.jsonl").mkdir()
    cases = EvalCases(tmp_path)
    with pytest.raises(Refused) as failed:
        cases.add(Turn(id="t", session_id="s", run_id="r", text="x"), "c1", None, None)
    assert failed.value.status == 500


async def test_streamed_chunks_are_not_lost_when_the_loop_ends(tmp_path: Path) -> None:
    from agent_adapter.core.messages import StreamDone, StreamEvent, TextDelta
    from agent_adapter.models.callable import CallableModel

    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        for word in ("a", "b", "c"):
            yield TextDelta(text=word)
        yield StreamDone(response=text("abc"))

    agent = BuiltinAgent(CallableModel(lambda r: "x", stream_fn=events), tools)
    session = ChatSession("ses_1", (), {}, Executor(tools), "run_1")
    chunks = [chunk async for chunk in agent(ChatMessage("hi"), session)]
    assert "".join(chunks) == "abc"
