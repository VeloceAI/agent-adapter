"""Tool arguments streamed into a draft card: parsed, masked, closed when the call runs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
)
from agent_adapter.core.types import ToolCall
from agent_adapter.models.callable import CallableModel
from agent_adapter.viewer.chat.contract import ChatMessage, ChatSession
from agent_adapter.viewer.chat.invoke import reply_chunks
from agent_adapter.viewer.chat.live import MAX_PREVIEW, LiveTurn
from tests.fakes.models import text
from tests.viewer.chat.helpers import chat_served, new_session, read_stream, send_turn, tools


def _drafts(live: LiveTurn) -> list[dict[str, Any]]:
    return [e for e in live.events if e["type"] == "tool_draft"]


def test_a_draft_shows_parsed_arguments_with_secrets_masked() -> None:
    live = LiveTurn("t", "s", "r")
    live.draft(ToolCallDelta(index=0, id="c1", name="send"))
    live.draft(ToolCallDelta(index=0, arguments_delta='{"to": "Ann", "api_key": "sk-se'))
    live.draft(ToolCallDelta(index=0, arguments_delta="cret"))
    shown = [e["arguments"] for e in _drafts(live)]
    assert shown[0] == ""
    assert all("sk-se" not in item and "cret" not in item for item in shown)
    assert '"to": "Ann"' in shown[-1]
    assert len(shown) == 2  # the masked value did not change with the last chunk
    assert _drafts(live)[-1] == {
        "seq": 1,
        "type": "tool_draft",
        "index": 0,
        "call_id": "c1",
        "name": "send",
        "arguments": shown[-1],
    }


def test_unparsable_arguments_show_nothing_and_long_ones_are_cut() -> None:
    live = LiveTurn("t", "s", "r")
    live.draft(ToolCallDelta(index=0, id="c1", name="add", arguments_delta="not json"))
    assert _drafts(live)[-1]["arguments"] == ""
    live.draft(ToolCallDelta(index=1, id="c2", name="add", arguments_delta='{"a": "' + "x" * 5000))
    cut = _drafts(live)[-1]["arguments"]
    assert len(cut) == MAX_PREVIEW + 3 and cut.endswith("...")


def test_the_call_closes_its_draft_by_id_or_by_name() -> None:
    live = LiveTurn("t", "s", "r")
    live.draft(ToolCallDelta(index=0, id="c1", name="add", arguments_delta="{}"))
    live.draft(ToolCallDelta(index=1, name="send", arguments_delta="{}"))  # no id (Gemini)
    live.tool("c1", name="add")
    live.tool("loop:x:0:1", name="send")
    closed = [e for e in _drafts(live) if e.get("closed")]
    assert [(e["index"], e["call_id"]) for e in closed] == [(0, "c1"), (1, "loop:x:0:1")]
    assert live.drafts == {}
    assert [t["call_id"] for t in live.tool_list()] == ["c1", "loop:x:0:1"]


def test_a_new_name_at_an_index_replaces_an_unclosed_draft() -> None:
    live = LiveTurn("t", "s", "r")
    live.draft(ToolCallDelta(index=0, id="c1", name="add"))
    live.draft(ToolCallDelta(index=0, id="c9", name="send"))
    assert [e.get("closed", False) for e in _drafts(live)] == [False, True, False]
    assert live.drafts[0].call_id == "c9"
    live.tool("other", name="nothing")  # no matching draft: nothing closed
    assert len(_drafts(live)) == 3


async def test_reply_chunks_passes_tool_deltas_and_refuses_other_objects() -> None:
    async def agent(message: ChatMessage, session: ChatSession) -> AsyncIterator[Any]:
        yield "hi"
        yield ToolCallDelta(index=0, name="add")
        yield 3

    session = ChatSession(id="s", history=(), state={}, executor=None, run_id="r")  # type: ignore[arg-type]
    seen: list[Any] = []
    with pytest.raises(ToolUsageError):
        async for chunk in reply_chunks(agent, ChatMessage("x"), session):
            seen.append(chunk)
    assert seen == ["hi", ToolCallDelta(index=0, name="add")]


def test_the_builtin_agent_streams_tool_arguments_into_a_draft(tmp_path: Path) -> None:
    replies = iter(["call", "answer"])

    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        if next(replies) == "answer":
            yield TextDelta(text="5")
            yield StreamDone(response=text("5"))
            return
        yield ToolCallDelta(index=0, id="c1", name="add")
        yield ToolCallDelta(index=0, arguments_delta='{"a": 2, ')
        yield ToolCallDelta(index=0, arguments_delta='"b": 3}')
        call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
        message = Message(role="assistant", tool_calls=(call,))
        response = LLMResponse(
            provider="callable",
            model="host",
            tool_calls=(call,),
            finish_reason="tool_calls",
            message=message,
        )
        yield StreamDone(response=response)

    model = CallableModel(lambda request: "unused", stream_fn=events)
    with chat_served(tmp_path, model=model, registry=tools) as server:
        found = read_stream(server, send_turn(server, new_session(server), "2+3?")["turn_id"])
    assert isinstance(found, list)
    kinds = [e["type"] for e in found if e["type"] in ("tool_draft", "tool")]
    assert kinds[:4] == ["tool_draft", "tool_draft", "tool_draft", "tool_draft"]
    drafts = [e for e in found if e["type"] == "tool_draft"]
    assert drafts[2]["arguments"] == '{"a": 2, "b": 3}'
    assert drafts[-1] == {**drafts[-1], "closed": True, "call_id": "c1"}
    assert [e for e in found if e["type"] == "tool"][-1]["status"] == "ok"
