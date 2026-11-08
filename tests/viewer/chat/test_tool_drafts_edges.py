"""Draft cards: repeated names, too many drafts, a new model call, and eval targets."""

from __future__ import annotations

from collections.abc import AsyncIterator

from agent_adapter.core.messages import ToolCallDelta
from agent_adapter.core.types import Event
from agent_adapter.evals._calls import joined
from agent_adapter.tracing.events import EventType
from agent_adapter.viewer.chat.live import MAX_DRAFTS, MAX_NAME, LiveTurn
from agent_adapter.viewer.chat.watch import _route


def _drafts(live: LiveTurn) -> list[dict[str, object]]:
    return [e for e in live.events if e["type"] == "tool_draft"]


def test_a_name_repeated_on_every_fragment_keeps_one_draft() -> None:
    live = LiveTurn("t", "s", "r")
    for piece in ('{"q', 'uery": "mu', 'gs"}'):
        live.draft(ToolCallDelta(index=0, id="c1", name="search", arguments_delta=piece))
    shown = _drafts(live)
    assert not any(e.get("closed") for e in shown)
    assert shown[-1]["arguments"] == '{"query": "mugs"}'
    assert len(live.drafts) == 1


def test_a_name_or_id_arriving_later_is_adopted() -> None:
    live = LiveTurn("t", "s", "r")
    live.draft(ToolCallDelta(index=0, arguments_delta="{"))
    live.draft(ToolCallDelta(index=0, id="c1", name="search"))
    assert (live.drafts[0].call_id, live.drafts[0].name) == ("c1", "search")
    assert not any(e.get("closed") for e in _drafts(live))


def test_drafts_and_names_are_bounded() -> None:
    live = LiveTurn("t", "s", "r")
    for index in range(MAX_DRAFTS + 10):
        live.draft(ToolCallDelta(index=index, id=f"c{index}", name="n" * 500))
    assert len(live.drafts) == MAX_DRAFTS
    assert all(len(d.name) == MAX_NAME for d in live.drafts.values())


def test_a_new_model_call_drops_drafts_whose_calls_never_ran() -> None:
    live = LiveTurn("t", "s", "r")
    live.draft(ToolCallDelta(index=0, id="c1", name="nothing_like_this"))
    live.draft(ToolCallDelta(index=1, id="c2", name="other"))
    _route(live, Event(type=EventType.LLM_STARTED, run_id="r", data={"provider": "p"}))
    closed = [(e["index"], e["closed"]) for e in _drafts(live) if e.get("closed")]
    assert closed == [(0, True), (1, True)]
    assert live.drafts == {}
    live.close_drafts()  # nothing open: nothing sent
    assert len(_drafts(live)) == 4


async def test_an_eval_target_that_streams_tool_deltas_joins_the_text_only() -> None:
    async def reply() -> AsyncIterator[str | ToolCallDelta]:
        yield "Looking. "
        yield ToolCallDelta(index=0, id="c1", name="stock")
        yield "12 left."

    assert await joined(reply(), "agent") == "Looking. 12 left."
