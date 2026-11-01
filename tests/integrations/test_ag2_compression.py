"""The "ag2" format and `ag2_compress_policy`, on real AG2 events and a real Agent."""

import runpy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("ag2")

from ag2 import Agent, Context
from ag2.compact import CompactionSummary
from ag2.config import LLMClient
from ag2.events import (
    BaseEvent,
    DataInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
    UsageEvent,
)
from ag2.testing import TestConfig

from agent_adapter.context.formats import convert
from agent_adapter.integrations.ag2 import ag2_compress_policy
from agent_adapter.integrations.ag2_compress import AG2Converter
from tests.integrations.compression_support import (
    DROP_OLD,
    EXPECTED,
    PAD,
    assert_cut_keeps_types,
    assert_drop_keeps_whole_rounds,
    assert_round_trip,
    assert_summary_is_native,
    render_read_back,
)

pytestmark = pytest.mark.contract
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "compression_framework_hooks.py"


def history() -> list[BaseEvent]:
    """A summary, a question, a call (announced again, with usage), its result, an answer."""
    call = ToolCallEvent("lookup", id="c1", arguments='{"q": "a"}')
    result = ToolResultEvent(
        parent_id="c1", name="lookup", result=ToolResult(DataInput({"v": PAD}))
    )
    return [
        CompactionSummary(summary=f"earlier {PAD}", event_count=3),
        ModelRequest([TextInput(f"old question {PAD}"), TextInput("more")]),
        ModelResponse(ModelMessage(f"let me look {PAD}"), tool_calls=ToolCallsEvent([call])),
        UsageEvent(Usage(prompt_tokens=3)),
        ToolCallsEvent([call]),
        ToolResultsEvent([result]),
        ModelResponse(ModelMessage(f"answer {PAD}")),
        ModelRequest([TextInput("new question")]),
    ]


class Recording(TestConfig):
    """AG2's scripted model that also records how many events each call got."""

    __test__ = False

    def __init__(self, *turns: Any) -> None:
        super().__init__(*turns)
        self.seen: list[int] = []

    def create(self) -> LLMClient:
        inner, seen = super().create(), self.seen

        class Client(LLMClient):
            async def __call__(
                self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any
            ) -> ModelResponse:
                seen.append(len(messages))
                return await inner(messages, context, **kwargs)

        return Client()


def test_round_trip_keeps_ids_and_every_event() -> None:
    messages = assert_round_trip(history(), "ag2")
    assert [m.role for m in messages] == ["user", "user", "assistant", "tool", "assistant", "user"]
    assert messages[2].tool_calls[0].id == "c1" and messages[3].tool_call_id == "c1"
    assert '"v"' in messages[3].content


def test_telemetry_and_announcements_ride_with_their_response() -> None:
    groups = convert(history()).groups
    assert [g.natives for g in groups] == [(0,), (1,), (2, 3, 4), (5,), (6,), (7,)]


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    smaller = assert_drop_keeps_whole_rounds(history())
    kinds = [type(e).__name__ for e in smaller]
    assert ("ToolResultsEvent" in kinds) == ("ToolCallsEvent" in kinds)


def test_cut_keeps_calls_names_and_types() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    assert "characters cut" in smaller[0].summary and smaller[0].event_count == 3
    assert [type(p).__name__ for p in smaller[1].parts] == ["TextInput"]
    assert smaller[2].tool_calls is items[2].tool_calls and smaller[3] is items[3]
    (result,) = smaller[5].results
    assert result.parent_id == "c1" and result.name == "lookup"
    assert "characters cut" in result.result.parts[0].content
    assert items[5].results[0].result.parts[0].data == {"v": PAD}  # the input is untouched


def test_summary_becomes_a_model_request() -> None:
    assert_summary_is_native(history(), BaseEvent)


async def test_the_policy_leaves_lists_the_model_never_reads() -> None:
    policy = ag2_compress_policy(DROP_OLD)
    events = [UsageEvent(Usage(prompt_tokens=1))]
    assert await policy.apply(["p"], events, None) == (["p"], events)
    assert policy.name == "context_compression"


async def test_agent_sends_fewer_events_and_the_stream_keeps_them_all() -> None:
    plain, compressed = (
        Recording("one " + PAD, "two " + PAD, "three"),
        Recording("one " + PAD, "two " + PAD, "three"),
    )
    stored = []
    for config, assembly in ((plain, ()), (compressed, (ag2_compress_policy(DROP_OLD),))):
        reply = await Agent("a", config=config, assembly=assembly).ask(f"first {PAD}")
        reply = await reply.ask(f"second {PAD}")
        reply = await reply.ask("third")
        assert reply.body == "three"
        stored.append(len(list(await reply.history.get_events())))
    assert compressed.seen[-1] < plain.seen[-1]
    assert stored[0] == stored[1]


async def test_the_example_builds_the_agent() -> None:
    agent = runpy.run_path(str(EXAMPLE))["ag2_agent"](TestConfig("ok"))
    assert (await agent.ask("hi")).body == "ok"


def test_new_messages_render_as_events_and_system_text_as_a_request() -> None:
    assert render_read_back(AG2Converter) == [("user", "rules", [], ""), *EXPECTED[1:]]
