"""The "autogen" format and `autogen_model_context`, on a real AssistantAgent."""

import runpy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("autogen_agentchat")

from autogen_agentchat.agents import AssistantAgent
from autogen_core import FunctionCall
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from agent_adapter.integrations.autogen import autogen_model_context
from agent_adapter.integrations.autogen_compress import AutoGenConverter
from tests.integrations.compression_support import (
    DENIED,
    DROP_OLD,
    EXPECTED,
    PAD,
    assert_cut_keeps_types,
    assert_drop_keeps_whole_rounds,
    assert_round_trip,
    assert_summary_is_native,
    render_read_back,
)
from tests.integrations.test_autogen_contract import ScriptedClient

pytestmark = pytest.mark.contract
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "compression_framework_hooks.py"
CALLS = [
    FunctionCall(id="c1", name="lookup", arguments='{"q": "a"}'),
    FunctionCall(id="c2", name="lookup", arguments="{}"),
]


def history() -> list[Any]:
    results = [
        FunctionExecutionResult(content=f"one {PAD}", name="lookup", call_id="c1"),
        FunctionExecutionResult(content="two", name="lookup", call_id="c2", is_error=True),
    ]
    return [
        SystemMessage(content="Be brief."),
        UserMessage(content=[f"old question {PAD}", "and more"], source="user"),
        AssistantMessage(content=CALLS, thought="I should look", source="assistant"),
        FunctionExecutionResultMessage(content=results),
        AssistantMessage(content=f"answer {PAD}", source="assistant"),
        UserMessage(content="new question", source="user"),
    ]


class RecordingClient(ScriptedClient):
    """`ScriptedClient` that also records how many messages each call got."""

    def __init__(self, turns: list[Any]) -> None:
        super().__init__(turns)
        self.seen: list[int] = []

    async def create(self, messages: Sequence[LLMMessage], **kwargs: Any) -> CreateResult:
        self.seen.append(len(messages))
        return await super().create(messages, **kwargs)


def test_round_trip_keeps_calls_results_and_sources() -> None:
    messages = assert_round_trip(history(), "autogen")
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
    assert [c.id for c in messages[2].tool_calls] == ["c1", "c2"]
    assert [(m.tool_call_id, m.name) for m in messages[3:5]] == [("c1", "lookup"), ("c2", "lookup")]
    assert messages[1].content == f"old question {PAD}\nand more"


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    assert_drop_keeps_whole_rounds(history())


def test_cut_keeps_thought_is_error_and_sources() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    assert smaller[1].source == "user" and "characters cut" in smaller[1].content[0]
    assert smaller[2] is items[2]
    first, second = smaller[3].content
    assert first.call_id == "c1" and "characters cut" in first.content
    assert second is items[3].content[1] and second.is_error


def test_summary_becomes_an_autogen_message() -> None:
    assert_summary_is_native(history(), (SystemMessage, UserMessage, AssistantMessage))


def test_an_approval_denial_keeps_its_round() -> None:
    items = history()
    denied = FunctionExecutionResult(content=DENIED, name="lookup", call_id="c2", is_error=True)
    items[3] = FunctionExecutionResultMessage(content=[items[3].content[0], denied])
    smaller = assert_drop_keeps_whole_rounds(items)
    assert items[3] in smaller and items[2] in smaller


async def test_model_context_keeps_every_message_and_returns_them_compressed() -> None:
    context = autogen_model_context(DROP_OLD, initial_messages=history()[:2])
    assert isinstance(context, ChatCompletionContext)
    for message in history()[2:]:
        await context.add_message(message)
    assert len(await context.get_messages()) < 6
    state = await context.save_state()
    assert len(state["messages"]) == 6


async def test_assistant_agent_sends_the_compressed_context() -> None:
    plain, compressed = RecordingClient(["ok"]), RecordingClient(["ok"])
    earlier = history()[1:5]
    contexts = (
        UnboundedChatCompletionContext(initial_messages=earlier),
        autogen_model_context(DROP_OLD, initial_messages=earlier),
    )
    for client, context in zip((plain, compressed), contexts, strict=True):
        agent = AssistantAgent("a", model_client=client, model_context=context)
        assert (await agent.run(task="new question")).messages[-1].content == "ok"
    assert compressed.seen[0] < plain.seen[0]


async def test_the_example_builds_the_agent() -> None:
    agent = runpy.run_path(str(EXAMPLE))["autogen_agent"](RecordingClient(["ok"]))
    result = await agent.run(task="hi")
    assert result.messages[-1].content == "ok"


def test_new_messages_render_as_autogen_messages() -> None:
    assert render_read_back(AutoGenConverter) == EXPECTED
