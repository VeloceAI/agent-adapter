"""The "smolagents" format and `smolagents_compressed_model`, on a real ToolCallingAgent."""

import runpy
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("smolagents")

from smolagents import ToolCallingAgent, tool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)

from agent_adapter.context.formats import convert
from agent_adapter.integrations.smolagents import smolagents_compressed_model
from agent_adapter.integrations.smolagents_compress import SmolagentsConverter
from tests.integrations.compression_support import (
    DROP_OLD,
    PAD,
    assert_cut_keeps_types,
    assert_drop_keeps_whole_rounds,
    assert_round_trip,
    assert_summary_is_native,
    render_read_back,
)

pytestmark = pytest.mark.contract
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "compression_framework_hooks.py"


def text(value: str) -> list[dict]:
    return [{"type": "text", "text": value}]


def history() -> list[ChatMessage]:
    """What `write_memory_to_messages` gives after one tool step with an image observation."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=text("You are an agent.")),
        ChatMessage(role=MessageRole.USER, content=text(f"New task:\n{PAD}")),
        ChatMessage(role=MessageRole.ASSISTANT, content=text(f"Thought: look {PAD}")),
        ChatMessage(role=MessageRole.TOOL_CALL, content=text("Calling tools:\n[{'id': 'c1'}]")),
        ChatMessage(role=MessageRole.USER, content=[{"type": "image", "image": "png-bytes"}]),
        ChatMessage(role=MessageRole.TOOL_RESPONSE, content=text(f"Observation:\n{PAD}")),
        ChatMessage(role=MessageRole.USER, content=text("Now go on.")),
    ]


class ScriptedModel(Model):
    """An offline model: calls `lookup` twice, then `final_answer`; records each input size."""

    def __init__(self) -> None:
        super().__init__(model_id="scripted")
        self.seen: list[int] = []

    def generate(self, messages: list[Any], **kwargs: Any) -> ChatMessage:
        self.seen.append(len(messages))
        name, args = (
            ("lookup", {"q": "a"}) if len(self.seen) < 3 else ("final_answer", {"answer": "ok"})
        )
        function = ChatMessageToolCallFunction(name=name, arguments=args)
        call = ChatMessageToolCall(function=function, id=f"c{len(self.seen)}", type="function")
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])


@tool
def lookup(q: str) -> str:
    """Look something up.

    Args:
        q: what to look up.
    """
    return f"result {PAD}"


def test_round_trip_pairs_the_call_with_its_response() -> None:
    messages = assert_round_trip(history(), "smolagents")
    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "user",
    ]
    (call,) = messages[3].tool_calls
    assert messages[4].tool_call_id == call.id and call.id.startswith("smolagents-")


def test_the_image_observation_rides_with_its_response() -> None:
    conversion = convert(history())
    assert [g.natives for g in conversion.groups][3:5] == [(3,), (4, 5)]


def test_a_response_without_a_call_is_a_user_turn() -> None:
    items = [history()[0], ChatMessage(role=MessageRole.TOOL_RESPONSE, content=text("Error: x"))]
    assert [m.role for m in convert(items).messages] == ["system", "user"]


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    smaller = assert_drop_keeps_whole_rounds(history())
    roles = [m.role for m in smaller]
    assert (MessageRole.TOOL_CALL in roles) == (MessageRole.TOOL_RESPONSE in roles)


def test_cut_keeps_the_block_lists_and_the_image() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    assert smaller[4] is items[4] and smaller[5].role == MessageRole.TOOL_RESPONSE
    assert (
        isinstance(smaller[5].content, list) and "characters cut" in smaller[5].content[0]["text"]
    )


def test_summary_becomes_a_chat_message() -> None:
    assert_summary_is_native(history(), ChatMessage)


def test_the_wrapped_model_gets_fewer_messages_and_memory_keeps_every_step() -> None:
    plain, wrapped = ScriptedModel(), smolagents_compressed_model(ScriptedModel(), DROP_OLD)
    assert not hasattr(wrapped, "generate_stream")  # the agent then calls generate
    for model in (plain, wrapped):
        agent = ToolCallingAgent(tools=[lookup], model=model, verbosity_level=0)
        assert agent.run(f"task {PAD}") == "ok"
    assert wrapped.seen[-1] < plain.seen[-1]
    assert len(agent.memory.steps) == 4  # the task and three action steps


def test_the_example_builds_the_agent() -> None:
    plain, model = ScriptedModel(), ScriptedModel()
    ToolCallingAgent(tools=[lookup], model=plain, verbosity_level=0).run("task")
    agent = runpy.run_path(str(EXAMPLE))["smolagents_agent"](model, [lookup])
    assert agent.run("task") == "ok" and model.seen[-1] < plain.seen[-1]


def test_new_messages_render_as_chat_messages_with_text_tool_rounds() -> None:
    assert render_read_back(SmolagentsConverter) == [
        ("system", "rules", [], ""),
        ("user", "question", [], ""),
        ("assistant", "", [("smolagents-2", "tools", {})], ""),
        ("tool", "result", [], "smolagents-2"),
        ("assistant", "answer", [], ""),
    ]
