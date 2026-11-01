"""The "agno" format and `agno_compressed_model`, on real Agno messages and a real Agent."""

import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("agno")

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse

from agent_adapter.integrations.agno import agno_compressed_model
from agent_adapter.integrations.agno_compress import AgnoConverter
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
CALL = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "a"}'}}


def history() -> list[Message]:
    return [
        Message(role="system", content="Be brief."),
        Message(role="user", content=f"old question {PAD}", name="ann"),
        Message(role="assistant", content=None, tool_calls=[CALL], reasoning_content="r"),
        Message(role="tool", tool_call_id="c1", tool_name="lookup", content=f"result {PAD}"),
        Message(role="assistant", content=[{"type": "text", "text": f"answer {PAD}"}]),
        Message(role="user", content="new question"),
    ]


@dataclass
class RecordingModel(Model):
    """An offline Agno model: records each call's messages and answers "done"."""

    id: str = "recording"
    name: str = "Recording"
    provider: str = "Test"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.seen: list[list[Message]] = []

    def _answer(self, kwargs: dict[str, Any]) -> ModelResponse:
        self.seen.append(list(kwargs["messages"]))
        return ModelResponse(role="assistant", content="done")

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._answer(kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._answer(kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield self._answer(kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield self._answer(kwargs)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def test_round_trip_keeps_ids_roles_and_fields() -> None:
    messages = assert_round_trip(history(), "agno")
    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert messages[2].tool_calls[0].id == "c1" and messages[2].tool_calls[0].arguments == {
        "q": "a"
    }
    assert (messages[3].tool_call_id, messages[3].name) == ("c1", "lookup")


def test_dropping_old_turns_never_orphans_a_tool_round() -> None:
    assert_drop_keeps_whole_rounds(history())


def test_cut_keeps_name_ids_and_the_block_list() -> None:
    items = history()
    smaller = assert_cut_keeps_types(items)
    assert smaller[1].name == "ann" and smaller[1].id == items[1].id
    assert smaller[2] is items[2]  # the call has no long text
    assert smaller[3].tool_call_id == "c1" and "characters cut" in smaller[3].content
    assert smaller[4].content[0]["type"] == "text"


def test_summary_becomes_an_agno_message() -> None:
    assert_summary_is_native(history(), Message)


def test_the_wrapped_model_sends_less_and_the_session_keeps_everything() -> None:
    plain, wrapped = RecordingModel(), agno_compressed_model(RecordingModel(), DROP_OLD)
    assert type(wrapped).__name__ == "RecordingModel" and isinstance(wrapped, RecordingModel)
    stored = []
    for model in (plain, wrapped):
        agent = Agent(model=model, db=InMemoryDb(), add_history_to_context=True)
        for question in (f"first {PAD}", f"second {PAD}", "third"):
            assert agent.run(question, session_id="s").content == "done"
        stored.append([m.content for m in agent.get_chat_history(session_id="s")])
    assert len(wrapped.seen[-1]) < len(plain.seen[-1])
    assert stored[0] == stored[1] and len(stored[1]) == 6  # three questions, three answers
    assert wrapped.seen[-1][-1].content == "third"


async def test_the_async_and_stream_paths_compress_too() -> None:
    wrapped = agno_compressed_model(RecordingModel(), DROP_OLD)
    await wrapped.aresponse(messages=history())
    list(wrapped.response_stream(messages=history()))
    assert [len(s) for s in wrapped.seen] == [2, 2]


def test_the_example_builds_the_agent() -> None:
    agent = runpy.run_path(str(EXAMPLE))["agno_agent"](RecordingModel())
    assert agent.run("hi").content == "done"


def test_new_messages_render_as_agno_messages() -> None:
    assert render_read_back(AgnoConverter) == EXPECTED
