"""run_tool_loop(on_text=): text chunks stream through the port's stream(); default unchanged."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_adapter import Registry
from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import LLMRequest, Message, StreamDone, StreamEvent, TextDelta
from agent_adapter.core.types import Usage
from agent_adapter.exec import Executor, run_tool_loop
from agent_adapter.models.callable import CallableModel
from tests.fakes.models import ScriptedModel, calls, text

registry = Registry()


@registry.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


USER = (Message(role="user", content="go"),)


async def test_on_text_gets_each_chunk_and_tools_still_run() -> None:
    model = ScriptedModel([calls(("add", {"a": 1, "b": 2}), content="let me add"), text("3")])
    chunks: list[str] = []
    result = await run_tool_loop(model, registry, USER, on_text=chunks.append)
    assert chunks == ["let me add", "3"]
    assert (result.finish, result.final.text, result.steps) == ("stop", "3", 2)
    assert [m.content for m in result.messages if m.role == "tool"] == ["3"]


async def test_an_async_on_text_is_awaited_and_split_deltas_arrive_in_order() -> None:
    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="he")
        yield TextDelta(text="")
        yield TextDelta(text="llo")
        yield StreamDone(response=text("hello", usage=Usage(input_tokens=2, output_tokens=1)))

    model = CallableModel(lambda request: "unused", stream_fn=events)
    seen: list[str] = []

    async def collect(chunk: str) -> None:
        seen.append(chunk)

    result = await run_tool_loop(model, registry, USER, on_text=collect)
    assert seen == ["he", "llo"] and result.final.text == "hello"
    assert result.usage == Usage(input_tokens=2, output_tokens=1)


async def test_a_stream_without_done_is_a_model_error() -> None:
    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="half")

    class Bare:
        provider, model = "bare", "m"

        async def generate(self, request: LLMRequest) -> None:
            raise AssertionError("generate is not used when streaming")

        def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
            return events(request)

        def cache_key_fields(self) -> dict[str, str]:
            return {}

    with pytest.raises(ModelOutputInvalid, match="StreamDone"):
        await run_tool_loop(Bare(), registry, USER, on_text=lambda chunk: None)  # type: ignore[arg-type]


async def test_without_on_text_the_loop_calls_generate_not_stream() -> None:
    class GenerateOnly(ScriptedModel):
        async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
            raise AssertionError("stream must not be called")
            yield  # pragma: no cover

    model = GenerateOnly([text("plain")])
    result = await run_tool_loop(model, registry, USER, executor=Executor(registry))
    assert result.final.text == "plain"
