"""run_tool_loop(on_event=): tool-call deltas, partial output and StreamDone reach the callback."""

from __future__ import annotations

import json
import runpy
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_adapter.core.messages import (
    LLMRequest,
    PartialOutput,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
)
from agent_adapter.exec import run_tool_loop
from agent_adapter.models.callable import CallableModel
from tests.exec.test_tool_loop_stream import USER, registry
from tests.fakes.models import ScriptedModel, calls, text

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "models_streaming.py"


async def test_on_event_gets_tool_deltas_and_done_and_on_text_the_text() -> None:
    model = ScriptedModel([calls(("add", {"a": 1, "b": 2}), content="adding"), text("3")])
    seen: list[StreamEvent] = []
    chunks: list[str] = []
    result = await run_tool_loop(model, registry, USER, on_text=chunks.append, on_event=seen.append)
    assert chunks == ["adding", "3"]
    kinds = [type(e).__name__ for e in seen]
    assert kinds == ["ToolCallDelta", "StreamDone", "StreamDone"]
    delta = seen[0]
    assert isinstance(delta, ToolCallDelta) and delta.name == "add"
    assert json.loads(delta.arguments_delta) == {"a": 1, "b": 2}
    assert result.final.text == "3"


async def test_on_event_alone_streams_and_an_async_callback_is_awaited() -> None:
    async def events(request: LLMRequest) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text='{"n": ')
        yield PartialOutput(value={})
        yield TextDelta(text="1}")
        yield PartialOutput(value={"n": 1})
        yield StreamDone(response=text('{"n": 1}'))

    model = CallableModel(lambda request: "unused", stream_fn=events)
    seen: list[StreamEvent] = []

    async def collect(event: StreamEvent) -> None:
        seen.append(event)

    result = await run_tool_loop(model, registry, USER, on_event=collect)
    assert [e.value for e in seen if isinstance(e, PartialOutput)] == [{}, {"n": 1}]
    assert not any(isinstance(e, TextDelta) for e in seen)
    assert isinstance(seen[-1], StreamDone) and result.final.text == '{"n": 1}'


async def test_without_callbacks_the_loop_does_not_stream() -> None:
    class NoStream(CallableModel):
        async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
            raise AssertionError("stream() must not be called")
            yield  # pragma: no cover

    result = await run_tool_loop(NoStream(lambda request: "hi"), registry, USER)
    assert result.final.text == "hi"


def test_the_streaming_example_prints_fragments_then_the_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        'tool args + {"sk',
        'tool args + u": "MU',
        'tool args + G-1"}',
        'answer so far {"sku": "MU"}',
        'answer so far {"sku": "MUG-1"}',
        'answer so far {"sku": "MUG-1", "in_stock": 12}',
        "final {'sku': 'MUG-1', 'in_stock': 12}",
    ]
