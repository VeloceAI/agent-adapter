"""Streaming: tool-call arguments and structured output as they arrive, through the tool loop."""

import asyncio
import json
from collections.abc import AsyncIterator

from agent_adapter import (
    CallableModel,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.models import (
    OutputSchema,
    PartialOutput,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
)

registry = Registry()


@registry.tool
def stock(sku: str) -> int:
    """Units of a product in stock."""
    return {"MUG-1": 12}.get(sku, 0)


async def fake_stream(request: LLMRequest) -> AsyncIterator[StreamEvent]:
    """Stands in for a provider: a tool call in fragments, then a JSON answer in fragments."""
    if request.messages[-1].role != "tool":
        for piece in ('{"sk', 'u": "MU', 'G-1"}'):
            yield ToolCallDelta(index=0, id="call_1", name="stock", arguments_delta=piece)
        call = ToolCall(id="call_1", name="stock", arguments={"sku": "MUG-1"})
        message = Message(role="assistant", tool_calls=(call,))
        reply = LLMResponse(
            provider="fake",
            model="shop-1",
            tool_calls=(call,),
            finish_reason="tool_calls",
            message=message,
        )
        yield StreamDone(response=reply)
        return
    answer = '{"sku": "MUG-1", "in_stock": 12}'
    for piece in ('{"sku": "MU', 'G-1", "in_st', 'ock": 12}'):
        yield TextDelta(text=piece)
    reply = LLMResponse(
        provider="fake",
        model="shop-1",
        text=answer,
        parsed=json.loads(answer),
        message=Message(role="assistant", content=answer),
    )
    yield StreamDone(response=reply)


# --8<-- [start:events]
def on_event(event: StreamEvent) -> None:
    """Print each tool-call fragment and each partial answer as it streams in."""
    if isinstance(event, ToolCallDelta) and event.arguments_delta:
        print("tool args +", event.arguments_delta)
    elif isinstance(event, PartialOutput):
        print("answer so far", json.dumps(event.value))


async def main() -> object:
    model = CallableModel(lambda request: "unused", provider="fake", stream_fn=fake_stream)
    schema = OutputSchema(name="Stock", json_schema={"type": "object"})
    result = await run_tool_loop(
        model,
        registry,
        [Message(role="user", content="How many MUG-1 are left?")],
        output_schema=schema,
        on_event=on_event,
    )
    return result.final.parsed


# --8<-- [end:events]


if __name__ == "__main__":
    print("final", asyncio.run(main()))
