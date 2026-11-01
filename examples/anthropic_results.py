"""Answer Anthropic tool_use blocks with tool_result blocks."""

import asyncio
import json
from typing import Any

from agent_adapter import Executor, Registry, ToolCall, ToolError
from agent_adapter.integrations.anthropic import tool_result_blocks

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    if region != "us":
        raise ToolError(f"no rate for {region!r}", tool="shipping_rate")
    return 18.0


async def main() -> None:
    # The model's tool_use blocks (SDK objects work the same through their attributes).
    tool_uses: list[dict[str, Any]] = [
        {"type": "tool_use", "id": "toolu_1", "name": "shipping_rate", "input": {"region": "us"}},
        {"type": "tool_use", "id": "toolu_2", "name": "shipping_rate", "input": {"region": "mars"}},
    ]
    # --8<-- [start:results]
    executor = Executor(registry)
    calls = [ToolCall(name=u["name"], arguments=u["input"], id=u["id"]) for u in tool_uses]
    results = await executor.run_many(calls)  # one ToolResult per call, errors in place
    reply = {"role": "user", "content": tool_result_blocks(results)}
    # --8<-- [end:results]
    print(json.dumps(reply, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
