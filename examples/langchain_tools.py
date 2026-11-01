"""Export registry tools to LangChain and import a LangChain tool into a registry."""

import asyncio
from typing import Any

from langchain_core.tools import tool

from agent_adapter import Executor, Registry, ToolContext

registry = Registry()


@registry.tool
def shipping_rate(region: str, ctx: ToolContext) -> dict[str, Any]:
    """Shipping rate for a region, in percent of the order value."""
    rate = {"eu": 5.0, "us": 18.0}.get(region, 12.0)
    return {"region": region, "rate": rate, "call_id": ctx.call_id}


@tool
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


async def main() -> None:
    # --8<-- [start:export]
    from agent_adapter.integrations.langchain import structured_tools

    executor = Executor(registry)  # hooks, approval, cache, ledger apply to every call
    tools = structured_tools(registry, executor=executor)  # pass to bind_tools / an agent
    call = {"name": "shipping_rate", "args": {"region": "us"}, "id": "call_1", "type": "tool_call"}
    message = await tools[0].ainvoke(call)  # a ToolMessage; ctx.call_id == "call_1"
    # --8<-- [end:export]
    print(message.content)
    # --8<-- [start:import]
    from agent_adapter.integrations.langchain import import_langchain

    [spec] = import_langchain(registry, [word_count], prefix="lc_")
    count = await executor.run("lc_word_count", {"text": "one two three"})
    # --8<-- [end:import]
    print(spec.name, count)


if __name__ == "__main__":
    asyncio.run(main())
