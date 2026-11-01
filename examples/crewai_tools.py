"""CrewAI: registry tools as BaseTools (sync `run` via run_sync, async `arun`)."""

import asyncio

from agent_adapter import Executor, Registry

registry = Registry()


@registry.tool
async def shipping_rate(item: str, region: str = "us") -> float:
    """Shipping rate for an item, in percent of the order value (by region)."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


async def main() -> None:
    # --8<-- [start:run]
    from agent_adapter.integrations.crewai import crewai_tools

    tools = crewai_tools(registry, executor=Executor())  # Agent(role=..., tools=tools)
    tool = tools[0]
    value = await tool.arun(item="book")  # async path
    sync_value = tool.run(item="book", region="eu")  # sync path, even inside a running loop
    # --8<-- [end:run]
    print(tool.name, tool.description)
    print(value, sync_value)


if __name__ == "__main__":
    asyncio.run(main())
