"""LlamaIndex: registry tools as FunctionTools (async `acall`, sync `call`)."""

import asyncio

from agent_adapter import Executor, Registry

registry = Registry()


@registry.tool
async def shipping_rate(item: str, region: str = "us") -> float:
    """Shipping rate for an item, in percent of the order value (by region)."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


async def main() -> None:
    # --8<-- [start:run]
    from agent_adapter.integrations.llamaindex import llamaindex_tools

    tools = llamaindex_tools(registry, executor=Executor())  # pass these to a LlamaIndex agent
    tool = tools[0]
    output = await tool.acall(item="tea")  # what an async agent does
    sync_output = tool.call(item="tea", region="eu")  # sync agents: run_sync bridge
    bad = await tool.acall(item=3)  # invalid args come back as an error value
    # --8<-- [end:run]
    print(tool.metadata.to_openai_tool()["function"]["name"])
    print(output.raw_output, sync_output.raw_output)
    print(bad.raw_output["error"]["kind"])


if __name__ == "__main__":
    asyncio.run(main())
