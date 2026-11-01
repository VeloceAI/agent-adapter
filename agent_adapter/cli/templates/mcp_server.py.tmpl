"""Starter MCP server: registry tools over stdio for any MCP host. Run: python server.py"""

import asyncio

from mcp.server.stdio import stdio_server

from agent_adapter import Executor, Registry
from agent_adapter.integrations.mcp import mcp_server

registry = Registry()


@registry.tool(effects="read_only", idempotent=True)
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


server = mcp_server(registry, executor=Executor(registry), name="my-tools")


async def main() -> None:
    """Serve over stdin/stdout until the host closes the pipe."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
