"""Serve registry tools over MCP: an in-process demo, or `--stdio` for a real MCP host."""

import asyncio
import sys

from agent_adapter import Registry
from agent_adapter.integrations.mcp import mcp_server

registry = Registry()


# --8<-- [start:tools]
@registry.tool(effects="read_only", idempotent=True)
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


server = mcp_server(registry, name="shop-tools")
# --8<-- [end:tools]


# --8<-- [start:demo]
async def demo() -> None:
    from mcp import Client
    from mcp.types import TextContent

    async with Client(server) as client:  # in-process: no sockets, full MCP semantics
        for tool in (await client.list_tools()).tools:
            print(tool.name, tool.annotations)
        rate = await client.call_tool("shipping_rate", {"region": "us"})
        print(rate.structured_content)  # {'result': 18.0}
        filed = await client.call_tool("send_invoice", {"customer_id": "cus_1024", "amount": 49.5})
        text = [block.text for block in filed.content if isinstance(block, TextContent)]
        print(filed.is_error, *text)  # no approval port configured: denied


# --8<-- [end:demo]


# --8<-- [start:stdio]
async def serve_stdio() -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


# --8<-- [end:stdio]


if __name__ == "__main__":
    asyncio.run(serve_stdio() if "--stdio" in sys.argv[1:] else demo())
