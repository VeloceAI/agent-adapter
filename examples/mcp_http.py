"""The universal export: tools as a stateless Streamable HTTP MCP app (`--serve` runs it)."""

import sys

from agent_adapter import Registry
from agent_adapter.integrations.mcp import streamable_http_app

registry = Registry()


@registry.tool(effects="read_only", idempotent=True)
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


# --8<-- [start:app]
app = streamable_http_app(registry)  # stateless, JSON responses, served at /mcp


def serve() -> None:
    import uvicorn  # installed with the mcp extra

    uvicorn.run(app, host="127.0.0.1", port=8000)  # http://127.0.0.1:8000/mcp


# --8<-- [end:app]


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        serve()
    else:
        print([route.path for route in app.routes])
