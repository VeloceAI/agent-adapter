"""The tool server every client in this folder calls: one read-only tool behind a bearer token."""

import os

from mcp.server.transport_security import TransportSecuritySettings

from agent_adapter import Registry
from agent_adapter.server import Principal, bearer_auth, create_app, serve

HOST = os.environ.get("AA_HOST", "server:8000")  # the Host header clients send; others get 421
TOKEN = os.environ.get("AA_TOKEN", "dev-token")

registry = Registry()


@registry.tool(effects="read_only", idempotent=True)
def get_weather(city: str) -> str:
    """Today's weather for a city."""
    return f"Sunny in {city}, 21 C"


app = create_app(
    registry,
    authenticate=bearer_auth({TOKEN: Principal("client")}),
    host="0.0.0.0",
    allowed_hosts=[HOST],
    mcp_options={
        "transport_security": TransportSecuritySettings(allowed_hosts=[HOST], allowed_origins=[])
    },
)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
