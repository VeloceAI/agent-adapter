"""Write the OpenAPI 3.0.3 documents: one for Go/Rust generators, one for an API gateway.

Run as a script it prints the gateway document's version, operations and whether any
`oneOf`/`anyOf`/`allOf` is left. No network.
"""

import json
from typing import Any

from pydantic import BaseModel

from agent_adapter import Registry
from agent_adapter.server import openapi_document

registry = Registry()


class Refund(BaseModel):
    order_id: str
    reason: str | None = None  # 3.1: anyOf [string, null]; 3.0: nullable


@registry.tool(effects="read_only", idempotent=True)
async def refund_status(refund: Refund) -> str:
    """Status of a refund request."""
    return "approved"


# --8<-- [start:dialects]
generators = openapi_document(registry, title="shop-tools", version="1.0", dialect="3.0")
gateway = openapi_document(
    registry,
    title="shop-tools",
    version="1.0",
    server_url="https://tools.example.com",
    dialect="3.0-gateway",  # invoke operations only, no auth blocks
)
inline_payload = json.dumps(gateway)  # paste inline, or upload to S3
# --8<-- [end:dialects]


def unions(node: Any) -> int:
    """How many oneOf/anyOf/allOf keys `node` holds."""
    if isinstance(node, dict):
        here = sum(key in node for key in ("oneOf", "anyOf", "allOf"))
        return here + sum(unions(value) for value in node.values())
    if isinstance(node, list):
        return sum(unions(item) for item in node)
    return 0


if __name__ == "__main__":
    print(gateway["openapi"], sorted(gateway["paths"]))
    print(unions(generators), unions(gateway))
