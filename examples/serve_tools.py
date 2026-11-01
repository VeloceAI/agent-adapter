"""The HTTP tool server: invoke, approve and decide over REST (`--serve` runs uvicorn)."""

import asyncio
import json
import sys
from typing import Any

from agent_adapter import Registry
from agent_adapter.approval import ApprovalRecord
from agent_adapter.server import (
    APPROVER_SCOPE,
    Principal,
    RequestInfo,
    ToolService,
    bearer_auth,
    is_invoker,
)

registry = Registry()


@registry.tool(effects="read_only", idempotent=True)
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True, effects="destructive")
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


# --8<-- [start:auth]
TOKENS = {
    "dev-token": Principal("dev@example.com"),
    "reviewer-token": Principal("reviewer@example.com", scopes=frozenset({APPROVER_SCOPE})),
}
options: dict[str, Any] = {"authenticate": bearer_auth(TOKENS)}
# --8<-- [end:auth]


# --8<-- [start:four_eyes]
async def four_eyes(principal: Principal, record: ApprovalRecord) -> bool:
    """Only a reviewer who did not make the call may decide it."""
    return APPROVER_SCOPE in principal.scopes and not is_invoker(principal, record)


options["authorize_decision"] = four_eyes
# --8<-- [end:four_eyes]


# --8<-- [start:app]
def make_app() -> Any:
    from agent_adapter.server import create_app  # needs agent-adapter[server]

    return create_app(registry, **options)  # MCP at /mcp too (needs agent-adapter[mcp])


# --8<-- [end:app]


async def demo() -> None:
    service = ToolService(registry, **options)

    async def send(method: str, path: str, token: str, body: Any = None) -> Any:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        info = RequestInfo(method, path, headers, {}, None)
        raw = json.dumps(body).encode() if body is not None else b""
        reply = await service.handle(info, raw)
        body = reply.body
        print(reply.status, body.get("result", body.get("detail", body.get("status"))))
        return body

    rate = {"arguments": {"region": "us"}}
    await send("POST", "/tools/shipping_rate/invoke", "dev-token", rate)
    invoice = {"arguments": {"customer_id": "cus_1024", "amount": 49.5}, "run_id": "invoice-1"}
    parked = await send("POST", "/tools/send_invoice/invoke", "dev-token", invoice)
    approval = f"/approvals/{parked['approval_id']}"
    await send("GET", approval, "dev-token")  # the invoker sees its own approval
    await send("POST", f"{approval}/decide", "reviewer-token", {"approved": True})
    await send(
        "POST", "/tools/shipping_rate/invoke", "reviewer-token", {**rate, "run_id": "invoice-1"}
    )


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        from agent_adapter.server import serve

        serve(make_app())
    else:
        asyncio.run(demo())
