"""Our side of every "Connect anywhere" recipe: one tool server, local or public.

Run as a script it replays a remote host offline (MCP and REST against the ASGI app, no
network). `--serve` reads AA_PUBLIC_HOST, AA_AGENT_TOKEN and AA_REVIEWER_TOKEN and runs the
public app on 0.0.0.0:8000; put TLS in front of it.
"""

import asyncio
import json
import os
import sys
from typing import Any, cast

from agent_adapter import Registry
from agent_adapter.server import APPROVER_SCOPE, Principal, bearer_auth

registry = Registry()


@registry.tool(effects="read_only", idempotent=True, tags=["hosted"])
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True, effects="destructive")
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


# --8<-- [start:local]
def local_app() -> Any:
    """MCP only, no auth, loopback peers only: a framework on this machine, in development."""
    from agent_adapter.integrations.mcp import streamable_http_app

    return streamable_http_app(registry)  # http://127.0.0.1:8000/mcp; remote peers get 403


# --8<-- [end:local]


# --8<-- [start:public]
def public_app(tools: Registry, host: str, agent_token: str, reviewer_token: str) -> Any:
    """REST at /, MCP at /mcp, one bearer check for both: for a host on the internet."""
    from mcp.server.transport_security import TransportSecuritySettings

    from agent_adapter.server import create_app

    tokens = {
        agent_token: Principal("hosted-agent"),  # the token the remote framework sends
        reviewer_token: Principal("reviewer@example.com", scopes=frozenset({APPROVER_SCOPE})),
    }
    security = TransportSecuritySettings(allowed_hosts=[host], allowed_origins=[f"https://{host}"])
    return create_app(
        tools,
        authenticate=bearer_auth(tokens),
        host="0.0.0.0",  # a public bind: the public-host checks apply
        allowed_hosts=[host],  # REST: any other Host header gets 421
        mcp_options={"transport_security": security},  # MCP: the same rule
    )


# --8<-- [end:public]

# --8<-- [start:hosted]
hosted = registry.subset(tags=["hosted"])  # no confirm tools: hosted runtimes cannot resume them
# --8<-- [end:hosted]


# --8<-- [start:realtime]
async def park_realtime_approval(
    item: dict[str, Any], rest: Any, session_id: str
) -> tuple[str, dict[str, Any]]:
    """OpenAI's `mcp_approval_request` -> one of our approvals; OpenAI never runs the call."""
    invoked = await rest.post(  # `rest`: an httpx client with base_url and the agent token
        f"/tools/{item['name']}/invoke",
        json={"arguments": json.loads(item["arguments"]), "run_id": session_id},
        headers={"Idempotency-Key": item["id"]},  # a replayed event parks nothing new
    )
    approval_id = invoked.json()["approval_id"]
    return approval_id, {  # send it on the Realtime socket: the model hears "sent for review"
        "type": "conversation.item.create",
        "item": {
            "type": "mcp_approval_response",
            "approval_request_id": item["id"],
            "approve": False,
            "reason": f"sent to a human reviewer as {approval_id}; it runs once they approve",
        },
    }


def realtime_outcome(approval_id: str, decided: dict[str, Any]) -> list[dict[str, Any]]:
    """The reviewer's decision (the body `/approvals/{id}/decide` returns), for the model."""
    said = f"Approval {approval_id}: {decided['status']}, result {decided.get('result')}"
    message = {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": said}],
    }
    return [{"type": "conversation.item.create", "item": message}, {"type": "response.create"}]


# --8<-- [end:realtime]


HOST = "tools.example.com"
AGENT, REVIEWER = "agent-test-token", "reviewer-test-token"  # demo only: never ship tokens


async def demo() -> None:
    """A remote MCP host, then a REST client, against the public app in-process."""
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import CallToolResult, RequestParamsMeta, TextContent

    app = public_app(registry, HOST, AGENT, REVIEWER)

    def http(token: str | None) -> httpx2.AsyncClient:
        peer = httpx2.ASGITransport(app=app, client=("203.0.113.9", 5000))
        auth = {"Authorization": f"Bearer {token}"} if token else {}
        return httpx2.AsyncClient(transport=peer, base_url=f"https://{HOST}", headers=auth)

    def mcp_host() -> Client:
        return Client(streamable_http_client(f"https://{HOST}/mcp", http_client=http(AGENT)))

    def text(result: CallToolResult) -> str:
        return "".join(part.text for part in result.content if isinstance(part, TextContent))

    args = {"customer_id": "cus_1024", "amount": 49.5}
    async with app.router.lifespan_context(app):
        async with mcp_host() as client:
            print([tool.name for tool in (await client.list_tools()).tools])
            print((await client.call_tool("shipping_rate", {"region": "us"})).structured_content)
            parked = await client.call_tool("send_invoice", args)
        meta = parked.meta or {}
        approval = meta["agent-adapter/approval_id"]
        keys = {key: meta[key] for key in ("agent-adapter/run_id", "agent-adapter/call_id")}
        resume = cast("RequestParamsMeta", keys)  # what `import_mcp` sends back for you
        print(parked.is_error, text(parked).startswith("approval required: apr_"))
        async with http(REVIEWER) as reviewer:
            decided = await reviewer.post(f"/approvals/{approval}/decide", json={"approved": True})
        print(decided.status_code, decided.json()["status"])
        async with mcp_host() as client:
            again = await client.call_tool("send_invoice", args)  # no resume keys: a new approval
            print(text(again).startswith("approval required: apr_"))
            print(text(await client.call_tool("send_invoice", args, meta=resume)))
        async with http(AGENT) as agent:
            invoked = await agent.post("/tools/send_invoice/invoke", json={"arguments": args})
            body = invoked.json()
            print(invoked.status_code, body["status"])
            async with http(REVIEWER) as reviewer:
                url = f"/approvals/{body['approval_id']}/decide"
                print((await reviewer.post(url, json={"approved": True})).json()["result"])
            print((await agent.post("/mcp", json={}, headers={"Host": "evil.example"})).status_code)
        async with http(None) as anonymous:
            print((await anonymous.get("/tools")).status_code)
    print(hosted.names())


def serve() -> None:
    """Run the public app; every secret comes from the environment."""
    from agent_adapter.server import serve as run

    env = os.environ
    app = public_app(
        registry, env["AA_PUBLIC_HOST"], env["AA_AGENT_TOKEN"], env["AA_REVIEWER_TOKEN"]
    )
    run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        serve()
    else:
        asyncio.run(demo())
