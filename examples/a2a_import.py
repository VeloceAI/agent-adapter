"""Import a remote A2A agent's skills as tools; a remote approval is a ToolError, never a yes."""

import asyncio

import httpx

from agent_adapter import DeferredApproval, Executor, MemoryApprovalStore, Registry, ToolError
from agent_adapter.integrations.a2a import a2a_app, connect_a2a, import_a2a, static_bearer

URL = "http://tools.internal.example"

# The remote agent (another team's service), served in-process for this demo.
remote = Registry()


@remote.tool(effects="read_only")
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@remote.tool(confirm=True)
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


app = a2a_app(
    remote,
    url=f"{URL}/",
    executor=Executor(approval=DeferredApproval(MemoryApprovalStore())),
    authenticate=static_bearer({"s3cret-token": "agent-team"}),
    allowed_hosts=["tools.internal.example"],
)


async def main() -> None:
    async with app.router.lifespan_context(app):
        offline = httpx.ASGITransport(app=app)  # the demo serves the agent in-process
        # --8<-- [start:import]
        connection = await connect_a2a(
            URL,  # the card is read from URL + "/.well-known/agent-card.json"
            headers={"Authorization": "Bearer s3cret-token"},  # same origin only
            trusted_hosts=["tools.internal.example"],  # a private host you trust
            http_transport=offline,  # omit in real use
        )
        async with connection:
            tools = Registry()
            await import_a2a(tools, connection, prefix="shop_")
            print(tools.names())  # ['shop_shipping_rate', 'shop_send_invoice']
            executor = Executor(tools)
            print(await executor.run("shop_shipping_rate", {"region": "eu"}))  # 5.0
            try:
                await executor.run("shop_send_invoice", {"customer_id": "cus_1024", "amount": 49.5})
            except ToolError as exc:  # the remote task waits for the remote reviewers
                print(exc.details["state"])  # input-required
        # --8<-- [end:import]


if __name__ == "__main__":
    asyncio.run(main())
