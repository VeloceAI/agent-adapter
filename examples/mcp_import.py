"""Import an MCP server's tools, and resume a remote approval with the same run and call ids."""

import asyncio

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    ToolError,
)
from agent_adapter.integrations.mcp import import_mcp, mcp_client, mcp_server

remote = Registry()


@remote.tool(effects="read_only")
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@remote.tool(confirm=True)
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


# The remote side parks approvals in a store that its reviewers decide on.
store = MemoryApprovalStore()
server = mcp_server(remote, executor=Executor(approval=DeferredApproval(store)))


async def main() -> None:
    # --8<-- [start:import]
    async with mcp_client(server) as client:  # or mcp_client("https://host/mcp", headers=...)
        tools = Registry()
        await import_mcp(tools, client, prefix="shop_")
        print(tools.names())  # ['shop_shipping_rate', 'shop_send_invoice']
        print(tools.get("shop_shipping_rate").effects)  # read_only, from the remote hints
        executor = Executor(tools)
        print(await executor.run("shop_shipping_rate", {"region": "eu"}))  # 5.0
        # --8<-- [end:import]
        # --8<-- [start:resume]
        ctx = executor.context(run_id="run_demo")
        args = {"customer_id": "cus_1024", "amount": 49.5}
        try:
            await executor.run("shop_send_invoice", args, ctx, call_id="call_demo")
        except ToolError as exc:  # the remote result: "approval required: apr_..."
            approval_id = exc.details["approval_id"]
            print("pending", approval_id.startswith("apr_"))
        await store.decide(approval_id, ApprovalDecision(approved=True, decided_by="ops"))
        print(await executor.run("shop_send_invoice", args, ctx, call_id="call_demo"))
        # --8<-- [end:resume]


if __name__ == "__main__":
    asyncio.run(main())
