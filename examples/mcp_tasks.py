"""MCP tasks: a confirm tool parks as a task; a host polls it by id until a reviewer decides."""

import asyncio

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    ToolError,
)
from agent_adapter.integrations.mcp import (
    TaskOptions,
    import_mcp,
    mcp_client,
    mcp_server,
)

remote = Registry()


@remote.tool(confirm=True)
async def send_invoice(customer_id: str, amount: float) -> str:
    """Send an invoice to a customer (needs approval)."""
    return f"invoiced {customer_id} for {amount:.2f} USD"


# --8<-- [start:serve]
store = MemoryApprovalStore()  # reviewers decide here (or over the HTTP server)
server = mcp_server(
    remote,
    executor=Executor(approval=DeferredApproval(store)),
    tasks=TaskOptions(threshold_s=5.0, poll_interval_s=0.05),
)
# --8<-- [end:serve]


async def review() -> None:
    """A reviewer approves the first pending call, from the store only."""
    while not (records := await store.list(status="pending")):
        await asyncio.sleep(0.01)
    await store.decide(records[0].approval_id, ApprovalDecision(approved=True, decided_by="ops"))


async def main() -> None:
    # --8<-- [start:import]
    async with mcp_client(server, tasks=True, task_wait_s=0.1) as client:
        tools = Registry()
        await import_mcp(tools, client)
        executor = Executor(tools)
        ctx = executor.context(run_id="run_demo")
        args = {"customer_id": "cus_1024", "amount": 49.5}
        try:
            await executor.run("send_invoice", args, ctx, call_id="call_demo")
        except ToolError as exc:  # still waiting after task_wait_s: the task id to resume
            print("pending", exc.details["remote_kind"], bool(exc.details["task_id"]))
        await review()
        print(await executor.run("send_invoice", args, ctx, call_id="call_demo"))
        # --8<-- [end:import]


if __name__ == "__main__":
    asyncio.run(main())
