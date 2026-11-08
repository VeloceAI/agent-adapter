"""Temporal: a support-desk refund flow as a workflow; a desk lead approves it in the store.

Each superstep runs as an activity through the executor. Runs on a local Temporal dev
server (downloaded on first use); without temporalio installed it says so and stops.
"""

import asyncio
import tempfile
from pathlib import Path

from pydantic import BaseModel

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    Flow,
    FlowRunner,
    MemoryApprovalStore,
    Registry,
)
from agent_adapter.exec.checkpoints import SQLiteCheckpointStore
from agent_adapter.flow import Call

registry = Registry()


@registry.tool
def refund(ticket: str, amount: float) -> str:
    """Refund a support ticket's order."""
    return f"refunded {amount:.2f} EUR on {ticket}"


# --8<-- [start:flow]
class Ticket(BaseModel):
    ticket: str
    amount: float
    note: str = ""


flow = (
    Flow("refunds", state=Ticket)
    .node("review", lambda t: {"note": f"refund {t.amount:.2f} asked"}, pause=True)
    .node(
        "pay", Call("refund", args=lambda t: {"ticket": t.ticket, "amount": t.amount}, into="note")
    )
    .edge("review", "pay")
    .end("pay")
)
KEY = b"use-a-secret-from-your-vault-32b"  # the same key on every worker
# --8<-- [end:flow]


async def main() -> None:
    try:
        from temporalio.testing import WorkflowEnvironment
    except ImportError:
        print('temporalio is not installed: pip install "agent-adapter[temporal]"')
        return
    # --8<-- [start:worker]
    from agent_adapter.integrations.temporal import (
        DurableFlows,
        flow_worker,
        notify,
        pending,
        result,
        start_flow,
    )

    with tempfile.TemporaryDirectory() as folder:
        store = SQLiteCheckpointStore(Path(folder) / "runs.db")
        approvals = MemoryApprovalStore()  # shared with your review screen in production
        executor = Executor(registry, approval=DeferredApproval(approvals))
        runner = FlowRunner(executor, checkpoints=store, checkpoint_key=KEY)
        flows = DurableFlows(runner, [flow])
        async with (
            await WorkflowEnvironment.start_local() as env,
            flow_worker(env.client, flows, task_queue="refunds"),
        ):
            handle = await start_flow(
                env.client,
                "refunds",
                {"ticket": "t-17", "amount": 40.0},
                run_id="refund-t-17",
                task_queue="refunds",
            )
            # --8<-- [end:worker]
            # --8<-- [start:decide]
            waiting = await pending(handle)
            while waiting is None:
                await asyncio.sleep(0.1)
                waiting = await pending(handle)
            print("waiting for", waiting.node)
            # the decision lives in the approval store; the update only wakes the run
            decision = ApprovalDecision(approved=True, decided_by="desk-lead")
            await approvals.decide(waiting.approval_id, decision)
            await notify(handle, waiting.approval_id)
            report = await result(handle)
            # --8<-- [end:decide]
        print(report.status, "|", report.state["note"] if report.state else "")


if __name__ == "__main__":
    asyncio.run(main())
