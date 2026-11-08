"""DBOS: a support-desk refund flow as a DBOS workflow on SQLite; a desk lead approves it.

The decision goes into the approval store; a DBOS message only wakes the workflow.

Each superstep runs as a DBOS step through the executor. Fully offline; without dbos
installed it says so and stops.
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


async def main() -> None:
    try:
        from dbos import DBOS
    except ImportError:
        print('dbos is not installed: pip install "agent-adapter[dbos]"')
        return
    from agent_adapter.integrations.dbos import DbosFlows, DurableFlows

    with tempfile.TemporaryDirectory() as folder:
        # --8<-- [start:setup]
        store = SQLiteCheckpointStore(Path(folder) / "runs.db")
        approvals = MemoryApprovalStore()  # shared with your review screen in production
        executor = Executor(registry, approval=DeferredApproval(approvals))
        runner = FlowRunner(executor, checkpoints=store, checkpoint_key=KEY)
        url = f"sqlite:///{(Path(folder) / 'dbos.sqlite').as_posix()}"
        DBOS(config={"name": "refund_desk", "system_database_url": url})
        refunds = DbosFlows(DurableFlows(runner, [flow]))  # before DBOS.launch()
        DBOS.launch()
        # --8<-- [end:setup]
        try:
            # --8<-- [start:run]
            handle = await refunds.start(
                "refunds", {"ticket": "t-17", "amount": 40.0}, run_id="t-17"
            )
            waiting = await refunds.pending("t-17", timeout_s=10)
            print("waiting for", waiting.node if waiting else None)
            if waiting is not None:
                decision = ApprovalDecision(approved=True, decided_by="desk-lead")
                await approvals.decide(waiting.approval_id, decision)
                await refunds.notify("t-17", waiting.approval_id)  # a wake-up, not a verdict
            report = await refunds.result(handle)
            # --8<-- [end:run]
        finally:
            DBOS.destroy(destroy_registry=True)
    print(report.status, "|", report.state["note"] if report.state else "")


if __name__ == "__main__":
    asyncio.run(main())
