"""Crash-safe runs: a ticket order flow dies mid-run, and a new worker finishes it.

The card is charged once: the new worker replays the logged charge instead of calling it.
"""

import asyncio
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from agent_adapter import Executor, Flow, FlowRunner, Registry
from agent_adapter.errors import RecoveryNeedsDecision
from agent_adapter.exec.checkpoints import SQLiteCheckpointStore
from agent_adapter.flow import Call

CALLS: Counter[str] = Counter()
POWER_CUT = {"on": True}
registry = Registry()


class PowerCut(BaseException):
    """The worker process dies here (a stand-in for a killed container)."""


@registry.tool(idempotent=True)
def hold_seats(show: str, seats: int) -> str:
    """Hold seats for a show (holding twice is harmless)."""
    CALLS["hold_seats"] += 1
    return f"{seats} seats held for {show}"


@registry.tool
def charge_card(show: str, seats: int) -> str:
    """Charge the visitor's card (must never run twice)."""
    CALLS["charge_card"] += 1
    return f"charged {seats * 12} EUR"


@registry.tool
def email_tickets(show: str) -> str:
    """Email the tickets."""
    CALLS["email_tickets"] += 1
    if POWER_CUT["on"]:
        POWER_CUT["on"] = False
        raise PowerCut
    return f"tickets for {show} sent"


# --8<-- [start:flow]
class Order(BaseModel):
    show: str
    seats: int
    note: str = ""


def args(order: Order) -> dict[str, object]:
    return {"show": order.show, "seats": order.seats}


flow = (
    Flow("tickets", state=Order)
    .node("hold", Call("hold_seats", args=args, into="note"))
    .node("pay", Call("charge_card", args=args, into="note"))
    .node("send", Call("email_tickets", args=lambda o: {"show": o.show}, into="note"))
    .edge("hold", "pay")
    .edge("pay", "send")
    .end("send")
)
KEY = b"use-a-secret-from-your-vault-32b"  # the same key on every worker


def worker(path: Path) -> FlowRunner:
    store = SQLiteCheckpointStore(path)
    return FlowRunner(Executor(registry), checkpoints=store, checkpoint_key=KEY)


# --8<-- [end:flow]


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "runs.db"
        first = worker(path)
        try:
            ctx = first.executor.context(run_id="order-7")
            await first.run(flow, {"show": "Night at the museum", "seats": 2}, ctx)
        except PowerCut:
            print("worker 1 died while emailing the tickets")
        # --8<-- [start:recover]
        second = worker(path)  # a new process on startup
        try:
            await second.recover(flow, "order-7")
        except RecoveryNeedsDecision as exc:
            print("in doubt:", [tool for tool, _ in exc.calls])
        # email_tickets may have sent the mail: we checked, so it may run again
        out = await second.recover_all(flow, rerun_in_doubt=True)
        # --8<-- [end:recover]
        result = out["order-7"]
        print(result.status, "|", result.state.note)  # type: ignore[union-attr]
        print(dict(sorted(CALLS.items())))


if __name__ == "__main__":
    asyncio.run(main())
