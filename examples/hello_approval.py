"""Human approval in a few lines: park a call in SQLite, resume it by its approval id."""

import tempfile
from pathlib import Path

import agent_adapter as aa


# --8<-- [start:tool]
@aa.tool(confirm=True)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


# --8<-- [end:tool]


def main(folder: Path) -> None:
    # --8<-- [start:flow]
    executor = aa.configure(approval=aa.DeferredApproval.sqlite(folder / "approvals.db"))
    try:
        aa.call(refund, order_id="o-7", amount=500)
    except aa.ApprovalPending as pending:
        approval_id = pending.approval_id  # store it, show it to a reviewer
    # ... later, in any process that opens the same file:
    result = executor.resume_sync(approval_id, True, decided_by="ana")
    print(result.value)  # runs once; a second resume is refused
    # --8<-- [end:flow]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        main(Path(folder))
