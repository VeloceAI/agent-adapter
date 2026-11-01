"""Migration: from a hand-rolled "needs approval" dict to confirm= and DeferredApproval."""

import asyncio
from typing import Any

from agent_adapter import (
    Decision,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    ToolCall,
)

refunds: list[float] = []


# --8<-- [start:before]
PENDING: dict[str, dict[str, Any]] = {}  # ticket id -> arguments
APPROVED: set[str] = set()


def refund_before(ticket: str, order_id: str, amount: float) -> str:
    if amount > 100 and ticket not in APPROVED:
        PENDING[ticket] = {"order_id": order_id, "amount": amount}
        return "needs approval"
    refunds.append(amount)  # runs again on every replay, with whatever amount it is given
    return f"refunded {amount}"


# --8<-- [end:before]

# --8<-- [start:after]
registry = Registry()


@registry.tool(confirm=lambda call, ctx: float(call.arguments["amount"]) > 100)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    refunds.append(amount)
    return f"refunded {amount}"


store = MemoryApprovalStore()  # SQLiteApprovalStore / RedisApprovalStore in production
executor = Executor(registry, approval=DeferredApproval(store))
# --8<-- [end:after]


async def main() -> None:
    # --8<-- [start:bugs]
    print(refund_before("t1", "o-1", 500))
    APPROVED.add("t1")
    print(refund_before("t1", "o-1", 500), refund_before("t1", "o-1", 5000))  # replay, bigger
    # --8<-- [end:bugs]
    refunds.clear()
    # --8<-- [start:fixed]
    ctx = executor.context(run_id="run_t1")
    call = ToolCall(name="refund", arguments={"order_id": "o-1", "amount": 500}, id="c1")
    first = await executor.run_call(call, ctx)
    approval_id = first.error.details["approval_id"] if first.error else ""
    await store.decide(approval_id, Decision(approved=True, decided_by="ana"))
    print(first.status, (await executor.run_call(call, ctx)).value)
    print((await executor.run_call(call, ctx)).status)  # replay: a new review, no refund
    bigger = ToolCall(name="refund", arguments={"order_id": "o-1", "amount": 5000}, id="c1")
    swapped = await executor.run_call(bigger, ctx)
    print(swapped.error.kind if swapped.error else "", refunds)  # refused, one refund
    # --8<-- [end:fixed]


if __name__ == "__main__":
    asyncio.run(main())
