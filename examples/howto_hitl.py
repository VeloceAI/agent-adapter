"""Human in the loop: pause a call for review, resume it, edit it, deny it, use it once."""

import asyncio

from agent_adapter import (
    Decision,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.errors import ApprovalAlreadyDecided

registry = Registry()
paid: list[float] = []


# --8<-- [start:tool]
def big_refund(call: ToolCall, ctx: ToolContext) -> bool:
    return float(call.arguments["amount"]) > 100  # small refunds run without review


@registry.tool(confirm=big_refund, effects="destructive")
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    paid.append(amount)
    return f"refunded {amount} on {order_id}"


# --8<-- [end:tool]

# --8<-- [start:setup]
store = MemoryApprovalStore()  # use a database-backed ApprovalStore across processes
executor = Executor(registry, approval=DeferredApproval(store))
ctx = executor.context(run_id="run_support_42")  # keep the run id to resume later
# --8<-- [end:setup]


async def pause_and_resume() -> None:
    # --8<-- [start:pause]
    call = ToolCall(name="refund", arguments={"order_id": "o-7", "amount": 500}, id="call_1")
    first = await executor.run_call(call, ctx)
    print(first.status, first.error.kind if first.error else "")  # pending, nothing ran
    # --8<-- [end:pause]

    # --8<-- [start:review]
    [waiting] = await store.list(run_id=ctx.run_id, status="pending")
    print(waiting.request.tool, waiting.request.arguments)  # what the reviewer sees
    edit = Decision(approved=True, edited_args={"order_id": "o-7", "amount": 450}, decided_by="ana")
    await store.decide(waiting.approval_id, edit)
    try:
        await store.decide(waiting.approval_id, Decision(approved=False, decided_by="bob"))
    except ApprovalAlreadyDecided:
        print("first decision wins")
    # --8<-- [end:review]

    # --8<-- [start:resume]
    resumed = await executor.run_call(call, ctx)  # same run, same call id
    print(resumed.status, resumed.value)  # runs once, with the edited amount
    # --8<-- [end:resume]

    # --8<-- [start:one_use]
    print((await store.get(waiting.approval_id)).consumed)  # the decision is spent
    again = await executor.run_call(call, ctx)  # a replay of the same call ...
    new_id = again.error.details["approval_id"] if again.error else ""
    print(again.status, new_id != waiting.approval_id)  # ... is a new review, not a rerun
    # --8<-- [end:one_use]


async def deny() -> None:
    # --8<-- [start:deny]
    call = ToolCall(name="refund", arguments={"order_id": "o-9", "amount": 900}, id="call_2")
    pending = await executor.run_call(call, ctx)
    approval_id = pending.error.details["approval_id"] if pending.error else ""
    no = Decision(approved=False, reason="refund policy: max 500", decided_by="ana")
    await store.decide(approval_id, no)
    denied = await executor.run_call(call, ctx)
    print(denied.status, denied.error.message if denied.error else "")  # the tool never ran
    # --8<-- [end:deny]


async def main() -> None:
    small = await executor.run("refund", {"order_id": "o-1", "amount": 20}, ctx)
    print(small)  # under the rule's limit: no review
    await pause_and_resume()
    await deny()
    print(paid)


if __name__ == "__main__":
    asyncio.run(main())
