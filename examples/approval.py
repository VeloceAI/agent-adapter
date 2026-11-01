"""Approval: a conditional confirm rule, in-process ports, and the deferred round trip."""

import asyncio

from agent_adapter import (
    ApprovalDecision,
    ApprovalPending,
    ApprovalRequest,
    AutoDeny,
    CallbackApproval,
    DeferredApproval,
    MemoryApprovalStore,
    Registry,
    ToolCall,
    ToolContext,
)
from agent_adapter.approval import evaluate_confirm

registry = Registry()


# --8<-- [start:rule]
def large_refund(call: ToolCall, ctx: ToolContext) -> bool:
    return bool(call.arguments["amount"] > 100)  # small refunds run without review


@registry.tool(confirm=large_refund)
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


# --8<-- [end:rule]


def refund_request(amount: float, approval_id: str = "") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id="run_1",
        call_id="call_1",
        tool="refund",
        arguments={"order_id": "o-7", "amount": amount},
    )


async def main() -> None:
    ctx = ToolContext(run_id="run_1")
    spec = registry.get("refund")
    for amount in (20, 500):
        call = ToolCall(name="refund", arguments={"order_id": "o-7", "amount": amount})
        print(amount, await evaluate_confirm(spec.confirm, call, ctx))

    # --8<-- [start:ports]
    under_limit = CallbackApproval(lambda req: req.arguments["amount"] <= 1000)
    print((await under_limit.request(refund_request(500))).approved)
    print((await AutoDeny(reason="read-only mode").request(refund_request(500))).reason)
    # --8<-- [end:ports]

    # --8<-- [start:deferred]
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    try:
        await port.request(refund_request(500))  # 1. no record yet: create one, pause
    except ApprovalPending as pending:
        approval_id = pending.approval_id
    print((await store.get(approval_id)).status)  # 2. a reviewer lists pending records

    decision = ApprovalDecision(approved=True, edited_args={"amount": 450}, decided_by="ana")
    await store.decide(approval_id, decision)  # 3. ... and decides (HTTP server, queue)

    resumed = await port.request(refund_request(500))  # 4. the resumed call gets it, once
    print(resumed.approved, resumed.edited_args, (await store.get(approval_id)).consumed)
    # --8<-- [end:deferred]


if __name__ == "__main__":
    asyncio.run(main())
