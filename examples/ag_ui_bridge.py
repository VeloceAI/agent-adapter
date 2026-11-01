"""AG-UI: a run streams tool calls, pauses on an approval, and resumes through our port."""

import asyncio
from typing import Any

from agent_adapter import (
    ApprovalRequest,
    CallbackApproval,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
)
from agent_adapter.integrations.ag_ui import AGUIBridge, AGUIRun

registry = Registry()


@registry.tool
def order_total(order_id: str) -> float:
    """The total of an order."""
    return 120.0


@registry.tool(confirm=True, effects="destructive")
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


# --8<-- [start:port]
def support_lead_approves(request: ApprovalRequest) -> bool:
    # The remote "approved: true" only asks; this port decides who may approve what.
    resume = request.metadata.get("resume_payload") or {}
    return request.metadata.get("principal") == "lead-7" and resume.get("approved") is True


# The executor parks confirm calls; the bridge's port decides each resume request.
executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
bridge = AGUIBridge(executor, port=CallbackApproval(support_lead_approves))
# --8<-- [end:port]


# --8<-- [start:agent]
async def agent(run: AGUIRun) -> Any:
    if run.input.resume:  # a resumed run: the bridge already ran or denied the paused calls
        result = run.resumed[0]
        run.text(f"refund {'done' if result.ok else 'refused'}")
        return result.value
    total = await run.call("order_total", {"order_id": "o-7"})
    await run.call("refund", {"order_id": "o-7", "amount": total.value})  # pauses
    return None


# --8<-- [end:agent]


async def main() -> None:
    # --8<-- [start:round-trip]
    first = {"threadId": "t-1", "runId": "r-1", "messages": []}
    events = [e async for e in bridge.stream(first, agent, principal="lead-7")]
    print([e.type.value for e in events])
    interrupt = events[-1].outcome.interrupts[0]
    refund_call = [e for e in events if e.type.value == "TOOL_CALL_START"][-1]
    print(interrupt.reason, interrupt.tool_call_id == refund_call.tool_call_id)

    answer = {"interruptId": interrupt.id, "status": "resolved", "payload": {"approved": True}}
    second = {"threadId": "t-1", "runId": "r-2", "messages": [], "resume": [answer]}
    events = [e async for e in bridge.stream(second, agent, principal="lead-7")]
    print([e.type.value for e in events])
    print(events[-1].result)
    # --8<-- [end:round-trip]


if __name__ == "__main__":
    asyncio.run(main())
