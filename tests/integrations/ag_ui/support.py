"""Shared AG-UI test helpers: a registry with a confirm tool, and run collectors."""

from typing import Any

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations.ag_ui import AGUIBridge, AGUIRun

THREAD = "t-1"


class Shop:
    """`total` (no approval) and `refund` (confirm), recording every refund that ran."""

    def __init__(self) -> None:
        self.refunds: list[tuple[str, float]] = []
        self.registry = Registry()

        @self.registry.tool
        def total(order_id: str) -> float:
            """An order total."""
            return 120.0

        @self.registry.tool(confirm=True, effects="destructive")
        def refund(order_id: str, amount: float) -> str:
            """Refund an order."""
            self.refunds.append((order_id, amount))
            return f"refunded {amount}"

        self.store = MemoryApprovalStore()

    def bridge(self, port: ApprovalPort | None = None, **options: Any) -> AGUIBridge:
        executor = Executor(self.registry, approval=DeferredApproval(self.store))
        return AGUIBridge(executor, port=port, **options)


async def refund_agent(run: AGUIRun) -> Any:
    """Calls `refund` on a fresh run; reports the resumed results on a resumed one."""
    if run.input.resume:
        return [r.status for r in run.resumed]
    await run.call("refund", {"order_id": "o-7", "amount": 50.0}, call_id="c-refund")
    return None


def run_input(run_id: str, *, thread: str = THREAD, resume: list[Any] | None = None) -> dict:
    body: dict[str, Any] = {"threadId": thread, "runId": run_id, "messages": []}
    if resume is not None:
        body["resume"] = resume
    return body


def approve(interrupt_id: str, payload: Any = None, status: str = "resolved") -> dict:
    entry: dict[str, Any] = {"interruptId": interrupt_id, "status": status}
    entry["payload"] = {"approved": True} if payload is None else payload
    return entry


async def collect(bridge: AGUIBridge, body: Any, agent: Any, principal: str = "") -> list[Any]:
    return [event async for event in bridge.stream(body, agent, principal=principal)]


def types(events: list[Any]) -> list[str]:
    return [event.type.value for event in events]


def interrupts(events: list[Any]) -> list[Any]:
    outcome = events[-1].outcome
    return list(outcome.interrupts) if outcome is not None and outcome.type == "interrupt" else []


async def paused(bridge: AGUIBridge, principal: str = "") -> Any:
    """Run `refund_agent` once; the single interrupt it ends with."""
    (found,) = interrupts(await collect(bridge, run_input("r-1"), refund_agent, principal))
    return found
