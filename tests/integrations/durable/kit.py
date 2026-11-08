"""Shared pieces of the durable-engine tests: a three-step order flow, a gate, a crash."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import MemoryCheckpointStore, idempotency_key
from agent_adapter.flow import Call, Flow, FlowRunner
from agent_adapter.integrations._durable_flows import DurableFlows
from agent_adapter.tracing.sinks import EventLog

KEY = b"k" * 32
RAN: Counter[str] = Counter()
KEYS: list[str | None] = []  # idempotency keys seen by `pay`
DRIFT = {"n": 0}  # changes the arguments of `drift` between attempts
CRASH_AT: dict[str, int] = {}  # tool -> crash in its body on this call number


class Crash(BaseException):
    """The worker died here."""


class Order(BaseModel):
    order: str = "o-1"
    note: str = ""
    sent: str = ""


registry = Registry()


def _count(tool: str) -> None:
    RAN[tool] += 1
    if CRASH_AT.get(tool) == RAN[tool]:
        raise Crash(tool)


@registry.tool
async def charge(order: str) -> str:
    """Charge an order (not idempotent)."""
    _count("charge")
    return f"charged {order}"


@registry.tool
async def ship(order: str) -> str:
    """Ship an order (not idempotent)."""
    _count("ship")
    return f"shipped {order}"


@registry.tool(confirm=True)
async def refund(order: str) -> str:
    """Refund an order (needs approval)."""
    _count("refund")
    return f"refunded {order}"


@registry.tool
async def pay(order: str) -> str:
    """Pay an order; records the idempotency key it would send to the payment API."""
    KEYS.append(idempotency_key())
    _count("pay")
    return f"paid {order}"


@registry.tool(idempotent=True)
async def lookup(order: str) -> str:
    """Read an order (safe to repeat)."""
    KEYS.append(idempotency_key())
    _count("lookup")
    return f"found {order}"


def orders() -> Flow[Order]:
    """charge -> ship -> done: two supersteps."""
    return (
        Flow("orders", state=Order)
        .node("charge", Call("charge", args=lambda s: {"order": s.order}, into="note"))
        .node("ship", Call("ship", args=lambda s: {"order": s.order}, into="sent"))
        .edge("charge", "ship")
        .end("ship")
    )


def gated() -> Flow[Order]:
    """A gate before shipping: the run pauses for a node approval."""
    return (
        Flow("gated", state=Order)
        .node("check", lambda s: {"note": "checked"}, pause=True)
        .node("ship", Call("ship", args=lambda s: {"order": s.order}, into="sent"))
        .edge("check", "ship")
        .end("ship")
    )


def refunds() -> Flow[Order]:
    """One node whose tool needs approval: the run pauses for a tool approval."""
    return (
        Flow("refunds", state=Order)
        .node("refund", Call("refund", args=lambda s: {"order": s.order}, into="note"))
        .end("refund")
    )


def broken() -> Flow[Order]:
    """A node that fails: the run is marked failed."""

    def fail(state: Order) -> dict[str, str]:
        raise ValueError("boom")

    return Flow("broken", state=Order).node("fail", fail).end("fail")


def twice() -> Flow[Order]:
    """Two gates in a row: two pauses, two approval ids."""
    return (
        Flow("twice", state=Order)
        .node("first", lambda s: {"note": "one"}, pause=True)
        .node("second", lambda s: {"sent": "two"}, pause=True)
        .edge("first", "second")
        .end("second")
    )


def payments() -> Flow[Order]:
    """pay -> lookup -> ship: one non-idempotent call, one idempotent call."""
    return (
        Flow("payments", state=Order)
        .node("pay", Call("pay", args=lambda s: {"order": s.order}, into="note"))
        .node("lookup", Call("lookup", args=lambda s: {"order": s.order}, into="sent"))
        .edge("pay", "lookup")
        .end("lookup")
    )


def drifting() -> Flow[Order]:
    """A call whose arguments differ on each attempt (a non-deterministic node)."""
    return (
        Flow("drifting", state=Order)
        .node(
            "ship", Call("ship", args=lambda s: {"order": f"{s.order}-{DRIFT['n']}"}, into="sent")
        )
        .node("lookup", Call("lookup", args=lambda s: {"order": s.order}, into="note"))
        .edge("ship", "lookup")
        .end("lookup")
    )


def durable(store: MemoryCheckpointStore | None = None) -> tuple[DurableFlows, EventLog]:
    """DurableFlows over every test flow, with a deferred approval store and an event log."""
    RAN.clear()
    CRASH_AT.clear()
    KEYS.clear()
    DRIFT["n"] = 0
    log = EventLog()
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()), sink=log)
    runner = FlowRunner(executor, checkpoints=store or MemoryCheckpointStore(), checkpoint_key=KEY)
    return DurableFlows(
        runner, [orders(), gated(), refunds(), broken(), twice(), payments(), drifting()]
    ), log


async def decide(flows: DurableFlows, approval_id: str, approved: bool, **fields: str) -> None:
    """A reviewer's decision, written where it counts: the approval store."""
    store = flows.runner.executor.approval.store  # type: ignore[union-attr]
    await store.decide(approval_id, ApprovalDecision(approved=approved, **fields))
