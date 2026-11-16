"""The `FlowRunner` subject: charge, ship and quote in parallel, a gated refund, two calls."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval
from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow, FlowRunner, Join, Merge
from tests.resume_contract.harness import (
    ACTIVE,
    EXECUTOR,
    KEY,
    FaultStore,
    OurStores,
    Outcome,
    SnapApprovals,
    World,
    effect,
    enter,
)

RUN = "trip-1"
APPROVE = ApprovalDecision(approved=True, decided_by="ana")
registry = Registry()


@registry.tool
async def charge(order: str) -> str:
    """Charge the card (an effect)."""
    effect("charge")
    return f"charged {order}"


@registry.tool
async def ship(order: str) -> str:
    """Ship the parcel (an effect)."""
    effect("ship")
    return f"shipped {order}"


@registry.tool(idempotent=True)
async def quote(order: str) -> str:
    """Read a delivery quote (safe to repeat)."""
    ACTIVE.world.boundary("inside quote")
    return f"quote {order}"


@registry.tool(confirm=True)
async def refund(order: str, amount: int) -> str:
    """Refund part of the order (an effect that needs approval)."""
    effect("refund")
    return f"refunded {amount} on {order}"


@registry.tool
async def email(order: str) -> str:
    """Mail the receipt (an effect)."""
    effect("email")
    return f"mailed {order}"


@registry.tool
async def audit(order: str) -> str:
    """Write the audit row (an effect)."""
    effect("audit")
    return f"audited {order}"


class Trip(BaseModel):
    order: str = "o-1"
    log: Annotated[list[str], Merge.append] = []


async def _run(tool: str, state: Trip, ctx: ToolContext, **extra: Any) -> str:
    return str(await EXECUTOR.get().run(tool, {"order": state.order, **extra}, ctx))


async def pay(state: Trip, ctx: ToolContext) -> dict[str, Any]:
    enter("pay", state.model_dump())
    return {"log": [await _run("charge", state, ctx)]}


async def send(state: Trip, ctx: ToolContext) -> dict[str, Any]:
    enter("ship", state.model_dump())
    return {"log": [await _run("ship", state, ctx)]}


async def look(state: Trip, ctx: ToolContext) -> dict[str, Any]:
    enter("quote", state.model_dump())
    return {"log": [await _run("quote", state, ctx)]}


async def gathered(state: Trip) -> dict[str, Any]:
    enter("gather", state.model_dump())
    return {}


async def give_back(state: Trip, ctx: ToolContext) -> dict[str, Any]:
    enter("refund", state.model_dump())
    return {"log": [await _run("refund", state, ctx, amount=5)]}


async def close(state: Trip, ctx: ToolContext) -> dict[str, Any]:
    enter("close", state.model_dump())
    mailed = await _run("email", state, ctx)
    return {"log": [mailed, await _run("audit", state, ctx)]}


FLOW = (
    Flow("resume_contract", state=Trip)
    .node("pay", pay)
    .node("ship", send)
    .node("quote", look)
    .node("gather", Join(["ship", "quote"], fn=gathered))
    .node("refund", give_back)
    .node("close", close)
    .edge("pay", ["ship", "quote"])
    .edge("ship", "gather")
    .edge("quote", "gather")
    .edge("gather", "refund")
    .edge("refund", "close")
    .end("close")
)


class FlowSubject(OurStores):
    """Our `FlowRunner` with a durable store and a deferred approval port."""

    name = "FlowRunner"
    run_id = RUN
    effects = ("charge", "ship", "refund", "email", "audit")

    def __init__(self, approvals: SnapApprovals | None = None) -> None:
        self.approvals = approvals or SnapApprovals()
        self.executor = Executor(registry, approval=DeferredApproval(self.approvals))

    def fresh(self) -> FlowSubject:
        return FlowSubject(self.approvals.snapshot())

    def runner(self, store: Any) -> FlowRunner:
        EXECUTOR.set(self.executor)
        return FlowRunner(self.executor, checkpoints=store, checkpoint_key=KEY)

    def _out(self, result: Any) -> Outcome:
        if result.status == "paused":
            return Outcome("paused", token=result.checkpoint)
        return Outcome(result.status, final=result.state.model_dump())

    async def start(self, store: FaultStore) -> Outcome:
        ctx = self.executor.context(run_id=RUN)
        return self._out(await self.runner(store).run(FLOW, {}, ctx))

    async def resume(self, store: FaultStore, token: Any) -> Outcome:
        return self._out(await self.runner(store).resume(FLOW, token, APPROVE))

    async def recover(self, store: FaultStore, *, rerun: bool) -> Outcome:
        return self._out(await self.runner(store).recover(FLOW, RUN, rerun_in_doubt=rerun))

    async def ended(self, store: FaultStore) -> Outcome:
        status, checkpoint = await self.runner(store).inspect(FLOW, RUN)
        return Outcome(status, final=checkpoint.state.model_dump())

    async def position(self, store: FaultStore) -> Any:
        status, checkpoint = await self.runner(store).inspect(FLOW, RUN)
        return status, checkpoint.frontier, checkpoint.state.model_dump()

    def continued(self, position: Any, world: World) -> str | None:
        """Prefix continuation: the first node entered is on the durable frontier, same state."""
        status, frontier, state = position
        if not world.entered:
            return None
        node, seen = world.entered[0]
        if status not in ("running", "paused") or node not in frontier or seen != state:
            return f"entered {node} with {seen}, durable frontier {frontier} with {state}"
        return None
