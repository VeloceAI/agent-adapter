"""Resume-contract fixes: no invalid state is stored; recovery ignores process memory."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import MemoryCheckpointStore
from agent_adapter.flow import Call, Flow
from agent_adapter.flow.visit import SPENT
from tests.flow.durable_helpers import RAN, Crash, CrashingStore, Order, registry, reset, runner


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset()


def first(state: Order) -> dict[str, Any]:
    return {"log": ["first"]}


def spoil(state: Order) -> dict[str, Any]:
    state.items.append(7)  # type: ignore[arg-type]  # in place: no update to validate
    return {}


async def test_a_state_changed_in_place_into_an_invalid_one_fails_and_is_not_stored() -> None:
    flow = Flow("spoiled", state=Order).node("a", first).node("b", spoil).edge("a", "b").end("b")
    store = MemoryCheckpointStore()
    r = runner(store)
    with pytest.raises(FlowError, match="invalid"):
        await r.run(flow, {}, r.executor.context(run_id="run-1"))
    status, checkpoint = await r.inspect(flow, "run-1")
    assert (status, checkpoint.state.items, checkpoint.state.log) == ("failed", [], ["first"])


def refunds() -> Flow[Order]:
    return (
        Flow("refunds", state=Order)
        .node("pay", Call("charge", args=lambda s: {"order": s.order}, into="note"))
        .node("back", Call("refund", args=lambda s: {"order": s.order}, into="note"))
        .edge("pay", "back")
        .end("back")
    )


async def test_recovering_a_stored_spend_does_not_depend_on_this_process_memory() -> None:
    store, approvals = CrashingStore(), MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(approvals))
    first_run = await runner(store, executor=executor).run(
        refunds(), {}, executor.context(run_id="run-1")
    )
    checkpoint = first_run.checkpoint
    assert checkpoint is not None and checkpoint.pending is not None
    store.after = store.saves[-1][0] + 1  # dies once the decision is stored, before it is spent
    approve = ApprovalDecision(approved=True, decided_by="ana")
    with pytest.raises(Crash):
        await runner(store, executor=executor).resume(refunds(), checkpoint, approve)
    pending = checkpoint.pending
    assert not (await approvals.get(pending.approval_id)).consumed
    SPENT.remember((pending.run_id, pending.approval_id))  # this process spent it elsewhere
    done = await runner(store, executor=executor).recover(refunds(), "run-1")
    assert (done.status, RAN["refund"]) == ("done", 1)
    assert (await approvals.get(pending.approval_id)).consumed
