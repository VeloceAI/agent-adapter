"""Durable runs under attack and bad timing: fencing, signed records, resume crashes, identity."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    CheckpointConflict,
    CheckpointStoreCorrupt,
    FlowError,
    RecoveryNeedsDecision,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import Checkpoint, MemoryCheckpointStore, idempotency_key
from agent_adapter.flow import Call, Flow
from tests.fakes.sinks import RecordingSink
from tests.flow.durable_helpers import (
    CRASH_AT,
    RAN,
    Crash,
    CrashingStore,
    Order,
    registry,
    reset,
    runner,
    three_steps,
)

APPROVE = ApprovalDecision(approved=True, decided_by="ana")


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset()


class HookedStore(MemoryCheckpointStore):
    """Runs `on_save(step)` right after each save, to interleave two workers."""

    on_save: Any = None

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        await super().save(run_id, step, checkpoint)
        if self.on_save is not None:
            await self.on_save(step)


async def test_a_slow_worker_is_fenced_before_its_next_call_after_a_claim() -> None:
    store = HookedStore()
    executor = Executor(registry)
    gate, entered = asyncio.Event(), asyncio.Event()
    visits = {"n": 0}

    async def slow(state: Order, ctx: Any) -> dict[str, Any]:
        visits["n"] += 1
        me = visits["n"]
        first = await executor.run("charge", {"order": state.order}, ctx)
        if me == 1:
            entered.set()
            await gate.wait()
        else:
            for _ in range(20):
                await asyncio.sleep(0)  # the new worker gives the zombie time to go on
        second = await executor.run("ship", {"order": state.order}, ctx)
        return {"log": [first, second]}

    flow = Flow("slow", state=Order).node("work", slow).node("end", lambda s: None)
    flow = flow.edge("work", "end").end("end")
    zombie = asyncio.create_task(
        runner(store, executor=executor).run(flow, {}, executor.context(run_id="run-1"))
    )
    await entered.wait()

    async def after(step: int) -> None:
        if step == 1:  # the claim: let the zombie run while the new worker starts
            gate.set()
            for _ in range(5):
                await asyncio.sleep(0)

    store.on_save = after
    result = await runner(store, executor=executor).recover(flow, "run-1")
    assert result.status == "done"
    with pytest.raises(FlowError):
        await zombie
    assert dict(RAN) == {"charge": 1, "ship": 1}


CRASH_AFTER = {"on": False}


async def crashed_with_calls() -> MemoryCheckpointStore:
    store = MemoryCheckpointStore()
    executor = Executor(registry)
    CRASH_AFTER["on"] = True
    with pytest.raises(Crash):
        await runner(store, executor=executor).run(
            two_flow(executor), {}, executor.context(run_id="run-1")
        )
    CRASH_AFTER["on"] = False
    return store


def two_flow(executor: Executor, order: str | None = None) -> Flow[Order]:
    async def both(state: Order, ctx: Any) -> dict[str, Any]:
        a = await executor.run("charge", {"order": order or state.order}, ctx)
        b = await executor.run("lookup", {"order": state.order}, ctx)
        if CRASH_AFTER["on"]:
            raise Crash("after the calls")
        return {"log": [a, b]}

    return Flow("two", state=Order).node("both", both).end("both")


def segment_calls(store: MemoryCheckpointStore) -> dict[str, Any]:
    [(key, calls)] = list(store._calls.items())
    del key
    return calls


async def test_a_forged_or_changed_call_record_is_refused() -> None:
    store = await crashed_with_calls()
    calls = segment_calls(store)
    key, record = next((k, c) for k, c in calls.items() if c.tool == "charge")
    calls[key] = record.model_copy(update={"result": {**record.result, "value": "forged"}})
    executor = Executor(registry)
    with pytest.raises(CheckpointStoreCorrupt, match="signed"):
        await runner(store, executor=executor).recover(two_flow(executor), "run-1")
    calls[key] = record.model_copy(update={"state": "started", "result": None})
    with pytest.raises(CheckpointStoreCorrupt, match="signed"):
        await runner(store, executor=executor).recover(two_flow(executor), "run-1")
    assert RAN["charge"] == 1


async def test_a_deleted_call_record_is_noticed() -> None:
    store = await crashed_with_calls()
    calls = segment_calls(store)
    first = next(k for k, c in calls.items() if c.seq == 0)
    del calls[first]
    executor = Executor(registry)
    with pytest.raises(CheckpointStoreCorrupt, match="missing"):
        await runner(store, executor=executor).recover(two_flow(executor), "run-1")


async def test_a_status_flip_or_a_copied_checkpoint_is_refused() -> None:
    store = await crashed_with_calls()
    found = store._runs["run-1"]
    flipped = found.checkpoint.model_copy(update={"status": "running"})
    failed = found.model_copy(
        update={"checkpoint": found.checkpoint.model_copy(update={"status": "failed"})}
    )
    store._runs["run-1"] = failed
    executor = Executor(registry)
    with pytest.raises(CheckpointStoreCorrupt, match="disagrees"):  # failed is signed too
        await runner(store, executor=executor).recover(two_flow(executor), "run-1")
    store._runs["run-1"] = found.model_copy(update={"checkpoint": flipped})
    store._runs["run-2"] = found.model_copy(update={"run_id": "run-2"})
    with pytest.raises(CheckpointStoreCorrupt, match="disagrees"):
        await runner(store, executor=executor).recover(two_flow(executor), "run-2")


def refunds() -> Flow[Order]:
    return (
        Flow("refunds", state=Order)
        .node("pay", Call("charge", args=lambda s: {"order": s.order}, into="note"))
        .node("back", Call("refund", args=lambda s: {"order": s.order}, into="note"))
        .edge("pay", "back")
        .end("back")
    )


async def paused_run() -> tuple[CrashingStore, Executor, Any]:
    store = CrashingStore()
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    first = await runner(store, executor=executor).run(
        refunds(), {"order": "o-1"}, executor.context(run_id="run-1")
    )
    assert first.status == "paused" and first.checkpoint is not None
    return store, executor, first.checkpoint


async def test_a_crash_before_the_resume_save_leaves_the_run_resumable() -> None:
    store, executor, checkpoint = await paused_run()
    store.before = store.saves[-1][0] + 1
    with pytest.raises(Crash):
        await runner(store, executor=executor).resume(refunds(), checkpoint, APPROVE)
    done = await runner(store, executor=executor).resume(refunds(), checkpoint, APPROVE)
    assert (done.status, RAN["refund"]) == ("done", 1)


async def test_a_crash_between_the_resume_save_and_the_spend_recovers() -> None:
    store, executor, checkpoint = await paused_run()
    store.after = store.saves[-1][0] + 1  # the decision is stored, not yet spent
    with pytest.raises(Crash):
        await runner(store, executor=executor).resume(refunds(), checkpoint, APPROVE)
    done = await runner(store, executor=executor).recover(refunds(), "run-1")
    assert (done.status, done.state.note, RAN["refund"]) == ("done", "refunded o-1", 1)
    with pytest.raises(CheckpointConflict):
        await runner(store, executor=executor).resume(refunds(), checkpoint, APPROVE)


async def test_a_crash_after_the_spend_recovers_with_the_spent_decision() -> None:
    store, executor, checkpoint = await paused_run()
    CRASH_AT["refund"] = 1
    with pytest.raises(Crash):
        await runner(store, executor=executor).resume(refunds(), checkpoint, APPROVE)
    with pytest.raises(RecoveryNeedsDecision):
        await runner(store, executor=executor).recover(refunds(), "run-1")
    done = await runner(store, executor=executor).recover(refunds(), "run-1", rerun_in_doubt=True)
    assert (done.status, RAN["refund"]) == ("done", 2)


user_registry = Registry()
SEEN: list[str] = []


@user_registry.tool
async def whoami(ctx: ToolContext) -> str:
    """Report the run's end user."""
    SEEN.append(ctx.user_id)
    if len(SEEN) == 1:
        raise Crash("first try")
    return ctx.user_id


def whoami_flow() -> Flow[Order]:
    return Flow("who", state=Order).node("w", Call("whoami", args={}, into="note")).end("w")


async def test_recovery_keeps_the_end_user_and_refuses_another_one() -> None:
    SEEN.clear()
    store = MemoryCheckpointStore()
    executor = Executor(user_registry)
    r = runner(store, executor=executor)
    with pytest.raises(Crash):
        await r.run(whoami_flow(), {}, executor.context(run_id="run-1", user_id="ana"))
    other = executor.context(run_id="run-1", user_id="bob")
    with pytest.raises(ToolUsageError, match="another user_id"):
        await runner(store, executor=executor).recover(whoami_flow(), "run-1", other)
    out = await runner(store, executor=executor).recover_all(whoami_flow(), rerun_in_doubt=True)
    assert out["run-1"].state.note == "ana"  # type: ignore[union-attr]
    assert SEEN == ["ana", "ana"]


async def test_a_run_with_a_principal_needs_its_context_back() -> None:
    SEEN.clear()
    store = MemoryCheckpointStore()
    executor = Executor(user_registry)
    ctx = executor.context(run_id="run-1", extras={"principal": {"id": "svc-7"}})
    with pytest.raises(Crash):
        await runner(store, executor=executor).run(whoami_flow(), {}, ctx)
    with pytest.raises(ToolUsageError, match="principal"):
        await runner(store, executor=executor).recover(whoami_flow(), "run-1")
    wrong = executor.context(run_id="run-1", extras={"principal": "svc-8"})
    with pytest.raises(ToolUsageError, match="another principal"):
        await runner(store, executor=executor).recover(whoami_flow(), "run-1", wrong)
    again = executor.context(run_id="run-1", extras={"principal": {"id": "svc-7"}})
    done = await runner(store, executor=executor).recover(
        whoami_flow(), "run-1", again, rerun_in_doubt=True
    )
    assert done.status == "done"


async def test_a_logged_result_that_cannot_replay_fails_closed() -> None:
    store = await crashed_with_calls()
    calls = segment_calls(store)
    key, record = next((k, c) for k, c in calls.items() if c.tool == "charge")
    result = {k: v for k, v in record.result.items() if k != "value"}  # no longer loads
    broken = record.model_copy(update={"result": result})
    from agent_adapter.exec._durable_seal import call_mac
    from tests.flow.durable_helpers import KEY

    seg = next(iter(store._calls))[1]
    calls[key] = broken.model_copy(update={"mac": call_mac(KEY, "run-1", seg, broken)})
    executor = Executor(registry)
    with pytest.raises(FlowError) as caught:
        await runner(store, executor=executor).recover(two_flow(executor), "run-1")
    chain, error = [], caught.value
    while error is not None:
        chain.append(type(error))
        error = error.__cause__  # type: ignore[assignment]
    assert CheckpointStoreCorrupt in chain
    assert RAN["charge"] == 1  # never run again
    assert store._runs["run-1"].checkpoint.status == "running"  # still recoverable


async def test_calls_whose_arguments_changed_are_reported() -> None:
    store = await crashed_with_calls()
    executor = Executor(registry)
    sink = RecordingSink()
    flow = two_flow(executor, order="changed")
    await runner(store, executor=executor, sink=sink).recover(flow, "run-1")
    saved = [e.data for e in sink.events if e.type == "checkpoint.saved"]
    assert saved[-1].get("unreplayed") == 1
    assert RAN["charge"] == 2


async def test_a_crash_during_the_claim_and_again_during_recovery_still_recovers() -> None:
    store = CrashingStore()
    CRASH_AT["ship"] = 1
    with pytest.raises(Crash):
        await runner(store).run(three_steps(), {}, runner(store).executor.context(run_id="run-1"))
    store.before = store.saves[-1][0] + 1  # dies saving the claim
    with pytest.raises(Crash):
        await runner(store).recover(three_steps(), "run-1", rerun_in_doubt=True)
    CRASH_AT["ship"] = 2  # dies again inside ship, after the claim
    with pytest.raises(Crash):
        await runner(store).recover(three_steps(), "run-1", rerun_in_doubt=True)
    done = await runner(store).recover(three_steps(), "run-1", rerun_in_doubt=True)
    assert done.status == "done"
    assert dict(RAN) == {"charge": 1, "ship": 3, "lookup": 1}


async def test_a_timed_out_call_is_in_doubt_on_recovery() -> None:
    slow_registry = Registry()

    @slow_registry.tool(timeout=0.01)
    async def mail(order: str) -> str:
        """Send a mail (slow)."""
        await asyncio.sleep(1)
        return "sent"

    store = MemoryCheckpointStore()
    executor = Executor(slow_registry)

    async def node(state: Order, ctx: Any) -> dict[str, Any]:
        try:
            await executor.run("mail", {"order": state.order}, ctx)
        except Exception:
            raise Crash("after the timeout") from None
        return {}

    flow = Flow("mail", state=Order).node("m", node).end("m")
    with pytest.raises(Crash):
        await runner(store, executor=executor).run(flow, {}, executor.context(run_id="run-1"))
    calls = list(segment_calls(store).values())
    assert [c.state for c in calls] == ["done"]  # a timeout is a finished (failed) call


async def test_the_idempotency_key_is_stable_across_a_recovery() -> None:
    keys: list[str | None] = []
    key_registry = Registry()

    @key_registry.tool
    async def pay(order: str) -> str:
        """Pay with an idempotency key."""
        keys.append(idempotency_key())
        if len(keys) == 1:
            raise Crash("mid payment")
        return "paid"

    store = MemoryCheckpointStore()
    executor = Executor(key_registry)
    flow = Flow("pay", state=Order).node("p", Call("pay", args={"order": "o"})).end("p")
    with pytest.raises(Crash):
        await runner(store, executor=executor).run(flow, {}, executor.context(run_id="run-1"))
    await runner(store, executor=executor).recover(flow, "run-1", rerun_in_doubt=True)
    assert keys[0] is not None and keys[0] == keys[1]
    assert idempotency_key() is None
