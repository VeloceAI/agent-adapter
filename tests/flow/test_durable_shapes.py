"""Durable runs across flow shapes: Map fan-out, sub-flows, node cache, pauses, zombies."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.cache import MemoryCache
from agent_adapter.core.errors import CheckpointConflict, FlowError, RecoveryNeedsDecision
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import MemoryCheckpointStore
from agent_adapter.flow import Call, Flow, Map, NodeCache, SubFlow
from tests.fakes.sinks import RecordingSink
from tests.flow.durable_helpers import CRASH_AT, RAN, Crash, Order, node_fn, registry, reset, runner

APPROVE = ApprovalDecision(approved=True, decided_by="ana")


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset()


async def crashed(r: Any, flow: Flow[Any], state: dict[str, Any] | None = None) -> None:
    with pytest.raises(Crash):
        await r.run(flow, state or {"order": "o-1"}, r.executor.context(run_id="run-1"))


def fan_out(**options: Any) -> Flow[Order]:
    action = Map(over="items", tool="charge", args_fn=lambda item: {"order": item}, into="log")
    return Flow("fan", state=Order).node("each", action, **options).end("each")


async def test_a_map_fan_out_partly_done_replays_finished_items() -> None:
    store = MemoryCheckpointStore()
    CRASH_AT["charge"] = 3
    state = {"items": ["a", "b", "c", "d"]}
    await crashed(runner(store), fan_out(), state)
    with pytest.raises(RecoveryNeedsDecision) as caught:
        await runner(store).recover(fan_out(), "run-1")
    assert [tool for tool, _ in caught.value.calls] == ["charge"]
    result = await runner(store).recover(fan_out(), "run-1", rerun_in_doubt=True)
    assert result.state.log == ["charged a", "charged b", "charged c", "charged d"]
    assert RAN["charge"] == 5  # a, b replay; c ran twice (in doubt); d once


async def test_identical_map_items_keep_their_own_results() -> None:
    store = MemoryCheckpointStore()
    CRASH_AT["lookup"] = 3
    action = Map(over="items", tool="lookup", args_fn=lambda item: {"order": item}, into="log")
    flow = Flow("same", state=Order).node("each", action).end("each")
    await crashed(runner(store), flow, {"items": ["x", "x", "x"]})
    result = await runner(store).recover(flow, "run-1")  # lookup is idempotent
    assert result.state.log == ["value x"] * 3
    assert RAN["lookup"] == 4


async def test_a_map_of_functions_replays_each_items_own_calls() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry)

    async def item(value: str, state: Order, ctx: Any) -> str:
        return str(await executor.run("charge", {"order": value}, ctx))

    flow = Flow("fn", state=Order).node("each", Map(over="items", fn=item, into="log")).end("each")
    CRASH_AT["charge"] = 2
    r = runner(store, executor=executor)
    await crashed(r, flow, {"items": ["a", "b", "c"]})
    result = await runner(store, executor=executor).recover(flow, "run-1", rerun_in_doubt=True)
    assert result.state.log == ["charged a", "charged b", "charged c"]
    assert RAN["charge"] == 4


def packing(executor: Executor) -> Flow[Order]:
    return (
        Flow("packing", state=Order)
        .node("pay", node_fn(executor, "charge"))
        .node("send", node_fn(executor, "ship"))
        .edge("pay", "send")
        .end("send")
    )


async def test_a_crash_inside_a_sub_flow_replays_the_childs_finished_calls() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry)
    flow = Flow("order", state=Order).node("pack", SubFlow(packing(executor))).end("pack")
    CRASH_AT["ship"] = 1
    await crashed(runner(store, executor=executor), flow)
    result = await runner(store, executor=executor).recover(flow, "run-1", rerun_in_doubt=True)
    assert result.state.log == ["charged o-1", "shipped o-1"]
    assert dict(RAN) == {"charge": 1, "ship": 2}


async def test_a_cached_node_is_not_run_again_when_its_superstep_recovers() -> None:
    store = MemoryCheckpointStore()
    sink = RecordingSink()
    executor = Executor(registry, cache=MemoryCache())
    cached = (
        Flow("cached", state=Order)
        .node("start", lambda s: None)
        .node(
            "read",
            Call("lookup", args=lambda s: {"order": s.order}, into="note"),
            cache=NodeCache(reads=["order"]),
        )
        .node("both", node_fn(executor, "charge", "ship"))
        .edge("start", ["read", "both"])
        .end("read", "both")
    )
    CRASH_AT["ship"] = 1
    await crashed(runner(store, executor=executor), cached)
    result = await runner(store, executor=executor, sink=sink).recover(
        cached, "run-1", rerun_in_doubt=True
    )
    assert result.state.note == "value o-1"
    assert dict(RAN) == {"lookup": 1, "charge": 1, "ship": 2}
    assert "node.cached" in sink.types()


def refunds() -> Flow[Order]:
    return (
        Flow("refunds", state=Order)
        .node("pay", Call("charge", args=lambda s: {"order": s.order}, into="note"))
        .node("back", Call("refund", args=lambda s: {"order": s.order}, into="note"))
        .edge("pay", "back")
        .end("back")
    )


def approving() -> Executor:
    return Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))


async def test_a_run_paused_at_the_crash_comes_back_paused_and_resumes() -> None:
    store = MemoryCheckpointStore()
    executor = approving()
    r = runner(store, executor=executor)
    first = await r.run(refunds(), {"order": "o-1"}, executor.context(run_id="run-1"))
    assert first.status == "paused"
    assert await store.list_unfinished() == []  # paused runs wait for a person, not a worker
    again = await runner(store, executor=executor).recover(refunds(), "run-1")
    assert again.status == "paused" and again.checkpoint is not None
    done = await runner(store, executor=executor).resume(refunds(), again.checkpoint, APPROVE)
    assert (done.status, done.state.note) == ("done", "refunded o-1")
    assert (await store.latest("run-1")).checkpoint.status == "done"  # type: ignore[union-attr]
    with pytest.raises(CheckpointConflict):  # the paused checkpoint is not the latest any more
        await runner(store, executor=executor).resume(refunds(), again.checkpoint, APPROVE)


async def test_a_crash_in_a_resumed_superstep_keeps_the_decision() -> None:
    store = MemoryCheckpointStore()
    executor = approving()
    r = runner(store, executor=executor)
    first = await r.run(refunds(), {"order": "o-1"}, executor.context(run_id="run-1"))
    assert first.checkpoint is not None
    CRASH_AT["refund"] = 1
    with pytest.raises(Crash):
        await r.resume(refunds(), first.checkpoint, APPROVE)
    with pytest.raises(RecoveryNeedsDecision):
        await runner(store, executor=executor).recover(refunds(), "run-1")
    done = await runner(store, executor=executor).recover(refunds(), "run-1", rerun_in_doubt=True)
    assert (done.status, done.state.note) == ("done", "refunded o-1")
    assert dict(RAN) == {"charge": 1, "refund": 2}


async def test_a_zombie_worker_is_fenced_off_once_another_took_over() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry)
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def slow(state: Order, ctx: Any) -> dict[str, Any]:
        first = await executor.run("charge", {"order": state.order}, ctx)
        entered.set()
        await gate.wait()
        second = await executor.run("ship", {"order": state.order}, ctx)
        return {"log": [first, second]}

    flow = Flow("slow", state=Order).node("work", slow).node("end", lambda s: None)
    flow = flow.edge("work", "end").end("end")
    zombie = runner(store, executor=executor)
    task = asyncio.create_task(zombie.run(flow, {}, executor.context(run_id="run-1")))
    await entered.wait()
    gate.set()  # the new worker's node goes straight through
    taken = await runner(store, executor=executor).recover(flow, "run-1")
    assert taken.status == "done"
    with pytest.raises(FlowError) as caught:
        await task
    assert isinstance(caught.value.__cause__, CheckpointConflict) or "moved on" in str(caught.value)
    assert RAN["charge"] == 1  # replayed by the new worker from the zombie's log
    assert RAN["ship"] == 1  # the zombie never got to ship
    assert (await store.latest("run-1")).checkpoint.status == "done"  # type: ignore[union-attr]


async def test_an_owned_run_starts_again_on_recovery_and_ends_once() -> None:
    from tests.flow.durable_helpers import CrashingStore, three_steps

    store = CrashingStore()
    store.after = 1
    with pytest.raises(Crash):
        await runner(store).run(three_steps(), {"order": "o-1"})  # the runner owns the run
    [run_id] = [s.run_id for s in await store.list_unfinished()]
    sink = RecordingSink()
    await runner(store, sink=sink).recover(three_steps(), run_id)
    types = sink.types()
    assert types.index("run.started") < types.index("run.recovered")
    assert types.count("run.finished") == 1


async def test_a_tool_spec_marked_idempotent_in_the_graph_runs_again() -> None:
    from agent_adapter.core.tool import build_spec

    async def note(order: str) -> str:
        """Write a note (safe to repeat)."""
        RAN["note"] += 1
        if RAN["note"] == 1:
            raise Crash("note")
        return f"noted {order}"

    spec = build_spec(note, idempotent=True)
    flow = (
        Flow("spec", state=Order).node("n", Call(spec, args={"order": "o"}, into="note")).end("n")
    )
    store = MemoryCheckpointStore()
    await crashed(runner(store), flow)
    result = await runner(store).recover(flow, "run-1")
    assert result.state.note == "noted o"


async def test_resuming_a_paused_checkpoint_the_store_does_not_hold_is_refused() -> None:
    executor = approving()
    first = await runner(MemoryCheckpointStore(), executor=executor).run(
        refunds(), {"order": "o-1"}, executor.context(run_id="run-1")
    )
    assert first.checkpoint is not None
    with pytest.raises(CheckpointConflict, match="paused"):
        await runner(MemoryCheckpointStore(), executor=executor).resume(
            refunds(), first.checkpoint, APPROVE
        )


async def test_a_store_that_fails_to_mark_a_failed_run_keeps_the_runs_error(caplog: Any) -> None:
    class NoFinal(MemoryCheckpointStore):
        async def save(self, run_id: str, step: int, checkpoint: Any) -> None:
            if checkpoint.status == "failed":
                raise ConnectionError("down")
            await super().save(run_id, step, checkpoint)

    flow = Flow("bad", state=Order).node("a", Call("nope", args={})).end("a")
    r = runner(NoFinal())
    with pytest.raises(FlowError, match="nope"):
        await r.run(flow, {}, r.executor.context(run_id="run-1"))
    assert "failed run not marked in the store" in caplog.text
