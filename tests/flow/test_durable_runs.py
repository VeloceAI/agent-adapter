"""Crash-safe flow runs: a checkpoint per superstep, recovery, no repeated side effects."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_adapter.core.errors import (
    CheckpointConflict,
    CheckpointNotFound,
    CheckpointStoreCorrupt,
    ConfigError,
    FlowError,
    RecoveryNeedsDecision,
    ToolUsageError,
)
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import MemoryCheckpointStore, SQLiteCheckpointStore
from agent_adapter.flow import Call, Flow, FlowRunner
from tests.fakes.sinks import RecordingSink
from tests.flow.durable_helpers import (
    CRASH_AT,
    KEY,
    RAN,
    Crash,
    CrashingStore,
    Order,
    YieldingStore,
    node_fn,
    registry,
    reset,
    runner,
    three_steps,
)


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset()


async def crashed(r: FlowRunner, flow: Flow[Any], run_id: str = "run-1") -> None:
    with pytest.raises(Crash):
        await r.run(flow, {"order": "o-1"}, r.executor.context(run_id=run_id))


async def test_a_clean_run_saves_a_checkpoint_after_every_superstep() -> None:
    store = CrashingStore()
    sink = RecordingSink()
    r = runner(store, sink=sink)
    result = await r.run(three_steps(), {"order": "o-1"}, r.executor.context(run_id="run-1"))
    assert result.status == "done"
    assert store.saves == [(0, "running"), (1, "running"), (2, "running"), (3, "done")]
    saved = [e for e in sink.events if e.type == "checkpoint.saved"]
    assert [(e.data["step"], e.data["status"], e.data["superstep"]) for e in saved] == [
        (0, "running", 0),
        (1, "running", 1),
        (2, "running", 2),
        (3, "done", 3),
    ]
    assert await store.list_unfinished() == []


async def test_a_crash_between_supersteps_recovers_without_running_finished_steps() -> None:
    store = CrashingStore()
    store.after = 2  # dies right after the checkpoint of superstep 2
    r = runner(store)
    await crashed(r, three_steps())
    assert dict(RAN) == {"charge": 1, "ship": 1}
    assert [s.run_id for s in await store.list_unfinished()] == ["run-1"]
    result = await runner(store).recover(three_steps(), "run-1")
    assert (result.status, result.state.note) == ("done", "value o-1")
    assert dict(RAN) == {"charge": 1, "ship": 1, "lookup": 1}


async def test_a_crash_before_a_checkpoint_replays_the_finished_call_from_the_log() -> None:
    store = CrashingStore()
    store.before = 2  # superstep 2's call finished, then the process died before its save
    r = runner(store)
    await crashed(r, three_steps())
    result = await runner(store).recover(three_steps(), "run-1")
    assert result.status == "done"
    assert dict(RAN) == {"charge": 1, "ship": 1, "lookup": 1}  # ship replayed, not re-run


async def test_a_crash_inside_a_node_replays_its_finished_calls() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry)
    flow = Flow("two", state=Order).node("both", node_fn(executor, "charge", "lookup")).end("both")
    CRASH_AT["lookup"] = 1  # charge done; lookup dies inside its body
    r = runner(store, executor=executor)
    await crashed(r, flow)
    result = await runner(store, executor=executor).recover(flow, "run-1")
    assert result.state.log == ["charged o-1", "value o-1"]
    assert dict(RAN) == {"charge": 1, "lookup": 2}  # lookup is idempotent: it ran again


async def test_a_call_in_doubt_of_a_tool_not_idempotent_needs_a_decision() -> None:
    store = MemoryCheckpointStore()
    CRASH_AT["ship"] = 1
    await crashed(runner(store), three_steps())
    with pytest.raises(RecoveryNeedsDecision) as caught:
        await runner(store).recover(three_steps(), "run-1")
    assert caught.value.calls == (("ship", "started, no result logged"),)
    assert caught.value.run_id == "run-1"
    assert (await store.latest("run-1")).checkpoint.status == "running"  # type: ignore[union-attr]
    result = await runner(store).recover(three_steps(), "run-1", rerun_in_doubt=True)
    assert result.status == "done"
    assert dict(RAN) == {"charge": 1, "ship": 2, "lookup": 1}


async def test_two_workers_recovering_one_run_only_one_continues() -> None:
    store = YieldingStore()
    store_crash = CrashingStore()
    store_crash.after = 1
    await crashed(runner(store_crash), three_steps())
    store._runs = store_crash._runs  # the same stored run, read through a yielding store
    outcomes = await asyncio.gather(
        runner(store).recover(three_steps(), "run-1"),
        runner(store).recover(three_steps(), "run-1"),
        return_exceptions=True,
    )
    done = [o for o in outcomes if not isinstance(o, BaseException)]
    assert len(done) == 1 and done[0].status == "done"
    assert [type(o) for o in outcomes if isinstance(o, BaseException)] == [CheckpointConflict]
    assert dict(RAN) == {"charge": 1, "ship": 1, "lookup": 1}


async def test_recover_all_reports_each_run_and_skips_ones_another_worker_holds() -> None:
    store = MemoryCheckpointStore()
    CRASH_AT["ship"] = 1
    await crashed(runner(store), three_steps(), "run-a")
    CRASH_AT.clear()
    store_b = CrashingStore()
    store_b.after = 1
    out = await runner(store).recover_all(three_steps())
    assert isinstance(out["run-a"], RecoveryNeedsDecision)
    out = await runner(store).recover_all(three_steps(), rerun_in_doubt=True)
    assert out["run-a"].status == "done"  # type: ignore[union-attr]
    assert await runner(store).recover_all(three_steps()) == {}
    del store_b


async def test_a_changed_graph_refuses_to_recover() -> None:
    store = CrashingStore()
    store.after = 1
    await crashed(runner(store), three_steps())
    changed = (
        Flow("checkout", state=Order)
        .node("pay", Call("charge", args=lambda s: {"order": s.order}, into="note"))
        .node("send", Call("ship", args=lambda s: {"order": s.order}, into="note"))
        .edge("pay", "send")
        .end("send")
    )
    with pytest.raises(FlowError, match="another version"):
        await runner(store).recover(changed, "run-1")


async def test_another_key_or_a_changed_document_is_store_corruption() -> None:
    store = CrashingStore()
    store.after = 1
    await crashed(runner(store), three_steps())
    other = FlowRunner(Executor(registry), checkpoints=store, checkpoint_key=b"x" * 32)
    with pytest.raises(CheckpointStoreCorrupt, match="MAC"):
        await other.recover(three_steps(), "run-1")
    found = await store.latest("run-1")
    assert found is not None
    tampered = found.checkpoint.data.replace("o-1", "o-9")
    store._runs["run-1"] = found.model_copy(
        update={"checkpoint": found.checkpoint.model_copy(update={"data": tampered})}
    )
    with pytest.raises(CheckpointStoreCorrupt, match="MAC"):
        await runner(store).recover(three_steps(), "run-1")
    store._runs["run-1"] = found.model_copy(
        update={"checkpoint": found.checkpoint.model_copy(update={"data": "{not json"})}
    )
    with pytest.raises(CheckpointStoreCorrupt, match="not JSON"):
        await runner(store).recover(three_steps(), "run-1")


async def test_recover_refuses_unknown_ended_and_foreign_runs() -> None:
    store = MemoryCheckpointStore()
    r = runner(store)
    with pytest.raises(CheckpointNotFound):
        await r.recover(three_steps(), "nope")
    await r.run(three_steps(), {}, r.executor.context(run_id="done-run"))
    with pytest.raises(CheckpointConflict, match="already ended"):
        await r.recover(three_steps(), "done-run")
    other = Flow("other", state=Order).node("a", lambda s: None).end("a")
    with pytest.raises(CheckpointConflict, match="not flow 'other'"):
        await r.recover(other, "done-run")
    with pytest.raises(ToolUsageError, match="not the recovered run"):
        await r.recover(three_steps(), "done-run", r.executor.context(run_id="else"))


async def test_a_run_id_the_store_already_holds_is_refused() -> None:
    store = MemoryCheckpointStore()
    r = runner(store)
    await r.run(three_steps(), {}, r.executor.context(run_id="run-1"))
    with pytest.raises(CheckpointConflict):
        await r.run(three_steps(), {}, r.executor.context(run_id="run-1"))
    assert RAN["charge"] == 1


async def test_a_failed_node_marks_the_run_failed() -> None:
    store = MemoryCheckpointStore()
    flow = Flow("bad", state=Order).node("a", Call("nope", args={})).end("a")
    r = runner(store)
    with pytest.raises(FlowError):
        await r.run(flow, {}, r.executor.context(run_id="run-1"))
    assert (await store.latest("run-1")).checkpoint.status == "failed"  # type: ignore[union-attr]
    with pytest.raises(CheckpointConflict, match="failed"):
        await r.recover(flow, "run-1")


async def test_a_run_whose_store_broke_stays_recoverable() -> None:
    class Flaky(MemoryCheckpointStore):
        down = False

        async def record_call(self, *args: Any) -> None:
            if self.down:
                raise ConnectionError("store down")
            await super().record_call(*args)

    store = Flaky()
    store.down = True
    with pytest.raises(FlowError):
        r = runner(store)
        await r.run(three_steps(), {}, r.executor.context(run_id="run-1"))
    assert RAN["charge"] == 0  # the call was never logged, so it never ran
    assert (await store.latest("run-1")).checkpoint.status == "running"  # type: ignore[union-attr]
    store.down = False
    assert (await runner(store).recover(three_steps(), "run-1")).status == "done"


async def test_a_zombie_worker_is_fenced_off_once_another_recovered_the_run() -> None:
    store = MemoryCheckpointStore()
    zombie = runner(store)
    first = await zombie.run(three_steps(), {}, zombie.executor.context(run_id="run-0"))
    assert first.status == "done"
    found = await store.latest("run-0")
    assert found is not None
    log_owner = runner(store)
    with pytest.raises(CheckpointConflict, match="ended"):
        await log_owner.recover(three_steps(), "run-0")


async def test_unsealable_state_is_refused_before_anything_runs() -> None:
    from pydantic import BaseModel

    class Loose(BaseModel):
        thing: Any = None

    flow = Flow("loose", state=Loose).node("a", Call("charge", args={"order": "x"})).end("a")
    r = runner(MemoryCheckpointStore())
    with pytest.raises(FlowError, match="state-not-sealable"):
        await r.run(flow, {"thing": object()}, r.executor.context(run_id="run-1"))
    assert RAN["charge"] == 0


def test_a_durable_store_needs_a_key_and_a_store_must_be_one() -> None:
    with pytest.raises(ConfigError, match="checkpoint_key"):
        FlowRunner(Executor(registry), checkpoints=SQLiteCheckpointStore("x.db"))
    with pytest.raises(ToolUsageError, match="CheckpointStore"):
        FlowRunner(Executor(registry), checkpoints=object())  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="16 bytes"):
        FlowRunner(Executor(registry), checkpoints=MemoryCheckpointStore(), checkpoint_key=b"x")
    FlowRunner(Executor(registry), checkpoints=MemoryCheckpointStore())  # in-process key


async def test_recover_needs_a_store() -> None:
    plain = FlowRunner(Executor(registry))
    with pytest.raises(FlowError, match="no checkpoints"):
        await plain.recover(three_steps(), "run-1")
    with pytest.raises(FlowError, match="no checkpoints"):
        await plain.recover_all(three_steps())


async def test_recovery_emits_run_recovered_with_counts() -> None:
    store = CrashingStore()
    store.before = 2
    await crashed(runner(store), three_steps())
    sink = RecordingSink()
    await runner(store, sink=sink).recover(three_steps(), "run-1")
    [event] = [e for e in sink.events if e.type == "run.recovered"]
    assert event.run_id == "run-1"
    assert (event.data["kind"], event.data["replayed"], event.data["rerun"]) == ("flow", 1, 0)
    assert event.data["flow"] == "checkout"
    assert not any(e.type == "run.started" for e in sink.events)  # the caller owns the run


async def test_a_sqlite_store_recovers_in_a_fresh_runner(tmp_path: Any) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.db")
    executor = Executor(registry)
    flow = Flow("two", state=Order).node("both", node_fn(executor, "charge", "ship")).end("both")
    CRASH_AT["ship"] = 1
    r = FlowRunner(executor, checkpoints=store, checkpoint_key=KEY)
    await crashed(r, flow)
    fresh = FlowRunner(
        Executor(registry),
        checkpoints=SQLiteCheckpointStore(tmp_path / "runs.db"),
        checkpoint_key=KEY,
    )
    with pytest.raises(RecoveryNeedsDecision):
        await fresh.recover(flow, "run-1")
    result = await fresh.recover(flow, "run-1", rerun_in_doubt=True)
    assert result.state.log == ["charged o-1", "shipped o-1"]
    assert dict(RAN) == {"charge": 1, "ship": 2}
