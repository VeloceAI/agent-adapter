"""Crash-safe tool loops: a checkpoint per step, `recover_tool_loop`, no repeated calls."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import (
    CheckpointConflict,
    CheckpointNotFound,
    CheckpointStoreCorrupt,
    ConfigError,
    RecoveryNeedsDecision,
    ToolUsageError,
)
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor, recover_tool_loop, run_tool_loop
from agent_adapter.exec.checkpoints import Checkpoint, MemoryCheckpointStore, SQLiteCheckpointStore
from agent_adapter.flow import Flow, FlowRunner
from tests.fakes.models import ScriptedModel, calls, text
from tests.fakes.sinks import RecordingSink
from tests.flow.durable_helpers import (
    CRASH_AT,
    KEY,
    RAN,
    Crash,
    CrashingStore,
    Order,
    registry,
    reset,
)

USER = (Message(role="user", content="close order o-1"),)
BOTH = calls(("charge", {"order": "o-1"}), ("ship", {"order": "o-1"}))


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset()


async def loop(store: Any, model: ScriptedModel, **options: Any) -> Any:
    options.setdefault("run_id", "run-1")
    return await run_tool_loop(
        model, registry, USER, checkpoints=store, checkpoint_key=KEY, **options
    )


async def recover(store: Any, model: ScriptedModel, **options: Any) -> Any:
    return await recover_tool_loop(
        model, registry, "run-1", checkpoints=store, checkpoint_key=KEY, **options
    )


async def test_a_clean_loop_saves_the_history_after_each_step() -> None:
    store = CrashingStore()
    sink = RecordingSink()
    executor = Executor(registry, sink=sink)
    result = await loop(store, ScriptedModel([BOTH, text("done")]), executor=executor)
    assert result.finish == "stop"
    assert store.saves == [(0, "running"), (1, "running"), (2, "running"), (3, "done")]
    saved = [e.data for e in sink.events if e.type == "checkpoint.saved"]
    assert (saved[0]["kind"], saved[0]["step"], saved[0]["stage"]) == ("loop", 0, "tool_loop")
    assert dict(RAN) == {"charge": 1, "ship": 1}


async def test_a_crash_inside_a_tool_keeps_the_finished_call_and_asks_about_the_other() -> None:
    store = MemoryCheckpointStore()
    CRASH_AT["ship"] = 1
    with pytest.raises(Crash):
        await loop(store, ScriptedModel([BOTH]))
    with pytest.raises(RecoveryNeedsDecision) as caught:
        await recover(store, ScriptedModel([]))
    assert [tool for tool, _ in caught.value.calls] == ["ship"]
    model = ScriptedModel([text("closed")])
    result = await recover(store, model, rerun_in_doubt=True)
    assert (result.finish, result.final.text, result.steps) == ("stop", "closed", 2)
    assert dict(RAN) == {"charge": 1, "ship": 2}
    tool_messages = [m.content for m in result.messages if m.role == "tool"]
    assert tool_messages == ["charged o-1", "shipped o-1"]
    assert len(model.requests) == 1


async def test_a_crash_in_the_next_model_call_never_runs_the_tools_again() -> None:
    store = MemoryCheckpointStore()
    with pytest.raises(Crash):
        await loop(store, ScriptedModel([BOTH, Crash("model call")]))
    result = await recover(store, ScriptedModel([text("closed")]))
    assert result.final.text == "closed"
    assert dict(RAN) == {"charge": 1, "ship": 1}


async def test_a_crash_before_the_result_save_replays_the_logged_calls() -> None:
    store = CrashingStore()
    store.before = 2  # both calls finished; the history with their results was never saved
    with pytest.raises(Crash):
        await loop(store, ScriptedModel([BOTH]))
    sink = RecordingSink()
    result = await recover(
        store, ScriptedModel([text("ok")]), executor=Executor(registry, sink=sink)
    )
    assert result.final.text == "ok"
    assert dict(RAN) == {"charge": 1, "ship": 1}
    [event] = [e for e in sink.events if e.type == "run.recovered"]
    assert (event.data["kind"], event.data["replayed"], event.data["rerun"]) == ("loop", 2, 0)


async def test_max_steps_counts_the_model_calls_before_the_crash() -> None:
    store = MemoryCheckpointStore()
    with pytest.raises(Crash):
        await loop(store, ScriptedModel([BOTH, Crash("down")]), max_steps=2)
    result = await recover(store, ScriptedModel([BOTH]), max_steps=2)
    assert (result.finish, result.steps) == ("max_steps", 2)


async def test_a_paused_loop_resumes_through_run_tool_loop_with_the_same_run_id() -> None:
    store = MemoryCheckpointStore()
    approvals = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(approvals))
    first = await loop(
        store, ScriptedModel([calls(("refund", {"order": "o-1"}))]), executor=executor
    )
    assert first.finish == "pending"
    assert (await store.latest("run-1")).checkpoint.status == "paused"  # type: ignore[union-attr]
    [pending] = await approvals.list(status="pending")
    await approvals.decide(pending.approval_id, ApprovalDecision(approved=True))
    done = await run_tool_loop(
        ScriptedModel([text("refunded")]),
        registry,
        first.messages,
        executor=executor,
        run_id="run-1",
        checkpoints=store,
        checkpoint_key=KEY,
    )
    assert done.final.text == "refunded"
    assert RAN["refund"] == 1
    assert (await store.latest("run-1")).checkpoint.status == "done"  # type: ignore[union-attr]


async def test_a_running_or_ended_run_id_is_refused() -> None:
    store = MemoryCheckpointStore()
    with pytest.raises(Crash):
        await loop(store, ScriptedModel([Crash("down")]))
    with pytest.raises(CheckpointConflict, match="recover_tool_loop"):
        await loop(store, ScriptedModel([text("x")]))
    await recover(store, ScriptedModel([text("x")]))
    with pytest.raises(CheckpointConflict, match="already ended"):
        await recover(store, ScriptedModel([text("x")]))


async def test_recover_checks_what_the_store_holds() -> None:
    store = MemoryCheckpointStore()
    with pytest.raises(CheckpointNotFound):
        await recover(store, ScriptedModel([]))
    flow_cp = Checkpoint(kind="flow", name="checkout", status="running", segment="s", data="{}")
    await store.save("run-1", 0, flow_cp)
    with pytest.raises(CheckpointConflict, match="not tool loop"):
        await recover(store, ScriptedModel([]))
    other = MemoryCheckpointStore()
    with pytest.raises(Crash):
        await loop(other, ScriptedModel([Crash("down")]))
    with pytest.raises(CheckpointConflict, match="not tool loop 'chat'"):
        await recover(other, ScriptedModel([]), stage="chat")
    with pytest.raises(CheckpointStoreCorrupt, match="MAC"):
        await recover_tool_loop(
            ScriptedModel([]), registry, "run-1", checkpoints=other, checkpoint_key=b"z" * 32
        )


async def test_a_tampered_or_odd_loop_document_is_corruption() -> None:
    from agent_adapter.exec._durable_seal import seal_checkpoint

    store = MemoryCheckpointStore()
    bad = {"stage": "tool_loop", "steps": -1, "messages": []}
    row = Checkpoint(kind="loop", name="tool_loop", status="running", segment="s", data="")
    identity = {"user_id": "", "thread_id": "", "principal": ""}
    await store.save("run-1", 0, seal_checkpoint(KEY, "run-1", 0, row, bad, identity))
    with pytest.raises(CheckpointStoreCorrupt, match="unreadable"):
        await recover(store, ScriptedModel([]))


async def test_durable_options_are_checked() -> None:
    with pytest.raises(ToolUsageError, match="needs run_id"):
        await run_tool_loop(ScriptedModel([]), registry, USER, checkpoints=MemoryCheckpointStore())
    with pytest.raises(ConfigError, match="checkpoint_key"):
        await run_tool_loop(
            ScriptedModel([]), registry, USER, run_id="r", checkpoints=SQLiteCheckpointStore("x.db")
        )
    executor = Executor(registry)
    with pytest.raises(ToolUsageError, match=r"is not ctx\.run_id"):
        await run_tool_loop(
            ScriptedModel([]), registry, USER, run_id="a", ctx=executor.context(run_id="b")
        )


async def test_a_durable_loop_inside_a_flow_node_is_refused() -> None:
    async def agent(state: Order, ctx: Any) -> dict[str, Any]:
        await run_tool_loop(
            ScriptedModel([text("x")]),
            registry,
            USER,
            ctx=ctx,
            checkpoints=MemoryCheckpointStore(),
        )
        return {}

    flow = Flow("agent", state=Order).node("think", agent).end("think")
    runner = FlowRunner(Executor(registry), checkpoints=MemoryCheckpointStore())
    with pytest.raises(Exception, match="already cover"):
        await runner.run(flow, {})


async def test_recover_tool_loop_sends_a_paused_loop_to_run_tool_loop() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    first = await loop(
        store, ScriptedModel([calls(("refund", {"order": "o-1"}))]), executor=executor
    )
    assert first.finish == "pending"
    with pytest.raises(CheckpointConflict, match="run_tool_loop"):
        await recover(store, ScriptedModel([]), executor=executor)


async def test_a_recovered_loop_keeps_its_user_and_refuses_another() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry)
    ana = executor.context(run_id="run-1", user_id="ana")
    with pytest.raises(Crash):
        await run_tool_loop(
            ScriptedModel([Crash("down")]),
            registry,
            USER,
            ctx=ana,
            checkpoints=store,
            checkpoint_key=KEY,
        )
    bob = executor.context(run_id="run-1", user_id="bob")
    with pytest.raises(ToolUsageError, match="another user_id"):
        await recover(store, ScriptedModel([]), ctx=bob)
    sink = RecordingSink()
    result = await recover(
        store, ScriptedModel([text("hi")]), executor=Executor(registry, sink=sink)
    )
    assert result.final.text == "hi"
    assert all(e.run_id == "run-1" for e in sink.events)


async def test_a_paused_loop_of_another_user_is_not_resumed() -> None:
    store = MemoryCheckpointStore()
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    ana = executor.context(run_id="run-1", user_id="ana")
    first = await run_tool_loop(
        ScriptedModel([calls(("refund", {"order": "o-1"}))]),
        registry,
        USER,
        ctx=ana,
        executor=executor,
        checkpoints=store,
        checkpoint_key=KEY,
    )
    with pytest.raises(ToolUsageError, match="another user_id"):
        await run_tool_loop(
            ScriptedModel([]),
            registry,
            first.messages,
            ctx=executor.context(run_id="run-1", user_id="bob"),
            checkpoints=store,
            checkpoint_key=KEY,
        )


async def test_a_cut_off_loop_worker_cannot_log_after_a_claim() -> None:
    store = MemoryCheckpointStore()
    with pytest.raises(Crash):
        await loop(store, ScriptedModel([BOTH, Crash("down")]))
    stale = (await store.latest("run-1")).checkpoint.segment  # type: ignore[union-attr]
    await recover(store, ScriptedModel([text("ok")]))
    from agent_adapter.exec.checkpoints import CallRecord

    with pytest.raises(CheckpointConflict, match="stale"):
        await store.record_call("run-1", stale, CallRecord(key="k", tool="t", state="started"))
