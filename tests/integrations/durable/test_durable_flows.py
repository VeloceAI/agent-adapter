"""DurableFlows: one superstep per call, crash replay, one-use decisions, store as the truth."""

from __future__ import annotations

import pytest

from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ApprovalPending,
    CheckpointConflict,
    CheckpointNotFound,
    FlowError,
    RecoveryNeedsDecision,
    ToolUsageError,
)
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import MemoryCheckpointStore
from agent_adapter.flow import FlowRunner
from agent_adapter.integrations._durable_engine import failure_message, retryable
from agent_adapter.integrations._durable_flows import (
    DurableFlows,
)
from agent_adapter.tracing.sinks import EventLog
from tests.integrations.durable.kit import (
    CRASH_AT,
    KEY,
    RAN,
    Crash,
    decide,
    durable,
    gated,
    orders,
    registry,
)


async def _drive(flows: DurableFlows, flow: str, run_id: str, state: dict | None = None):
    report = await flows.advance(flow, run_id, state)
    seen = [report.status]
    while report.status == "running":
        report = await flows.advance(flow, run_id)
        seen.append(report.status)
    return report, seen


async def test_each_advance_runs_one_superstep_then_the_run_ends_done():
    flows, log = durable()
    first = await flows.advance("orders", "r1", {"order": "o-9"})
    assert (first.status, first.steps, dict(RAN)) == ("running", 1, {"charge": 1})
    done = await flows.advance("orders", "r1")
    assert done.status == "done" and done.steps == 2
    assert done.state == {"order": "o-9", "note": "charged o-9", "sent": "shipped o-9"}
    again = await flows.advance("orders", "r1")  # a retried step after the end: read back
    assert again.status == "done" and again.state == done.state
    assert dict(RAN) == {"charge": 1, "ship": 1}
    types = [e.type for e in log.events if e.type.startswith(("run.", "flow."))]
    assert types == ["run.started", "flow.started", "flow.finished", "run.finished"]


async def test_a_crash_inside_a_step_retries_it_and_finished_calls_replay():
    flows, _ = durable()
    await flows.advance("orders", "r2", {"order": "o-1"})
    CRASH_AT["ship"] = 1
    with pytest.raises(Crash):
        await flows.advance("orders", "r2")
    with pytest.raises(RecoveryNeedsDecision):  # ship may have run: fail closed
        await flows.advance("orders", "r2")
    assert RAN["ship"] == 1 and RAN["charge"] == 1


async def test_a_crash_after_a_call_finished_replays_it_on_the_retry():
    flows, log = durable()
    calls = {"n": 0}
    original = flows.store.save

    async def dies_on_the_step_save(run_id, step, checkpoint):
        calls["n"] += 1
        if calls["n"] == 4:  # start, step 1, claim, then the final save: the worker dies
            raise Crash("store")
        await original(run_id, step, checkpoint)

    flows.store.save = dies_on_the_step_save  # type: ignore[method-assign]
    await flows.advance("orders", "r3", {})
    with pytest.raises(Crash):
        await flows.advance("orders", "r3")
    flows.store.save = original  # type: ignore[method-assign]
    report, _ = await _drive(flows, "orders", "r3")
    assert report.status == "done" and RAN["ship"] == 1  # replayed, not run again
    recovered = log.of_type("run.recovered")
    assert len(recovered) == 1 and recovered[0].data["replayed"] == 1


async def test_a_gate_resumes_only_from_a_store_decision_and_only_once():
    flows, _ = durable()
    paused = await flows.advance("gated", "g1", {"order": "o-2"})
    assert paused.status == "paused" and paused.pending is not None
    pending = paused.pending
    assert (pending.kind, pending.node) == ("node", "check")
    assert (await flows.advance("gated", "g1")).status == "paused"  # waiting stays waiting
    request = await flows.pending_request("gated", "g1")
    assert request is not None and request.approval_id == pending.approval_id
    forged = await flows.resume("gated", "g1", pending.approval_id)  # a notice, no decision
    assert forged.status == "paused" and RAN["ship"] == 0
    await decide(flows, pending.approval_id, True, decided_by="ana")
    step = await flows.resume("gated", "g1", pending.approval_id)
    assert step.status == "running"
    retried = await flows.resume("gated", "g1", pending.approval_id)  # a retried resume continues
    assert retried.status == "done" and retried.state["sent"] == "shipped o-2"
    assert await flows.pending_request("gated", "g1") is None
    assert (await flows.resume("gated", "g1", pending.approval_id)).status == "done"
    store = flows.runner.executor.approval.store
    assert (await store.get(pending.approval_id)).consumed  # used once
    assert not await store.consume(pending.approval_id)
    assert RAN["ship"] == 1


async def test_a_denied_gate_ends_the_run_denied():
    flows, _ = durable()
    paused = await flows.advance("gated", "g2", {})
    await decide(flows, paused.pending.approval_id, False, reason="no")
    denied = await flows.resume("gated", "g2", paused.pending.approval_id)
    assert denied.status == "denied" and denied.denied_at == "check" and RAN["ship"] == 0


async def test_a_timeout_records_a_denial_unless_the_store_has_a_decision():
    flows, _ = durable()
    paused = await flows.advance("gated", "g5", {})
    approval = paused.pending.approval_id
    denied = await flows.resume("gated", "g5", approval, timed_out=True)
    assert denied.status == "denied"
    record = await flows.runner.executor.approval.store.get(approval)
    assert record.decision.decided_by == "timeout" and record.consumed
    late = await flows.advance("gated", "g6", {})
    await decide(flows, late.pending.approval_id, True)  # decided just before the timeout
    report = await flows.resume("gated", "g6", late.pending.approval_id, timed_out=True)
    assert report.status == "running"


async def test_a_tool_approval_pauses_without_its_arguments_and_resumes():
    flows, _ = durable()
    paused = await flows.advance("refunds", "t1", {"order": "o-5"})
    assert paused.pending.kind == "tool" and paused.pending.tool == "refund"
    assert "o-5" not in paused.model_dump_json()  # arguments stay out of engine history
    await decide(flows, paused.pending.approval_id, True)
    done = await flows.resume("refunds", "t1", paused.pending.approval_id)
    assert done.status == "done" and RAN["refund"] == 1


async def test_a_notice_for_another_approval_or_an_unknown_record_keeps_waiting():
    flows, _ = durable()
    paused = await flows.advance("gated", "g3", {})
    assert (await flows.resume("gated", "g3", "apr_other")).status == "paused"
    store = flows.runner.executor.approval.store

    async def no_record(approval_id):
        raise ApprovalNotFound(approval_id)

    store.get = no_record  # a store that lost (or never had) the record
    still = await flows.resume("gated", "g3", paused.pending.approval_id)
    assert still.status == "paused"
    with pytest.raises(ToolUsageError, match="approval_id"):
        await flows.resume("gated", "g3", "")
    assert RAN["ship"] == 0


class _StorelessPending:
    """A port that parks every request but keeps no store (not a DeferredApproval)."""

    async def request(self, request):
        raise ApprovalPending(request.approval_id or "apr_x")


async def test_a_used_record_is_refused_and_a_port_without_a_store_fails_closed():
    flows, _ = durable()
    paused = await flows.advance("gated", "g7", {})
    approval = paused.pending.approval_id
    store = flows.runner.executor.approval.store
    await decide(flows, approval, True)
    assert await store.consume(approval)  # someone else used the decision
    with pytest.raises(ApprovalAlreadyDecided):
        await flows.resume("gated", "g7", approval)
    runner = FlowRunner(
        Executor(registry, approval=_StorelessPending()),
        checkpoints=MemoryCheckpointStore(),
        checkpoint_key=KEY,
    )
    bare = DurableFlows(runner, [gated()])
    waiting = await bare.advance("gated", "g8", {})
    with pytest.raises(ToolUsageError, match="DeferredApproval"):
        await bare.resume("gated", "g8", waiting.pending.approval_id)


async def test_resume_of_an_unknown_run_raises_not_found():
    flows, _ = durable()
    with pytest.raises(CheckpointNotFound):
        await flows.resume("gated", "nope", "a")


async def test_a_failed_node_marks_the_run_failed_and_ends_it():
    flows, log = durable()
    with pytest.raises(FlowError):
        await flows.advance("broken", "b1", {})
    assert (await flows.advance("broken", "b1")).status == "failed"
    assert log.of_type("run.finished")[-1].data == {"ok": False}


async def test_inputs_are_checked():
    flows, _ = durable()
    with pytest.raises(ToolUsageError, match="initial state"):
        await flows.advance("orders", "new")
    with pytest.raises(ToolUsageError, match="no durable flow"):
        await flows.advance("missing", "r", {})
    with pytest.raises(ToolUsageError, match="run_id"):
        await flows.advance("orders", "", {})
    await flows.advance("orders", "shared", {})
    with pytest.raises(ToolUsageError, match="another flow") as held:  # permanent: no retry
        await flows.advance("gated", "shared", {})
    assert not retryable(held.value)
    with pytest.raises(ToolUsageError, match="RunIdentity"):
        await flows.advance("orders", "shared", identity={"user_id": "u"})  # type: ignore[arg-type]
    assert flows.names[:4] == ("orders", "gated", "refunds", "broken")


def test_construction_is_checked():
    plain = FlowRunner(Executor(registry))
    with pytest.raises(ToolUsageError, match="checkpoints"):
        DurableFlows(plain, [orders()])
    runner = FlowRunner(Executor(registry), checkpoints=MemoryCheckpointStore(), checkpoint_key=KEY)
    with pytest.raises(ToolUsageError, match="at least one"):
        DurableFlows(runner, [])
    with pytest.raises(ToolUsageError, match="two flows"):
        DurableFlows(runner, [orders(), orders()])
    with pytest.raises(ToolUsageError, match="Flow objects"):
        DurableFlows(runner, ["orders"])  # type: ignore[list-item]
    with pytest.raises(ToolUsageError, match="FlowRunner"):
        DurableFlows(object(), [orders()])  # type: ignore[arg-type]


async def test_the_runners_own_sink_gets_the_events_once():
    extra = EventLog()
    executor = Executor(registry)
    store = MemoryCheckpointStore()
    runner = FlowRunner(executor, checkpoints=store, checkpoint_key=KEY, sink=extra)
    flows = DurableFlows(runner, [orders()])
    report, _ = await _drive(flows, "orders", "s1", {})
    assert report.status == "done" and runner.checkpoints is store  # the runner is not changed
    assert len(extra.of_type("flow.started")) == 1
    same = FlowRunner(executor, checkpoints=store, checkpoint_key=KEY, sink=executor.sink)
    report, _ = await _drive(DurableFlows(same, [orders()]), "orders", "s2", {})
    assert report.status == "done"


async def test_the_viewer_graph_reaches_a_graph_sink():
    class Taker(EventLog):
        def __init__(self) -> None:
            super().__init__()
            self.graphs: list[str] = []

        def take_flow_graph(self, run_id, flow, graph):
            self.graphs.append(flow)

    taker = Taker()
    runner = FlowRunner(Executor(registry, sink=taker), checkpoints=MemoryCheckpointStore())
    await _drive(DurableFlows(runner, [orders()]), "orders", "v1", {})
    assert taker.graphs and set(taker.graphs) == {"orders"}


def test_retryable_and_failure_message():
    assert retryable(CheckpointConflict("r", "m")) and retryable(OSError("disk"))
    assert not retryable(FlowError("secret text")) and not retryable(KeyboardInterrupt())
    assert failure_message(FlowError("secret text")) == "FlowError (flow-error)"
    assert failure_message(OSError("secret")) == "OSError"


def test_no_pending_request_is_no_pending_approval():
    from agent_adapter.integrations._durable_flows import _pending

    assert _pending(None) is None


async def test_a_failed_status_read_keeps_the_run_and_logs(caplog):
    flows, _ = durable()
    reads = {"n": 0}
    original = flows.store.latest

    async def fails_after_the_first_read(run_id):
        reads["n"] += 1
        if reads["n"] > 1:
            raise OSError("down")
        return await original(run_id)

    flows.store.latest = fails_after_the_first_read  # type: ignore[method-assign]
    with pytest.raises(FlowError):
        await flows.advance("broken", "f1", {})
    assert "status not read back" in caplog.text and "down" not in caplog.text
