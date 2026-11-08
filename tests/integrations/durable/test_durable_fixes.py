"""DurableFlows: per-superstep events, replay checks, identity, idempotency keys, resume crashes."""

from __future__ import annotations

import logging

import pytest

from agent_adapter.core.errors import RecoveryNeedsDecision, ToolUsageError
from agent_adapter.integrations._durable_engine import retryable
from agent_adapter.integrations._durable_flows import ApprovalNotice, RunIdentity
from tests.integrations.durable.kit import CRASH_AT, DRIFT, KEYS, RAN, Crash, decide, durable


async def _finish(flows, flow, run_id, **kwargs):
    report = await flows.advance(flow, run_id, **kwargs)
    while report.status == "running":
        report = await flows.advance(flow, run_id, identity=kwargs.get("identity"))
    return report


def _blip_once(flows, when):
    original = flows.store.save
    left = {"n": 1}

    async def save(run_id, step, checkpoint):
        if left["n"] and when(step, checkpoint):
            left["n"] = 0
            raise OSError("store down")
        await original(run_id, step, checkpoint)

    flows.store.save = save


async def test_every_superstep_saves_and_reports_its_checkpoint():
    flows, log = durable()
    assert (await _finish(flows, "orders", "c1", state={})).status == "done"
    saved = log.of_type("checkpoint.saved")
    assert [e.data["status"] for e in saved][-1] == "done"
    assert {1, 2} <= {e.data["superstep"] for e in saved}
    assert [e.data["superstep"] for e in saved if e.data["status"] == "running"].count(1) >= 1


async def test_a_recovered_call_that_did_not_replay_is_reported(caplog):
    flows, log = durable()
    _blip_once(flows, lambda step, cp: cp.status == "running" and step == 1)
    with pytest.raises(OSError):
        await flows.advance("drifting", "d1", {})  # ship ran; its checkpoint was not saved
    DRIFT["n"] = 1  # the node now asks with other arguments
    with caplog.at_level(logging.WARNING):
        assert (await _finish(flows, "drifting", "d1")).status == "done"
    assert RAN["ship"] == 2  # the documented hazard: arguments must be deterministic
    assert any(e.data.get("unreplayed") for e in log.of_type("checkpoint.saved"))
    assert "o-1-0" not in caplog.text


async def test_an_idempotent_call_cut_short_reruns_with_the_same_key():
    flows, _ = durable()
    CRASH_AT["lookup"] = 1
    await flows.advance("payments", "k1", {})
    with pytest.raises(Crash):
        await flows.advance("payments", "k1")
    assert (await _finish(flows, "payments", "k1")).status == "done"
    pay, first, second = KEYS
    assert first is not None and first == second and pay != first
    assert RAN["pay"] == 1 and RAN["lookup"] == 2


async def test_a_non_idempotent_call_cut_short_fails_closed_for_good():
    flows, _ = durable()
    CRASH_AT["pay"] = 1
    with pytest.raises(Crash):
        await flows.advance("payments", "k2", {})
    with pytest.raises(RecoveryNeedsDecision) as stopped:
        await flows.advance("payments", "k2")
    assert not retryable(stopped.value) and RAN["pay"] == 1


async def test_a_run_keeps_its_identity_and_refuses_another():
    flows, _ = durable()
    ana = RunIdentity(user_id="ana", thread_id="t-1", principal="desk")
    first = await flows.advance("orders", "i1", {}, ana)
    assert first.status == "running"
    for other in (None, RunIdentity(user_id="bob", thread_id="t-1", principal="desk")):
        with pytest.raises(ToolUsageError) as refused:
            await flows.advance("orders", "i1", identity=other)
        assert not retryable(refused.value)
    assert (await flows.advance("orders", "i1", identity=ana)).status == "done"
    with pytest.raises(ToolUsageError, match="identity"):
        RunIdentity.of({"user_id": 3})
    assert RunIdentity.of(None) == RunIdentity()


async def test_a_resume_cut_off_after_the_decision_save_continues_once():
    flows, _ = durable()
    paused = await flows.advance("gated", "s1", {})
    answer = paused.pending.approval_id
    await decide(flows, answer, True)
    original = flows.store.save
    cut = {"left": 1}

    async def dies_after_the_decision_save(run_id, step, checkpoint):
        await original(run_id, step, checkpoint)
        if cut["left"] and checkpoint.status == "running":
            cut["left"] = 0
            raise Crash("after the decision save")

    flows.store.save = dies_after_the_decision_save
    with pytest.raises(Crash):
        await flows.resume("gated", "s1", answer)
    flows.store.save = original
    again = await flows.resume("gated", "s1", answer)  # the engine retries the resume
    while again.status == "running":
        again = await flows.advance("gated", "s1")
    assert again.status == "done" and RAN["ship"] == 1


async def test_a_notice_carries_an_approval_id_and_nothing_else():
    for bad in ({"approval_id": "a", "approved": True}, {"approval_id": ""}, {"approval_id": 3}):
        with pytest.raises(ValueError):
            ApprovalNotice.model_validate(bad)
    assert ApprovalNotice(approval_id="a").approval_id == "a"


async def test_engine_input_is_checked_before_anything_runs():
    from agent_adapter.integrations._durable_engine import advanced

    flows, _ = durable()
    with pytest.raises(ToolUsageError, match="mapping"):
        await advanced(flows, "orders", "a1", [1])
    with pytest.raises(ToolUsageError, match="identity"):
        await advanced(flows, "orders", "a1", {}, "ana")
    assert RAN == {}


async def test_a_reviewer_who_decides_during_the_timeout_wins():
    from agent_adapter.core.errors import ApprovalAlreadyDecided

    flows, _ = durable()
    paused = await flows.advance("gated", "t1", {})
    approval = paused.pending.approval_id
    store = flows.runner.executor.approval.store
    real_decide = store.decide

    async def reviewer_first(approval_id, decision):
        await real_decide(approval_id, decision.model_copy(update={"approved": True}))
        raise ApprovalAlreadyDecided(approval_id, status="approved")

    store.decide = reviewer_first
    report = await flows.resume("gated", "t1", approval, timed_out=True)
    assert report.status == "running"  # the reviewer's approval, not the timeout denial


async def test_gate_approvals_use_the_runners_own_store():
    from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
    from agent_adapter.core.types import ApprovalDecision
    from agent_adapter.exec import Executor
    from agent_adapter.exec.checkpoints import MemoryCheckpointStore
    from agent_adapter.flow import FlowRunner
    from agent_adapter.integrations._durable_flows import DurableFlows
    from tests.integrations.durable.kit import KEY, gated, registry

    gates = MemoryApprovalStore()
    runner = FlowRunner(
        Executor(registry),
        approval=DeferredApproval(gates),
        checkpoints=MemoryCheckpointStore(),
        checkpoint_key=KEY,
    )
    flows = DurableFlows(runner, [gated()])
    paused = await flows.advance("gated", "r1", {})
    await gates.decide(paused.pending.approval_id, ApprovalDecision(approved=True))
    assert (await flows.resume("gated", "r1", paused.pending.approval_id)).status == "running"
