"""The DBOS adapter on a fake dbos: workflow loop, messages, retries, errors, helpers."""

from __future__ import annotations

import asyncio
import sys

import pytest

from agent_adapter.core.errors import FlowError, MissingDependency, ToolUsageError
from agent_adapter.integrations import dbos as adapter
from agent_adapter.integrations.dbos import DbosFlows, StepRetry
from tests.integrations.durable.fakes import WORKFLOW_ID, install_dbos
from tests.integrations.durable.kit import RAN, decide, durable


@pytest.fixture
def fake(monkeypatch):
    dbos = install_dbos(monkeypatch)
    flows, log = durable()
    dbos.flows, dbos.log = flows, log
    dbos.adapter = DbosFlows(flows, max_attempts=4, interval_s=0.5)
    return dbos


async def _paused(fake, run_id):
    for _ in range(500):
        found = await fake.adapter.pending(run_id)
        if found is not None:
            return found
        await asyncio.sleep(0)
    raise AssertionError("never paused")


async def test_registration_names_and_retry_options(fake):
    assert set(fake.workflows) == {adapter.WORKFLOW}
    assert set(fake.steps) == {adapter.ADVANCE, adapter.RESUME}
    options = fake.steps[adapter.ADVANCE]
    assert options["max_attempts"] == 4 and options["interval_seconds"] == 0.5
    should_retry = options["should_retry"]
    assert should_retry(StepRetry("x")) and not should_retry(FlowError("x"))


async def test_a_flow_runs_to_done(fake):
    handle = await fake.adapter.start("orders", {"order": "o-1"}, run_id="r1")
    report = await fake.adapter.result(handle)
    assert report.status == "done" and report.state["sent"] == "shipped o-1"


async def test_a_pause_waits_for_the_matching_message(fake):
    handle = await fake.adapter.start("gated", {}, run_id="g1")
    pending = await _paused(fake, "g1")
    assert pending.kind == "node"
    await fake.adapter.notify("g1", "other")
    await fake.adapter.notify("g1", pending.approval_id)  # forged: nothing in the store
    await asyncio.sleep(0.01)
    assert not handle.task.done() and RAN["ship"] == 0  # neither notice approved anything
    await decide(fake.flows, pending.approval_id, True, decided_by="ana")
    await fake.adapter.notify("g1", pending.approval_id)
    report = await fake.adapter.result(handle)
    assert report.status == "done" and await fake.adapter.pending("g1") is None


async def test_without_a_timeout_the_wait_goes_on(fake, monkeypatch):
    monkeypatch.setattr(adapter, "_WAIT_S", 0.01)
    handle = await fake.adapter.start("gated", {}, run_id="g4")
    pending = await _paused(fake, "g4")
    await asyncio.sleep(0.05)  # several empty waits
    assert not handle.task.done()
    await decide(fake.flows, pending.approval_id, False)
    await fake.adapter.notify("g4", pending.approval_id)
    assert (await fake.adapter.result(handle)).status == "denied"


async def test_a_store_decision_cannot_be_used_twice(fake):
    handle = await fake.adapter.start("gated", {}, run_id="g5")
    pending = await _paused(fake, "g5")
    await decide(fake.flows, pending.approval_id, True)
    store = fake.flows.runner.executor.approval.store
    assert await store.consume(pending.approval_id)  # used elsewhere first
    await fake.adapter.notify("g5", pending.approval_id)
    with pytest.raises(Exception, match="ApprovalAlreadyDecided"):
        await fake.adapter.result(handle)
    assert RAN["ship"] == 0


async def test_a_timeout_denies(fake):
    handle = await fake.adapter.start("gated", {}, run_id="g2", approval_timeout_s=0.01)
    report = await fake.adapter.result(handle)
    assert report.status == "denied" and RAN["ship"] == 0


async def test_failures_keep_only_type_and_kind(fake):
    handle = await fake.adapter.start("broken", {}, run_id="b1")
    with pytest.raises(FlowError, match=r"flow step failed: FlowError \(flow-error\)") as failed:
        await fake.adapter.result(handle)
    assert "boom" not in str(failed.value)
    again = await fake.adapter.start("broken", {}, run_id="b1")  # the store says failed
    with pytest.raises(FlowError, match="the flow run failed"):
        await fake.adapter.result(again)


async def test_steps_turn_passing_errors_into_retries(fake):
    async def blip(run_id):
        raise OSError("secret path")

    fake.flows.store.latest = blip  # type: ignore[method-assign]
    handle = await fake.adapter.start("orders", {}, run_id="e1")
    with pytest.raises(StepRetry) as retry:
        await fake.adapter.result(handle)
    assert str(retry.value) == "OSError"


async def test_a_malformed_decision_or_input_fails_for_good(fake):
    token = WORKFLOW_ID.set("w1")
    try:
        with pytest.raises(ToolUsageError, match="bad workflow input"):
            await fake.workflows[adapter.WORKFLOW]({"state": {}})
    finally:
        WORKFLOW_ID.reset(token)
    await fake.flows.advance("gated", "g3", {})
    fake.next_id = "g3"
    handle = await fake.adapter.start("gated", {}, run_id="g3")
    approval = (await _paused(fake, "g3")).approval_id
    await fake.send_async("g3", {"approval_id": approval, "approved": True})
    await fake.send_async("g3", "approve")
    await asyncio.sleep(0.01)
    assert not handle.task.done()  # neither message is a notice, and none is a decision
    handle.task.cancel()


async def test_start_and_decide_check_their_input(fake):
    bad = [
        (("", {}), {"run_id": "r"}, "flow"),
        (("orders", {}), {"run_id": ""}, "run_id"),
        (("orders", [1]), {"run_id": "r"}, "mapping"),
        (("orders", {}), {"run_id": "r", "approval_timeout_s": 0}, "approval_timeout_s"),
        (("missing", {}), {"run_id": "r"}, "no durable flow"),
    ]
    for args, kwargs, match in bad:
        with pytest.raises(ToolUsageError, match=match):
            await fake.adapter.start(*args, **kwargs)
    with pytest.raises(ToolUsageError, match="approval_id"):
        await fake.adapter.notify("r", "")
    with pytest.raises(ToolUsageError, match="run_id"):
        await fake.adapter.notify("", "a")


def test_construction_is_checked(fake):
    with pytest.raises(ToolUsageError, match="DurableFlows"):
        DbosFlows("flows")  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="max_attempts"):
        DbosFlows(fake.flows, max_attempts=0)
    with pytest.raises(ToolUsageError, match="interval_s"):
        DbosFlows(fake.flows, interval_s=True)


def test_missing_dbos_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "dbos", None)
    flows, _ = durable()
    with pytest.raises(MissingDependency, match=r"agent-adapter\[dbos\]"):
        DbosFlows(flows)


async def test_a_stale_decision_is_dropped_at_the_next_pause(fake):
    handle = await fake.adapter.start("twice", {}, run_id="w2")
    first = await _paused(fake, "w2")
    await decide(fake.flows, first.approval_id, True)
    await fake.adapter.notify("w2", first.approval_id)
    await fake.adapter.notify("w2", first.approval_id)  # sent twice: the copy waits in the queue
    second = None
    for _ in range(500):
        second = await fake.adapter.pending("w2")
        if second is not None and second.approval_id != first.approval_id:
            break
        await asyncio.sleep(0)
    assert second is not None and second.approval_id != first.approval_id
    await asyncio.sleep(0.01)
    assert not handle.task.done()  # the stale copy did not answer the second pause
    await fake.adapter.notify("w2", second.approval_id)
    await asyncio.sleep(0.01)
    assert not handle.task.done()  # a notice alone never approves
    await decide(fake.flows, second.approval_id, True)
    await fake.adapter.notify("w2", second.approval_id)
    assert (await fake.adapter.result(handle)).status == "done"


async def test_the_identity_reaches_every_step(fake):
    who = adapter.RunIdentity(user_id="ana", principal="desk")
    handle = await fake.adapter.start("orders", {}, run_id="i1", identity=who)
    assert (await fake.adapter.result(handle)).status == "done"
    with pytest.raises(ToolUsageError, match="RunIdentity"):
        await fake.adapter.start("orders", {}, run_id="i2", identity={"user_id": "x"})  # type: ignore[arg-type]
