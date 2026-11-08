"""The Temporal adapter on a fake temporalio: workflow loop, store decisions, errors, helpers."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from agent_adapter.core.errors import MissingDependency, ToolUsageError
from agent_adapter.integrations import temporal
from tests.integrations.durable.fakes import WORKFLOW_ID, install_temporal
from tests.integrations.durable.kit import RAN, decide, durable


@pytest.fixture
def fake(monkeypatch):
    temporal.flow_workflow.cache_clear()
    seen = install_temporal(monkeypatch)
    flows, log = durable()
    temporal.flow_activities(flows)
    seen.flows, seen.log = flows, log
    yield seen
    temporal.flow_workflow.cache_clear()


def _start(run_id: str, request: dict):
    workflow = temporal.flow_workflow()()
    token = WORKFLOW_ID.set(run_id)
    try:
        task = asyncio.ensure_future(workflow.run(request))
    finally:
        WORKFLOW_ID.reset(token)
    return workflow, task


async def _paused(workflow):
    for _ in range(500):
        if workflow.pending() is not None:
            return workflow.pending()
        await asyncio.sleep(0)
    raise AssertionError("never paused")


async def test_a_flow_runs_one_activity_per_superstep(fake):
    _, task = _start("r1", {"flow": "orders", "state": {"order": "o-1"}, "max_attempts": 3})
    report = await task
    assert report["status"] == "done" and report["state"]["sent"] == "shipped o-1"
    names = [name for name, _ in fake.calls]
    assert names == [temporal.ADVANCE, temporal.ADVANCE]
    options = fake.calls[0][1]
    assert options["retry_policy"].kwargs == {"maximum_attempts": 3}
    assert options["start_to_close_timeout"].total_seconds() == 600
    assert fake.defn == {"name": temporal.WORKFLOW, "sandboxed": False}


async def test_a_notice_without_a_store_decision_keeps_the_run_waiting(fake):
    workflow, task = _start("g1", {"flow": "gated", "state": {}, "activity_timeout_s": 5})
    with pytest.raises(ValueError, match="not waiting"):
        workflow.check({"approval_id": "x"})
    pending = await _paused(workflow)
    assert pending["kind"] == "node" and "arguments" not in pending
    for bad, why in (
        ({"approval_id": "other"}, "another approval"),
        ({"approval_id": pending["approval_id"], "approved": True}, "not a valid notice"),
        ("approve", "not a valid notice"),
    ):
        with pytest.raises(ValueError, match=why):
            workflow.check(bad)
    notice = {"approval_id": pending["approval_id"]}
    workflow.check(notice)
    assert await workflow.notify(notice) == pending["approval_id"]
    with pytest.raises(ValueError, match="already received"):
        workflow.check(notice)
    await asyncio.sleep(0.01)  # the resume ran and found no decision in the store
    forged = await _paused(workflow)  # so it waits again
    assert forged == pending and not task.done() and RAN["ship"] == 0
    await decide(fake.flows, pending["approval_id"], True, decided_by="ana")
    workflow.notify_signal(notice)
    report = await task
    assert report["status"] == "done" and workflow.pending() is None and RAN["ship"] == 1
    names = [name for name, _ in fake.calls]
    assert names == [temporal.ADVANCE, temporal.RESUME, temporal.RESUME, temporal.ADVANCE]


async def test_a_signal_with_the_wrong_approval_is_dropped(fake):
    workflow, task = _start("g2", {"flow": "gated", "state": {}})
    pending = await _paused(workflow)
    await decide(fake.flows, pending["approval_id"], False)
    workflow.notify_signal({"approval_id": "other"})
    assert workflow.pending() == pending and not task.done()
    workflow.notify_signal({"approval_id": pending["approval_id"]})
    report = await task
    assert report["status"] == "denied" and RAN["ship"] == 0


async def test_an_approval_timeout_denies_in_the_store(fake):
    _, task = _start("g3", {"flow": "gated", "state": {}, "approval_timeout_s": 0.01})
    report = await task
    assert report["status"] == "denied" and report["denied_at"] == "check"
    [record] = await fake.flows.runner.executor.approval.store.list()
    assert record.decision.decided_by == "timeout" and record.consumed


async def test_a_failed_run_and_bad_input_fail_the_workflow_for_good(fake):
    _, task = _start("b1", {"flow": "broken", "state": {}})
    with pytest.raises(fake.ApplicationError) as failed:
        await task
    assert failed.value.type == "flow-error" and failed.value.non_retryable
    assert "boom" not in str(failed.value)
    _, task = _start("b2", {"state": {}})
    with pytest.raises(fake.ApplicationError, match="bad workflow input"):
        await task


async def test_a_run_whose_store_says_failed_fails_the_workflow(fake):
    with pytest.raises(Exception):  # noqa: B017 - the first attempt marks the run failed
        await fake.flows.advance("broken", "b3", {})
    _, task = _start("b3", {"flow": "broken", "state": {}})
    with pytest.raises(fake.ApplicationError, match="the flow run failed"):
        await task


async def test_activity_errors_keep_only_type_and_kind(fake):
    advance = fake.activities[temporal.ADVANCE]
    resume = fake.activities[temporal.RESUME]
    token = WORKFLOW_ID.set("e1")
    try:
        with pytest.raises(fake.ApplicationError) as missing:
            await advance({"flow": "orders"})  # a new run without state
        assert missing.value.type == "tool-usage-error" and missing.value.non_retryable
        with pytest.raises(fake.ApplicationError) as bad:
            await resume({"flow": "orders", "decision": {"approved": "no"}})
        assert bad.value.non_retryable

        async def blip(run_id):
            raise OSError("secret path")

        fake.flows.store.latest = blip  # type: ignore[method-assign]
        with pytest.raises(fake.ApplicationError) as passing:
            await advance({"flow": "orders", "state": {}})
        assert passing.value.type == "OSError" and not passing.value.non_retryable
        assert "secret" not in str(passing.value)
    finally:
        WORKFLOW_ID.reset(token)


def test_flow_worker_registers_the_workflow_and_activities(fake):
    worker = temporal.flow_worker(
        "client", fake.flows, task_queue="tickets", max_concurrent_activities=2
    )
    kwargs = worker.kwargs
    assert kwargs["task_queue"] == "tickets" and kwargs["max_concurrent_activities"] == 2
    assert kwargs["workflows"] == [temporal.flow_workflow()] and len(kwargs["activities"]) == 2
    with pytest.raises(ToolUsageError, match="task_queue"):
        temporal.flow_worker("client", fake.flows, task_queue="")
    with pytest.raises(ToolUsageError, match="DurableFlows"):
        temporal.flow_activities("flows")  # type: ignore[arg-type]


class _Client:
    def __init__(self) -> None:
        self.started: list[tuple] = []

    async def start_workflow(self, name, request, *, id, task_queue):
        self.started.append((name, request, id, task_queue))
        return "handle"


async def test_start_flow_checks_and_sends_the_request(fake):
    client = _Client()
    handle = await temporal.start_flow(
        client, "orders", {"order": "o-2"}, run_id="r9", task_queue="q", approval_timeout_s=30
    )
    assert handle == "handle"
    name, request, run_id, queue = client.started[0]
    assert (name, run_id, queue) == (temporal.WORKFLOW, "r9", "q")
    assert request == {
        "flow": "orders",
        "state": {"order": "o-2"},
        "approval_timeout_s": 30,
        "activity_timeout_s": 600.0,
        "max_attempts": 0,
        "identity": None,
    }
    bad = [
        ({"flow": ""}, "flow"),
        ({"run_id": ""}, "run_id"),
        ({"task_queue": ""}, "task_queue"),
        ({"state": [1]}, "mapping"),
        ({"approval_timeout_s": 0}, "approval_timeout_s"),
        ({"activity_timeout_s": True}, "activity_timeout_s"),
        ({"max_attempts": -1}, "max_attempts"),
    ]
    for change, match in bad:
        args = {"flow": "orders", "state": {}, "run_id": "r", "task_queue": "q", **change}
        with pytest.raises(ToolUsageError, match=match):
            await temporal.start_flow(
                client,
                args.pop("flow"),
                args.pop("state"),
                **args,
            )


class _Handle:
    def __init__(self, pending=None, result=None) -> None:
        self.sent: list[tuple] = []
        self._pending, self._result = pending, result

    async def query(self, name):
        return self._pending

    async def execute_update(self, name, payload):
        self.sent.append(("update", name, payload))

    async def signal(self, name, payload):
        self.sent.append(("signal", name, payload))

    async def result(self):
        return self._result


async def test_client_helpers(fake):
    waiting = {"approval_id": "a1", "kind": "node", "tool": "node:gated.check", "node": "check"}
    handle = _Handle(pending=waiting, result={"run_id": "r", "flow": "orders", "status": "done"})
    assert (await temporal.pending(handle)).approval_id == "a1"
    assert await temporal.pending(_Handle()) is None
    await temporal.notify(handle, "a1")
    await temporal.notify(handle, "a1", wait=False)
    assert handle.sent == [
        ("update", "notify", {"approval_id": "a1"}),
        ("signal", "notify", {"approval_id": "a1"}),
    ]
    with pytest.raises(ToolUsageError, match="approval_id"):
        await temporal.notify(handle, "")
    assert (await temporal.result(handle)).status == "done"


def test_missing_temporalio_names_the_extra(monkeypatch):
    temporal.flow_workflow.cache_clear()
    monkeypatch.setitem(sys.modules, "temporalio", None)
    flows, _ = durable()
    with pytest.raises(MissingDependency, match=r"agent-adapter\[temporal\]"):
        temporal.flow_activities(flows)
    with pytest.raises(MissingDependency):
        temporal.flow_workflow()
    temporal.flow_workflow.cache_clear()


def test_the_module_imports_without_temporalio():
    assert isinstance(temporal, types.ModuleType) and temporal.WORKFLOW == "agent_adapter.flow"


async def test_an_early_signal_is_kept_for_the_next_pause(fake):
    workflow = temporal.flow_workflow()()
    workflow.notify_signal({"approval_id": "a1"})  # before the workflow saw the pause
    assert await workflow._wait({"approval_id": "a1"}, None) is False
    workflow.notify_signal({"approval_id": "old"})
    assert await workflow._wait({"approval_id": "a2"}, 0.01) is True  # stale: dropped, timed out


async def test_identity_travels_to_every_activity_and_in_doubt_stops_for_good(fake):
    who = {"user_id": "ana", "thread_id": "", "principal": "desk"}
    _, task = _start("i1", {"flow": "orders", "state": {}, "identity": who})
    assert (await task)["status"] == "done"
    with pytest.raises(fake.ApplicationError) as other:
        await _start("i1", {"flow": "orders", "state": {}})[1]
    assert other.value.type == "tool-usage-error" and other.value.non_retryable
    from tests.integrations.durable.kit import CRASH_AT, Crash

    CRASH_AT["pay"] = 1
    with pytest.raises(Crash):
        await _start("p1", {"flow": "payments", "state": {}})[1]
    with pytest.raises(fake.ApplicationError) as doubt:
        await _start("p1", {"flow": "payments", "state": {}})[1]
    assert doubt.value.type == "recovery-needs-decision" and doubt.value.non_retryable


async def test_start_flow_sends_the_identity_and_checks_it(fake):
    client = _Client()
    who = temporal.RunIdentity(user_id="ana")
    await temporal.start_flow(client, "orders", {}, run_id="r", task_queue="q", identity=who)
    assert client.started[0][1]["identity"] == {"user_id": "ana", "thread_id": "", "principal": ""}
    with pytest.raises(ToolUsageError, match="RunIdentity"):
        await temporal.start_flow(client, "orders", {}, run_id="r", task_queue="q", identity={})
