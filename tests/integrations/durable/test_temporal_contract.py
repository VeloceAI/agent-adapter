"""Contract: flows on a real local Temporal server (dev server, SQLite, no Docker)."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("temporalio")

from temporalio.client import WorkflowFailureError, WorkflowUpdateFailedError
from temporalio.testing import WorkflowEnvironment

from agent_adapter.integrations import temporal
from tests.integrations.durable.kit import RAN, decide, durable

pytestmark = pytest.mark.contract


@pytest.fixture
async def env():
    folder = os.environ.get("AA_TEMPORAL_DOWNLOADS")  # where the dev server binary is cached
    async with await WorkflowEnvironment.start_local(download_dest_dir=folder) as started:
        yield started


async def _waiting(handle):
    for _ in range(200):
        found = await temporal.pending(handle)
        if found is not None:
            return found
        await asyncio.sleep(0.05)
    raise AssertionError("the run never paused")


async def test_flows_run_pause_decide_and_fail_closed_on_temporal(env):
    flows, log = durable()
    queue = f"q-{uuid.uuid4().hex}"
    async with temporal.flow_worker(env.client, flows, task_queue=queue):
        done = await temporal.start_flow(
            env.client, "orders", {"order": "o-3"}, run_id="orders-1", task_queue=queue
        )
        report = await temporal.result(done)
        assert report.status == "done" and report.state["sent"] == "shipped o-3"
        assert dict(RAN) == {"charge": 1, "ship": 1}

        gated = await temporal.start_flow(
            env.client, "gated", {}, run_id="gated-1", task_queue=queue
        )
        waiting = await _waiting(gated)
        assert waiting.kind == "node" and waiting.node == "check"
        with pytest.raises(WorkflowUpdateFailedError):
            await temporal.notify(gated, "apr_wrong")
        await temporal.notify(gated, waiting.approval_id)  # forged: nothing in the store
        again = await _waiting(gated)
        assert again.approval_id == waiting.approval_id and RAN["ship"] == 1
        await decide(flows, waiting.approval_id, True, decided_by="ana")
        await temporal.notify(gated, waiting.approval_id)
        assert (await temporal.result(gated)).status == "done" and RAN["ship"] == 2

        signalled = await temporal.start_flow(
            env.client, "refunds", {"order": "o-8"}, run_id="refund-1", task_queue=queue
        )
        waiting = await _waiting(signalled)
        assert waiting.tool == "refund"
        await decide(flows, waiting.approval_id, False, reason="no")
        await temporal.notify(signalled, waiting.approval_id, wait=False)
        with pytest.raises(WorkflowFailureError):  # a denied tool fails its node
            await temporal.result(signalled)
        assert RAN["refund"] == 0

        timed = await temporal.start_flow(
            env.client, "gated", {}, run_id="gated-2", task_queue=queue, approval_timeout_s=0.5
        )
        report = await temporal.result(timed)
        assert report.status == "denied" and report.denied_at == "check"

        broken = await temporal.start_flow(env.client, "broken", {}, run_id="b-1", task_queue=queue)
        with pytest.raises(WorkflowFailureError) as failed:
            await temporal.result(broken)
        assert "boom" not in str(failed.value.cause)  # history keeps the kind, not the text
    assert len(log.of_type("flow.started")) == 5


async def test_a_store_blip_after_a_call_retries_the_activity_and_replays_it(env):
    flows, log = durable()
    original = flows.store.save
    blips = {"left": 1}

    async def blips_once(run_id, step, checkpoint):
        if checkpoint.status == "done" and blips["left"]:
            blips["left"] -= 1
            raise OSError("the store went away")  # after ship ran
        await original(run_id, step, checkpoint)

    flows.store.save = blips_once  # type: ignore[method-assign]
    queue = f"q-{uuid.uuid4().hex}"
    async with temporal.flow_worker(env.client, flows, task_queue=queue):
        handle = await temporal.start_flow(
            env.client, "orders", {}, run_id="blip", task_queue=queue
        )
        assert (await temporal.result(handle)).status == "done"
    assert RAN["ship"] == 1 and len(log.of_type("run.recovered")) == 1
