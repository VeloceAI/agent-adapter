"""Contract: flows on real DBOS with a SQLite system database (no Postgres)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("dbos")

from dbos import DBOS

from agent_adapter.integrations.dbos import DbosFlows
from tests.integrations.durable.kit import RAN, decide, durable

pytestmark = pytest.mark.contract


@pytest.fixture
def launched(tmp_path):
    flows, log = durable()
    store = flows.store
    original = store.save
    blips = {"left": 1}

    async def blips_once(run_id, step, checkpoint):
        if run_id == "blip-1" and checkpoint.status == "done" and blips["left"]:
            blips["left"] -= 1
            raise OSError("the store went away")  # after ship ran: a retry must replay it
        await original(run_id, step, checkpoint)

    store.save = blips_once  # type: ignore[method-assign]
    DBOS.destroy(destroy_registry=True)
    url = f"sqlite:///{(tmp_path / 'dbos.sqlite').as_posix()}"
    DBOS(config={"name": "aa_durable_test", "system_database_url": url})
    dbos_flows = DbosFlows(flows, max_attempts=3, interval_s=0.05)
    DBOS.launch()
    try:
        yield dbos_flows, log
    finally:
        DBOS.destroy(destroy_registry=True)


async def _waiting(dbos_flows, run_id):
    for _ in range(200):
        found = await dbos_flows.pending(run_id)
        if found is not None:
            return found
        await asyncio.sleep(0.05)
    raise AssertionError("the run never paused")


async def test_flows_run_pause_decide_and_recover_on_dbos(launched):
    dbos_flows, log = launched
    done = await dbos_flows.start("orders", {"order": "o-4"}, run_id="orders-1")
    report = await dbos_flows.result(done)
    assert report.status == "done" and report.state["note"] == "charged o-4"

    blip = await dbos_flows.start("orders", {}, run_id="blip-1")
    assert (await dbos_flows.result(blip)).status == "done"
    assert RAN["ship"] == 2  # once per run: the retried step replayed the logged call

    gated = await dbos_flows.start("gated", {}, run_id="gated-1")
    waiting = await _waiting(dbos_flows, "gated-1")
    assert waiting.node == "check"
    await dbos_flows.notify("gated-1", "apr_wrong")
    await dbos_flows.notify("gated-1", waiting.approval_id)  # forged: nothing in the store
    await asyncio.sleep(0.5)
    assert (await _waiting(dbos_flows, "gated-1")).approval_id == waiting.approval_id
    await decide(dbos_flows.flows, waiting.approval_id, True, decided_by="ana")
    await dbos_flows.notify("gated-1", waiting.approval_id)
    assert (await dbos_flows.result(gated)).status == "done"

    timed = await dbos_flows.start("gated", {}, run_id="gated-2", approval_timeout_s=0.5)
    report = await dbos_flows.result(timed)
    assert report.status == "denied" and report.denied_at == "check"

    broken = await dbos_flows.start("broken", {}, run_id="b-1")
    with pytest.raises(Exception) as failed:
        await dbos_flows.result(broken)
    assert "boom" not in str(failed.value)
    assert len(log.of_type("run.recovered")) == 1
