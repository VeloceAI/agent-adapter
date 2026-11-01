"""Withdrawing the approvals of a pause that cannot be sealed, whatever their store does."""

import logging

import pytest

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import FlowError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.flow import _pause
from agent_adapter.flow._withdraw import withdraw
from tests.fakes.sinks import RecordingSink
from tests.flow.test_runner_fail_closed import pay_flow, runner

CTX = ToolContext(run_id="run_w")


def _request(approval_id="apr_w"):
    return ApprovalRequest(
        approval_id=approval_id, run_id="run_w", call_id="c", tool="pay", arguments={"a": 1}
    )


async def test_an_approval_decided_meanwhile_is_still_consumed():
    store = MemoryApprovalStore()
    record = await store.create(_request())
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    await withdraw("f", CTX, [(_request(), store)])
    assert (await store.get(record.approval_id)).consumed


async def test_a_record_the_store_does_not_hold_is_logged_as_an_error(caplog):
    sink = RecordingSink()
    with caplog.at_level(logging.ERROR):
        await withdraw(
            "f", ToolContext(run_id="run_w", sink=sink), [(_request(), MemoryApprovalStore())]
        )
    assert "apr_w cannot be withdrawn" in caplog.text
    assert sink.of_type("approval.decided")[0].data["withdrawn"] is False


async def test_a_failing_store_is_logged_and_the_rest_are_withdrawn(caplog):
    class Broken(MemoryApprovalStore):
        async def decide(self, approval_id, decision):
            raise ConnectionError("down")

    good = MemoryApprovalStore()
    await good.create(_request("apr_ok"))
    with caplog.at_level(logging.ERROR):
        await withdraw("f", CTX, [(_request(), Broken()), (_request("apr_ok"), good)])
    assert "could not be withdrawn from its store" in caplog.text
    assert (await good.get("apr_ok")).consumed


async def test_a_value_error_while_sealing_fails_closed_too(monkeypatch):
    def broken_seal(sealer, checkpoint):
        raise ValueError("no JSON for this")

    monkeypatch.setattr(_pause, "seal", broken_seal)
    run, store = runner()
    with pytest.raises(FlowError, match=r"state-not-sealable.*no JSON for this"):
        await run.run(pay_flow("ve"), {})
    [record] = await store.list()
    assert record.status == "denied" and record.consumed
