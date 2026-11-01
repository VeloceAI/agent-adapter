"""DeferredApproval.consume trusts only the store, never the record or request it is given."""

import pytest

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

YES = ApprovalDecision(approved=True)
NO = ApprovalDecision(approved=False, reason="no")


def req(path: str = "/f") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="apr_1", run_id="run_1", call_id="call_1", tool="wipe", arguments={"path": path}
    )


async def prepared(decision: ApprovalDecision | None = None):  # type: ignore[no-untyped-def]
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    record = await port.prepare(req())
    if decision is not None:
        record = await store.decide(record.approval_id, decision)
    return store, port, record


async def test_c5_a_request_for_other_arguments_consumes_nothing():
    store, port, record = await prepared(YES)
    with pytest.raises(ToolUsageError, match="call id reused"):
        await port.consume(record, req("/evil"))
    assert not (await store.get(record.approval_id)).consumed


async def test_c6_a_forged_approved_copy_of_a_pending_record_is_still_pending():
    store, port, record = await prepared()
    forged = record.model_copy(update={"status": "approved", "decision": YES})
    with pytest.raises(ApprovalPending):
        await port.consume(forged, req())
    current = await store.get(record.approval_id)
    assert current.status == "pending" and not current.consumed


async def test_c7_a_denied_record_flipped_to_approved_is_denied(caplog):
    store, port, record = await prepared(NO)
    forged = record.model_copy(update={"status": "approved", "decision": YES})
    with pytest.raises(ApprovalDenied, match="decision changed"):
        await port.consume(forged, req())
    assert not (await store.get(record.approval_id)).consumed
    assert "decision changed" in caplog.text


async def test_consume_returns_the_stored_decision_once():
    _store, port, record = await prepared(YES)
    assert await port.consume(record, req()) == record.decision
    with pytest.raises(ApprovalDenied, match="concurrent"):
        await port.consume(record, req())


async def test_the_store_refuses_to_consume_an_undecided_record():
    store = MemoryApprovalStore()
    record = await store.create(req())
    assert await store.consume(record.approval_id) is False
    current = await store.get(record.approval_id)
    assert current.status == "pending" and not current.consumed
    assert await store.find("run_1", "call_1") == current


def test_check_json_is_the_digest_rule_for_any_value():
    from agent_adapter.approval.store import check_json

    check_json({"a": [1, 2.5, None, True, "x"]})
    for bad in ({1: "x"}, {"a": (1,)}, {"a": b"x"}, {"a": float("inf")}):
        with pytest.raises(ToolUsageError, match="not JSON"):
            check_json(bad)


async def test_c8_another_runs_decision_is_not_spent_on_this_run(caplog):
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    theirs = req().model_copy(update={"approval_id": "apr_2", "run_id": "run_2"})
    record = await store.decide((await port.prepare(theirs)).approval_id, YES)
    with pytest.raises(ToolUsageError, match="call id reused"):
        await port.consume(record, req())  # run_1, same tool and arguments
    assert not (await store.get(record.approval_id)).consumed
    assert "run_1" in caplog.text and "/f" not in caplog.text  # ids only, never arguments
    assert await port.consume(record, theirs) == record.decision


async def test_another_calls_decision_is_not_spent_on_this_call():
    store, port, record = await prepared(YES)
    other_call = req().model_copy(update={"call_id": "call_2"})
    with pytest.raises(ToolUsageError, match="call id reused"):
        await port.consume(record, other_call)
    assert not (await store.get(record.approval_id)).consumed
