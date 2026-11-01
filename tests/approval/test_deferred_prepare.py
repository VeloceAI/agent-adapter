"""DeferredApproval.prepare: find-or-create plus the digest check, never consuming."""

import pytest

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest


def req(arguments: dict | None = None, approval_id: str = "apr_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id="run_1",
        call_id="call_1",
        tool="delete",
        arguments=arguments or {"path": "/a"},
    )


async def test_prepare_creates_a_pending_record_once():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    first = await port.prepare(req())
    again = await port.prepare(req(approval_id="apr_other"))
    assert first.status == "pending" and again.approval_id == first.approval_id == "apr_1"
    assert len(await store.list()) == 1


async def test_prepare_never_consumes_a_decided_record():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    await port.prepare(req())
    await store.decide("apr_1", ApprovalDecision(approved=True))
    for _ in range(2):
        record = await port.prepare(req())
        assert record.status == "approved" and not record.consumed
    assert (await store.find("run_1", "call_1")) is not None
    assert (await port.request(req())).approved  # request still consumes exactly once
    assert (await store.find("run_1", "call_1")) is None


async def test_prepare_refuses_other_arguments_for_the_same_call():
    port = DeferredApproval(MemoryApprovalStore())
    await port.prepare(req())
    with pytest.raises(ToolUsageError, match="call id reused"):
        await port.prepare(req({"path": "/b"}))


async def test_prepare_fails_closed_on_non_json_arguments_before_the_store():
    store = MemoryApprovalStore()
    with pytest.raises(ToolUsageError):
        await DeferredApproval(store).prepare(req({"path": b"/a"}))
    assert await store.list() == []
