import asyncio
import logging

import pytest

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, ToolUsageError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime import run_sync, use_context
from tests.approval.support import ids_from_threads
from tests.fakes.sinks import RecordingSink


def req(approval_id: str = "apr_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id, run_id="run_1", call_id="call_1", tool="delete", hint="rm -rf"
    )


async def test_full_lifecycle():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    assert isinstance(port, ApprovalPort)

    with pytest.raises(ApprovalPending) as first:  # create
        await port.request(req())
    pending = first.value
    assert (pending.approval_id, pending.tool, pending.run_id, pending.call_id) == (
        "apr_1",
        "delete",
        "run_1",
        "call_1",
    )
    assert (await store.get("apr_1")).status == "pending"

    with pytest.raises(ApprovalPending) as again:  # still pending: same id, no new record
        await port.request(req("apr_other"))
    assert again.value.approval_id == "apr_1"
    assert len(await store.list()) == 1

    decision = ApprovalDecision(approved=True, edited_args={"path": "/safe"}, decided_by="ana")
    await store.decide("apr_1", decision)  # decide out of band

    resumed = await port.request(req())  # consume
    assert resumed.approved and resumed.edited_args == {"path": "/safe"}
    assert (await store.get("apr_1")).consumed

    with pytest.raises(ApprovalPending) as second:  # a new request: a new record
        await port.request(req())
    assert second.value.approval_id not in ("", "apr_1")
    assert second.value.approval_id.startswith("apr_")
    assert len(await store.list()) == 2


async def test_denied_decision_is_returned_and_consumed():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    with pytest.raises(ApprovalPending):
        await port.request(req())
    await store.decide("apr_1", ApprovalDecision(approved=False, reason="no"))
    decision = await port.request(req())
    assert not decision.approved and decision.reason == "no"
    assert (await store.get("apr_1")).consumed


async def test_empty_approval_id_gets_one_from_the_store():
    store = MemoryApprovalStore()
    with pytest.raises(ApprovalPending) as info:
        await DeferredApproval(store).request(req(""))
    assert info.value.approval_id.startswith("apr_")
    assert (await store.get(info.value.approval_id)).request.tool == "delete"


async def test_emits_no_events():
    sink = RecordingSink()
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    with use_context(ToolContext(run_id="run_1", call_id="call_1", sink=sink)):
        with pytest.raises(ApprovalPending):
            await port.request(req())
        await store.decide("apr_1", ApprovalDecision(approved=True))
        await port.request(req())
    assert sink.events == []  # the executor is the single emitter of approval.*


class YieldingStore(MemoryApprovalStore):
    """Yields to the loop before each call, like a database round trip would."""

    async def find(self, run_id, call_id):
        await asyncio.sleep(0)
        return await super().find(run_id, call_id)

    async def consume(self, approval_id):
        await asyncio.sleep(0)
        return await super().consume(approval_id)


async def test_two_concurrent_resumes_get_one_decision_and_one_denial(caplog):
    store = YieldingStore()
    port = DeferredApproval(store)
    with pytest.raises(ApprovalPending):
        await port.request(req())
    await store.decide("apr_1", ApprovalDecision(approved=True))
    with caplog.at_level(logging.WARNING):
        outcomes = await asyncio.gather(
            port.request(req()), port.request(req()), return_exceptions=True
        )
    decisions = [o for o in outcomes if isinstance(o, ApprovalDecision)]
    denials = [o for o in outcomes if isinstance(o, ApprovalDenied)]
    assert len(decisions) == 1 and decisions[0].approved and len(denials) == 1
    assert denials[0].reason == "approval already used by a concurrent request"
    assert (denials[0].tool, denials[0].call_id) == ("delete", "call_1")
    assert "was already consumed: denying" in caplog.text


class StaleFindStore(MemoryApprovalStore):
    """`find` misses (a stale read); `create` must still return the existing record."""

    async def find(self, run_id, call_id):
        return None


async def test_create_returning_a_decided_record_resumes_with_its_decision():
    store = StaleFindStore()
    port = DeferredApproval(store)
    with pytest.raises(ApprovalPending):
        await port.request(req())
    with pytest.raises(ApprovalPending) as again:  # create hands back the open pending record
        await port.request(req())
    assert again.value.approval_id == "apr_1" and len(await store.list()) == 1
    await store.decide("apr_1", ApprovalDecision(approved=False, reason="no"))
    decision = await port.request(req())
    assert not decision.approved and (await store.get("apr_1")).consumed


def test_concurrent_first_requests_across_two_run_sync_loops_share_one_record():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)

    async def burst() -> list[str]:
        async def ask() -> str:
            try:
                await port.request(req(""))
            except ApprovalPending as pending:
                return pending.approval_id
            raise AssertionError("expected a pending approval")

        return list(await asyncio.gather(*(ask() for _ in range(100))))

    ids = ids_from_threads(burst)
    assert len(ids) == 200 and len(set(ids)) == 1
    assert len(run_sync(store.list())) == 1


def call_req(arguments: dict, *, tool: str = "transfer") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="run_1", call_id="c1", tool=tool, arguments=arguments
    )


X = {"to": "alice", "amount": 10}
Y = {"to": "mallory", "amount": 10_000}


async def approved_for_x(edited_args: dict | None = None) -> tuple[MemoryApprovalStore, str]:
    store = MemoryApprovalStore()
    with pytest.raises(ApprovalPending) as info:
        await DeferredApproval(store).request(call_req(X))
    decision = ApprovalDecision(approved=True, edited_args=edited_args, decided_by="ana")
    await store.decide(info.value.approval_id, decision)
    return store, info.value.approval_id


async def test_reused_call_id_with_other_arguments_fails_closed(caplog):
    store, approval_id = await approved_for_x()
    port = DeferredApproval(store)
    with caplog.at_level(logging.WARNING), pytest.raises(ToolUsageError, match="reused"):
        await port.request(call_req(Y))
    assert not (await store.get(approval_id)).consumed  # the approval for X is untouched
    assert "'transfer'" in caplog.text and "c1" in caplog.text
    assert "mallory" not in caplog.text and "10000" not in caplog.text
    decision = await port.request(call_req(dict(reversed(list(X.items())))))  # same args
    assert decision.approved


async def test_reused_call_id_with_another_tool_fails_closed():
    store, _ = await approved_for_x()
    with pytest.raises(ToolUsageError, match="reused"):
        await DeferredApproval(store).request(call_req(X, tool="delete_account"))


async def test_reused_call_id_while_pending_fails_closed():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    with pytest.raises(ApprovalPending):
        await port.request(call_req(X))
    with pytest.raises(ToolUsageError, match="reused"):
        await port.request(call_req(Y))
    assert len(await store.list()) == 1


async def test_racer_record_with_other_arguments_fails_closed():
    store = StaleFindStore()  # find misses, so create hands back the open record
    port = DeferredApproval(store)
    with pytest.raises(ApprovalPending):
        await port.request(call_req(X))
    with pytest.raises(ToolUsageError, match="reused"):
        await port.request(call_req(Y))


async def test_edited_args_still_apply_and_match_on_the_original_arguments():
    store, approval_id = await approved_for_x(edited_args={"to": "alice", "amount": 5})
    port = DeferredApproval(store)
    with pytest.raises(ToolUsageError):
        await port.request(call_req({"to": "alice", "amount": 5}))  # the edit is not a request
    decision = await port.request(call_req(X))
    assert decision.edited_args == {"to": "alice", "amount": 5}
    assert (await store.get(approval_id)).consumed
