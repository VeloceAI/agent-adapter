"""`approval.batch.ask_each` as the native-approval bridges (WP13a) use it."""

import logging

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.approval.batch import RESET_SINCE_VETTED, ask_each
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, ToolUsageError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations._native_gate import approval_request
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.integrations._two_phase import TwoPhaseApproval


def requests(*items: str):
    return [approval_request("remove", {"item": i}, run_id="r", call_id=i) for i in items]


async def test_plain_ports_are_asked_in_order_and_vetted():
    seen = []
    port = RecordingApproval(ScriptedApproval([ApprovalDecision(approved=True)] * 2))
    decisions = await ask_each(port, requests("a", "b"), vet=lambda r, d: seen.append(r.call_id))
    assert [d.approved for d in decisions] == [True, True] and seen == ["a", "b"]


async def test_plain_ports_raise_the_first_pending_after_asking_all():
    port = RecordingApproval(
        ScriptedApproval([ApprovalPending("apr_1"), ApprovalDecision(approved=True)])
    )
    with pytest.raises(ApprovalPending) as info:
        await ask_each(port, requests("a", "b"))
    assert info.value.approval_id == "apr_1" and len(port.requests) == 2


async def test_two_phase_ports_consume_only_when_all_are_decided():
    port = TwoPhaseApproval()
    with pytest.raises(ApprovalPending):
        await ask_each(port, requests("a", "b"))
    records = await port.store.list()
    assert len(records) == 2 and not any(r.consumed for r in records)
    await port.decide_all(ApprovalDecision(approved=True, reason="ok"))

    def refuse(request, decision):
        raise ToolUsageError("vetted out")

    with pytest.raises(ToolUsageError):
        await ask_each(port, requests("a", "b"), vet=refuse)
    assert not any(r.consumed for r in await port.store.list())
    decisions = await ask_each(port, requests("a", "b"))
    assert [d.reason for d in decisions] == ["ok", "ok"]
    assert all(r.consumed for r in await port.store.list())


async def test_a_lost_consume_race_is_a_denial():
    port = TwoPhaseApproval()
    (request,) = requests("a")
    record = await port.prepare(request)
    await port.store.decide(record.approval_id, ApprovalDecision(approved=True))
    assert await port.store.consume(record.approval_id)
    with pytest.raises(ApprovalDenied):
        await port.consume(record, request)


class Racing(TwoPhaseApproval):
    """Consumes like the store would, but a named call loses the race or changes its decision."""

    def __init__(self, *, lost: str = "", changed: str = "") -> None:
        super().__init__()
        self.lost, self.changed = lost, changed

    async def consume(self, record, request):
        if request.call_id == self.lost:
            raise ApprovalDenied(request.tool, call_id=request.call_id, reason="used elsewhere")
        decision = await super().consume(record, request)
        if request.call_id == self.changed:
            return decision.model_copy(update={"edited_args": {"item": "PROD"}})
        return decision


async def decided(port, *calls):
    with pytest.raises(ApprovalPending):
        await ask_each(port, requests(*calls))
    await port.decide_all(ApprovalDecision(approved=True, reason="ok"))


async def test_a_lost_consume_race_denies_only_that_call(caplog):
    port = Racing(lost="a")
    await decided(port, "a", "b")
    first, second = await ask_each(port, requests("a", "b"))
    assert not first.approved and first.reason == "used elsewhere"
    assert second.approved and "was not usable: denying" in caplog.text
    loud = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("used elsewhere" in message for message in loud)


async def test_a_decision_changed_after_vetting_is_denied(caplog):
    port = Racing(changed="b")
    await decided(port, "a", "b")
    first, second = await ask_each(port, requests("a", "b"))
    assert first.approved and not second.approved
    assert "differs from the vetted one" in caplog.text


async def test_a_deferred_store_without_prepare_converges_one_decision_at_a_time():
    """Deciding the calls of a batch one by one converges; nothing is consumed early."""
    port = DeferredApproval(MemoryApprovalStore())
    batch = requests("c1", "c2")
    with pytest.raises(ApprovalPending):
        await ask_each(port, batch)
    first = next(r for r in await port.store.list() if r.request.call_id == "c1")
    await port.store.decide(first.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ApprovalPending) as info:
        await ask_each(port, batch)
    assert info.value.call_id == "c2"
    assert not any(r.consumed for r in await port.store.list())  # c1 kept for later
    second = next(r for r in await port.store.list() if r.request.call_id == "c2")
    await port.store.decide(second.approval_id, ApprovalDecision(approved=False, reason="no"))
    decisions = await ask_each(port, batch)
    assert [d.approved for d in decisions] == [True, False]
    records = await port.store.list()
    assert len(records) == 2 and all(r.consumed for r in records)


async def test_a_deferred_store_refuses_other_arguments_for_a_call_id():
    port = DeferredApproval(MemoryApprovalStore())
    with pytest.raises(ApprovalPending):
        await ask_each(port, requests("c1"))
    other = [approval_request("remove", {"item": "PROD"}, run_id="r", call_id="c1")]
    with pytest.raises(ToolUsageError):
        await ask_each(port, other)


async def test_deferred_consume_refuses_pending_and_a_second_use():
    port = DeferredApproval(MemoryApprovalStore())
    (request,) = requests("c1")
    record = await port.prepare(request)
    with pytest.raises(ApprovalPending):
        await port.consume(record, request)
    await port.store.decide(record.approval_id, ApprovalDecision(approved=True))
    record = await port.prepare(request)
    assert (await port.consume(record, request)).approved
    with pytest.raises(ApprovalDenied):
        await port.consume(record, request)


class Resetting(TwoPhaseApproval):
    """A store whose record of the named call goes back to pending after it was vetted."""

    def __init__(self, reset: str) -> None:
        super().__init__()
        self.reset = reset

    async def consume(self, record, request):
        if request.call_id == self.reset:
            raise ApprovalPending(record.approval_id, tool=request.tool, call_id=request.call_id)
        return await super().consume(record, request)


async def test_a_record_pending_again_at_consume_denies_only_that_call(caplog):
    port = Resetting(reset="a")
    await decided(port, "a", "b")
    first, second = await ask_each(port, requests("a", "b"))
    assert not first.approved and first.reason == RESET_SINCE_VETTED
    assert first.decided_by == "agent_adapter"
    assert second.approved
    assert "pending again since it was vetted" in caplog.text
    consumed = {r.request.call_id: r.consumed for r in await port.store.list()}
    assert consumed == {"a": False, "b": True}
