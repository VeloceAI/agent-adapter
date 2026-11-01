"""approval.batch: ask_each, two-phase for PreparedApproval ports, in order for the rest."""

import logging

import pytest

from agent_adapter.approval.batch import PreparedApproval, ask_each
from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from tests.fakes.approval import RecordingApproval, ScriptedApproval


def req(call_id: str, arguments: dict | None = None) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="",
        run_id="run_1",
        call_id=call_id,
        tool="wipe",
        arguments=arguments or {"path": "/a"},
    )


def test_deferred_approval_is_a_prepared_approval_and_plain_ports_are_not():
    assert isinstance(DeferredApproval(MemoryApprovalStore()), PreparedApproval)
    assert not isinstance(ScriptedApproval([]), PreparedApproval)
    assert not isinstance(RecordingApproval(ScriptedApproval([])), PreparedApproval)
    assert isinstance(RecordingApproval(DeferredApproval(MemoryApprovalStore())), PreparedApproval)


async def test_in_order_returns_decisions_and_raises_the_first_deferral_last(caplog):
    caplog.set_level(logging.INFO, logger="agent_adapter.approval.batch")
    port = ScriptedApproval([ApprovalDecision(approved=True), ApprovalDecision(approved=False)])
    seen: list[str] = []
    decisions = await ask_each(
        port, [req("c1"), req("c2")], vet=lambda r, d: seen.append(r.call_id)
    )
    assert [d.approved for d in decisions] == [True, False] and seen == ["c1", "c2"]
    assert "approved wipe (call c1)" in caplog.text and "denied wipe (call c2)" in caplog.text
    first, second = ApprovalPending("apr_a"), ApprovalPending("apr_b")
    port = ScriptedApproval([first, ApprovalDecision(approved=True), second])
    with pytest.raises(ApprovalPending) as caught:
        await ask_each(port, [req("c1"), req("c2"), req("c3")])
    assert caught.value is first and len(port.requests) == 3


async def test_nothing_to_ask():
    assert await ask_each(ScriptedApproval([]), []) == []


async def test_two_phase_consumes_nothing_until_every_call_is_decided():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    asks = [req("c1"), req("c2")]
    with pytest.raises(ApprovalPending) as caught:
        await ask_each(port, asks)
    assert caught.value.call_id == "c1" and len(await store.list()) == 2
    first = await store.find("run_1", "c1")
    await store.decide(first.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ApprovalPending) as caught:
        await ask_each(port, asks)
    assert caught.value.call_id == "c2"
    assert not (await store.get(first.approval_id)).consumed
    second = await store.find("run_1", "c2")
    await store.decide(second.approval_id, ApprovalDecision(approved=False, reason="no"))
    decisions = await ask_each(port, asks)
    assert [d.approved for d in decisions] == [True, False]
    records = await store.list()
    assert len(records) == 2 and all(r.consumed for r in records)


async def test_two_phase_vets_before_consuming():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    record = await port.prepare(req("c1"))
    await store.decide(record.approval_id, ApprovalDecision(approved=True))

    def refuse(request: ApprovalRequest, decision: ApprovalDecision) -> None:
        raise ToolUsageError("refused")

    with pytest.raises(ToolUsageError):
        await ask_each(port, [req("c1")], vet=refuse)
    assert not (await store.get(record.approval_id)).consumed


async def test_p6_a_record_replaced_between_phases_is_never_applied(caplog):
    """A concurrent resume consumes the vetted record and a new, edited one is decided."""
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    vetted = await port.prepare(req("c1"))
    await store.decide(vetted.approval_id, ApprovalDecision(approved=True))

    async def race() -> None:
        assert await store.consume(vetted.approval_id)  # the other resume wins
        rival = await store.create(req("c1").model_copy(update={"approval_id": "apr_rival"}))
        edit = ApprovalDecision(approved=True, edited_args={"path": "/edited-unvetted"})
        await store.decide(rival.approval_id, edit)

    class RacingPort(DeferredApproval):
        async def consume(self, record, request):  # type: ignore[no-untyped-def]
            await race()
            return await super().consume(record, request)

    (decision,) = await ask_each(RacingPort(store), [req("c1")])
    assert not decision.approved and decision.edited_args is None
    assert decision.decided_by == "agent_adapter"
    rival = await store.get("apr_rival")
    assert not rival.consumed  # the rival decision belongs to whoever created it
    assert "used by a concurrent request" in decision.reason
    assert "was not usable: denying" in caplog.text
    assert "used by a concurrent request" not in caplog.text  # the reason stays at DEBUG


async def test_p7_a_lost_race_leaves_no_phantom_pending_record():
    store = MemoryApprovalStore()
    vetted = await DeferredApproval(store).prepare(req("c1"))
    await store.decide(vetted.approval_id, ApprovalDecision(approved=True))

    class LosingPort(DeferredApproval):
        async def consume(self, record, request):  # type: ignore[no-untyped-def]
            await store.consume(record.approval_id)  # someone else consumes first
            return await super().consume(record, request)

    (decision,) = await ask_each(LosingPort(store), [req("c1")])
    assert not decision.approved
    records = await store.list()
    assert [r.approval_id for r in records] == [vetted.approval_id]
    assert await store.find("run_1", "c1") is None


async def test_p5_a_wrapped_deferred_approval_resolves_in_three_passes():
    store = MemoryApprovalStore()
    port = RecordingApproval(DeferredApproval(store))
    asks = [req("c1"), req("c2")]
    for call_id in ("c1", "c2"):  # one decision per pass
        with pytest.raises(ApprovalPending):
            await ask_each(port, asks)
        record = await store.find("run_1", call_id)
        await store.decide(record.approval_id, ApprovalDecision(approved=True))
    decisions = await ask_each(port, asks)  # third pass
    assert [d.approved for d in decisions] == [True, True]
    assert len(await store.list()) == 2 and len(port.decisions) == 2
    assert [r.call_id for r in port.requests] == ["c1", "c2"] * 3


async def test_deferred_consume_refuses_an_undecided_record_without_consuming():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    record = await port.prepare(req("c1"))
    with pytest.raises(ApprovalPending):
        await port.consume(record, req("c1"))
    assert not (await store.get(record.approval_id)).consumed
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    decided = await store.get(record.approval_id)
    assert (await port.consume(decided, req("c1"))).approved
    with pytest.raises(ApprovalDenied):
        await port.consume(decided, req("c1"))


async def test_s1_a_port_returning_another_decision_than_it_prepared_is_denied(caplog):
    from agent_adapter.approval.store import ApprovalRecord

    vetted = ApprovalDecision(approved=True, edited_args={"path": "/vetted"})

    class Storeless:
        """A PreparedApproval without a store, answering consume with other edits."""

        async def request(self, request: ApprovalRequest) -> ApprovalDecision:
            raise AssertionError("two-phase ports are not asked with request()")

        async def prepare(self, request: ApprovalRequest) -> ApprovalRecord:
            return ApprovalRecord(request=request, status="approved", decision=vetted)

        async def consume(
            self, record: ApprovalRecord, request: ApprovalRequest
        ) -> ApprovalDecision:
            return ApprovalDecision(approved=True, edited_args={"path": "/other"})

    seen: list[ApprovalDecision] = []
    (decision,) = await ask_each(Storeless(), [req("c1")], vet=lambda r, d: seen.append(d))
    assert seen == [vetted]
    assert not decision.approved and decision.edited_args is None
    assert "differs from the vetted one" in caplog.text
