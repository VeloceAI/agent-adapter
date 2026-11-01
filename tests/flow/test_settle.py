import pytest

from agent_adapter.approval import MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.flow._resume import SpentCheckpoints, settle_deferred

YES = ApprovalDecision(approved=True)


def _request(approval_id="apr_s"):
    return ApprovalRequest(
        approval_id=approval_id, run_id="r", call_id="c", tool="pay", arguments={"amount": 1}
    )


async def test_an_unknown_record_is_left_alone():
    await settle_deferred(MemoryApprovalStore(), _request(), YES)


async def test_a_record_decided_by_a_racer_is_checked_not_overwritten():
    store = MemoryApprovalStore()
    await store.create(_request())

    class Racing(MemoryApprovalStore):
        async def get(self, approval_id):
            return await store.get(approval_id)

        async def decide(self, approval_id, decision):
            await store.decide(approval_id, decision)  # a racer decided first
            raise ApprovalAlreadyDecided(approval_id, status="approved")

        async def consume(self, approval_id):
            return await store.consume(approval_id)

    await settle_deferred(Racing(), _request(), YES)
    assert (await store.get("apr_s")).consumed


async def test_a_record_consumed_between_check_and_use_is_refused():
    store = MemoryApprovalStore()
    await store.create(_request())

    class Consumed(MemoryApprovalStore):
        async def get(self, approval_id):
            return await store.get(approval_id)

        async def decide(self, approval_id, decision):
            return await store.decide(approval_id, decision)

        async def consume(self, approval_id):
            return False

    with pytest.raises(ApprovalAlreadyDecided):
        await settle_deferred(Consumed(), _request(), YES)


def test_the_spent_registry_forgets_the_oldest_and_the_expired():
    now = [0.0]
    spent = SpentCheckpoints(max_entries=2, ttl=10.0, clock=lambda: now[0])
    for n in range(3):
        spent.spend(("r", f"apr_{n}"))
    spent.check(("r", "apr_0"))  # evicted by size: only a store could refuse it now
    with pytest.raises(ApprovalAlreadyDecided):
        spent.check(("r", "apr_2"))
    with pytest.raises(ApprovalAlreadyDecided):
        spent.spend(("r", "apr_1"))  # a racer spent it first
    now[0] = 11.0
    spent.check(("r", "apr_2"))  # expired
    spent.spend(("r", "apr_2"))
    spent.clear()
    spent.check(("r", "apr_2"))
