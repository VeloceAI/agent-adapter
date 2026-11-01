import asyncio
import threading

import pytest

from agent_adapter.approval.store import ApprovalRecord, ApprovalStore, MemoryApprovalStore
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime import run_sync
from tests.approval.support import ids_from_threads

YES = ApprovalDecision(approved=True, decided_by="ana")
NO = ApprovalDecision(approved=False, reason="too risky", decided_by="ben", decided_at=5.0)


def req(run_id: str = "run_1", call_id: str = "call_1", approval_id: str = "") -> ApprovalRequest:
    return ApprovalRequest(approval_id=approval_id, run_id=run_id, call_id=call_id, tool="delete")


async def test_create_fills_id_and_created_at_and_get_returns_it():
    store = MemoryApprovalStore()
    assert isinstance(store, ApprovalStore)
    record = await store.create(req())
    assert record.approval_id.startswith("apr_") and record.request.created_at > 0
    assert (record.status, record.decision, record.consumed) == ("pending", None, False)
    assert await store.get(record.approval_id) == record


async def test_create_keeps_a_given_id_and_created_at():
    store = MemoryApprovalStore()
    request = req(approval_id="apr_x").model_copy(update={"created_at": 7.0})
    record = await store.create(request)
    assert record.request == request and record.approval_id == "apr_x"


async def test_create_with_a_taken_id_is_a_usage_error():
    store = MemoryApprovalStore()
    await store.create(req(approval_id="apr_x"))
    with pytest.raises(ToolUsageError, match="already exists"):
        await store.create(req(call_id="call_2", approval_id="apr_x"))


async def test_unknown_ids_raise_not_found():
    store = MemoryApprovalStore()
    for call in (store.get("nope"), store.decide("nope", YES), store.consume("nope")):
        with pytest.raises(ApprovalNotFound):
            await call


async def test_decide_sets_status_and_decided_at_once():
    store = MemoryApprovalStore()
    a = await store.create(req(call_id="a"))
    b = await store.create(req(call_id="b"))
    approved = await store.decide(a.approval_id, YES)
    denied = await store.decide(b.approval_id, NO)
    assert approved.status == "approved" and approved.decision.decided_at > 0
    assert denied.status == "denied" and denied.decision == NO  # given decided_at kept
    with pytest.raises(ApprovalAlreadyDecided) as info:
        await store.decide(a.approval_id, NO)
    assert info.value.status == "approved"
    assert (await store.get(a.approval_id)).status == "approved"


async def test_find_returns_the_unconsumed_record_of_the_call():
    store = MemoryApprovalStore()
    assert await store.find("run_1", "call_1") is None
    first = await store.create(req())
    await store.create(req(call_id="other"))
    assert await store.find("run_1", "call_1") == first
    await store.decide(first.approval_id, YES)
    assert await store.consume(first.approval_id) is True
    assert await store.find("run_1", "call_1") is None
    assert (await store.get(first.approval_id)).consumed
    second = await store.create(req())
    assert second.approval_id != first.approval_id
    assert await store.find("run_1", "call_1") == second


async def test_create_returns_the_open_record_of_the_call_and_stores_nothing():
    store = MemoryApprovalStore()
    first = await store.create(req(approval_id="apr_a"))
    again = await store.create(req(approval_id="apr_b"))
    assert again == first and len(await store.list()) == 1
    await store.decide(first.approval_id, YES)
    assert await store.create(req()) == await store.get("apr_a")  # decided, not consumed
    await store.consume("apr_a")
    assert (await store.create(req(approval_id="apr_b"))).approval_id == "apr_b"


async def test_consume_is_a_test_and_set():
    store = MemoryApprovalStore()
    record = await store.create(req())
    await store.decide(record.approval_id, YES)
    assert await store.consume(record.approval_id) is True
    assert await store.consume(record.approval_id) is False


def test_concurrent_creates_across_two_run_sync_loops_store_one_record():
    store = MemoryApprovalStore()

    async def burst() -> list[str]:
        records = await asyncio.gather(*(store.create(req()) for _ in range(100)))
        return [r.approval_id for r in records]

    ids = ids_from_threads(burst)
    assert len(ids) == 200 and len(set(ids)) == 1
    assert len(run_sync(store.list())) == 1


async def test_list_filters_newest_first_and_limits():
    store = MemoryApprovalStore()
    a = await store.create(req(run_id="r1", call_id="a"))
    b = await store.create(req(run_id="r1", call_id="b"))
    c = await store.create(req(run_id="r2", call_id="c"))
    await store.decide(b.approval_id, YES)
    ids = [r.approval_id for r in await store.list()]
    assert ids == [c.approval_id, b.approval_id, a.approval_id]
    assert [r.approval_id for r in await store.list(run_id="r1")] == [b.approval_id, a.approval_id]
    assert [r.approval_id for r in await store.list(status="pending")] == [
        c.approval_id,
        a.approval_id,
    ]
    assert [r.approval_id for r in await store.list(run_id="r1", status="approved")] == [
        b.approval_id
    ]
    assert len(await store.list(limit=1)) == 1 and await store.list(limit=0) == []


async def test_list_rejects_bad_status_and_limit():
    store = MemoryApprovalStore()
    with pytest.raises(ConfigError, match="status"):
        await store.list(status="maybe")
    with pytest.raises(ConfigError, match="limit"):
        await store.list(limit=-1)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_max_records_must_be_a_positive_int(bad):
    with pytest.raises(ConfigError, match="max_records"):
        MemoryApprovalStore(max_records=bad)


async def test_eviction_drops_oldest_consumed_then_decided_then_pending(caplog):
    store = MemoryApprovalStore(max_records=3)
    p1 = await store.create(req(call_id="p1"))
    d1 = await store.create(req(call_id="d1"))
    c1 = await store.create(req(call_id="c1"))
    await store.decide(d1.approval_id, YES)
    await store.decide(c1.approval_id, YES)
    await store.consume(c1.approval_id)
    await store.create(req(call_id="n1"))  # evicts the consumed record
    assert {r.request.call_id for r in await store.list()} == {"p1", "d1", "n1"}
    await store.create(req(call_id="n2"))  # then the oldest decided one
    assert {r.request.call_id for r in await store.list()} == {"p1", "n1", "n2"}
    with caplog.at_level("WARNING"):
        await store.create(req(call_id="n3"))  # only pending left: the oldest goes, loudly
    assert {r.request.call_id for r in await store.list()} == {"n1", "n2", "n3"}
    assert "evicted pending approval" in caplog.text
    with pytest.raises(ApprovalNotFound):
        await store.get(p1.approval_id)
    assert await store.find("run_1", "p1") is None


async def test_eviction_keeps_the_calls_other_records_findable():
    store = MemoryApprovalStore(max_records=2)
    old = await store.create(req())
    await store.decide(old.approval_id, YES)
    await store.consume(old.approval_id)
    newer = await store.create(req())
    other = await store.create(req(call_id="other"))  # evicts `old`; `newer` shares its call
    assert await store.find("run_1", "call_1") == newer
    assert [r.approval_id for r in await store.list()] == [other.approval_id, newer.approval_id]


async def test_eviction_prefers_the_earliest_consumed_record():
    store = MemoryApprovalStore(max_records=3)
    a = await store.create(req(call_id="a"))
    b = await store.create(req(call_id="b"))
    for record in (a, b):
        await store.decide(record.approval_id, YES)
    await store.consume(b.approval_id)
    await store.consume(a.approval_id)
    await store.create(req(call_id="c"))
    await store.create(req(call_id="d"))  # evicts b (consumed first)
    assert {r.request.call_id for r in await store.list()} == {"a", "c", "d"}


async def test_record_is_frozen():
    record = await MemoryApprovalStore().create(req())
    assert isinstance(record, ApprovalRecord)
    with pytest.raises(ValueError):
        record.status = "approved"  # type: ignore[misc]


async def test_concurrent_decides_have_exactly_one_winner():
    store = MemoryApprovalStore()
    record = await store.create(req())

    async def attempt(i: int) -> str:
        try:
            await store.decide(record.approval_id, ApprovalDecision(approved=i % 2 == 0))
        except ApprovalAlreadyDecided:
            return "lost"
        return "won"

    outcomes = await asyncio.gather(*(attempt(i) for i in range(100)))
    assert outcomes.count("won") == 1 and outcomes.count("lost") == 99


def test_concurrent_decides_across_two_run_sync_loops_have_one_winner():
    store = MemoryApprovalStore()
    record = run_sync(store.create(req()))
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    async def burst() -> None:
        async def attempt() -> str:
            try:
                await store.decide(record.approval_id, YES)
            except ApprovalAlreadyDecided:
                return "lost"
            return "won"

        results = await asyncio.gather(*(attempt() for _ in range(50)))
        with lock:
            outcomes.extend(results)

    def worker() -> None:
        barrier.wait()
        run_sync(burst)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert outcomes.count("won") == 1 and outcomes.count("lost") == 99


async def test_a_pending_record_cannot_be_consumed_and_is_queued_once_decided():
    store = MemoryApprovalStore(max_records=2)
    early = await store.create(req(call_id="w"))
    assert await store.consume(early.approval_id) is False  # only a decision can be used up
    assert not (await store.get(early.approval_id)).consumed
    await store.decide(early.approval_id, YES)
    kept = await store.create(req(call_id="k"))
    await store.decide(kept.approval_id, YES)
    await store.create(req(call_id="n"))  # the oldest decided record goes first
    assert {r.request.call_id for r in await store.list()} == {"k", "n"}
