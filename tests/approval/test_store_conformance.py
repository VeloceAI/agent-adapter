"""The ApprovalStore protocol, checked the same way on every shipped store (contract §7.3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime import run_sync
from tests.approval.stores import BACKENDS, EVICTING, Kit, make
from tests.approval.support import ids_from_threads

YES = ApprovalDecision(approved=True, decided_by="ana")
NO = ApprovalDecision(approved=False, reason="too risky", decided_by="ben", decided_at=5.0)


def req(run_id: str = "run_1", call_id: str = "call_1", approval_id: str = "") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id=run_id,
        call_id=call_id,
        tool="delete",
        arguments={"path": "/tmp/x", "depth": 2},
    )


@pytest.fixture(params=BACKENDS)
async def kit(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Kit]:
    made = make(request.param, tmp_path)
    yield made
    await made.aclose()


@pytest.fixture
def store(kit: Kit) -> ApprovalStore:
    return kit.store


async def test_create_fills_id_and_created_at_and_get_returns_it(store):
    assert isinstance(store, ApprovalStore)
    record = await store.create(req())
    assert record.approval_id.startswith("apr_") and record.request.created_at > 0
    assert (record.status, record.decision, record.consumed) == ("pending", None, False)
    assert await store.get(record.approval_id) == record


async def test_create_keeps_a_given_id_and_created_at(store):
    request = req(approval_id="apr_x").model_copy(update={"created_at": 7.0})
    record = await store.create(request)
    assert record.request == request and record.approval_id == "apr_x"
    assert record.args_digest and (await store.get("apr_x")).args_digest == record.args_digest


async def test_create_with_a_taken_id_is_a_usage_error(store):
    await store.create(req(approval_id="apr_x"))
    with pytest.raises(ToolUsageError, match="already exists"):
        await store.create(req(call_id="call_2", approval_id="apr_x"))
    assert await store.find("run_1", "call_2") is None


async def test_unknown_ids_raise_not_found(store):
    for call in (store.get("nope"), store.decide("nope", YES), store.consume("nope")):
        with pytest.raises(ApprovalNotFound):
            await call


async def test_decide_sets_status_and_decided_at_once(store):
    a = await store.create(req(call_id="a"))
    b = await store.create(req(call_id="b"))
    approved = await store.decide(a.approval_id, YES)
    denied = await store.decide(b.approval_id, NO)
    assert approved.status == "approved" and approved.decision.decided_at > 0
    assert denied.status == "denied" and denied.decision == NO  # given decided_at kept
    with pytest.raises(ApprovalAlreadyDecided) as info:
        await store.decide(a.approval_id, NO)
    assert info.value.status == "approved"
    assert await store.get(a.approval_id) == approved
    assert await store.get(b.approval_id) == denied


async def test_find_returns_pending_and_decided_records_until_consumed(store):
    assert await store.find("run_1", "call_1") is None
    first = await store.create(req())
    await store.create(req(call_id="other"))
    assert await store.find("run_1", "call_1") == first
    decided = await store.decide(first.approval_id, YES)
    assert await store.find("run_1", "call_1") == decided
    assert await store.consume(first.approval_id) is True
    assert await store.find("run_1", "call_1") is None
    assert (await store.get(first.approval_id)).consumed
    second = await store.create(req())
    assert second.approval_id != first.approval_id
    assert await store.find("run_1", "call_1") == second
    assert (
        await store.find("run_1", "missing") is None and await store.find("run_2", "call_1") is None
    )


async def test_create_returns_the_open_record_of_the_call_and_stores_nothing(store):
    first = await store.create(req(approval_id="apr_a"))
    again = await store.create(req(approval_id="apr_b"))
    assert again == first and len(await store.list()) == 1
    with pytest.raises(ApprovalNotFound):
        await store.get("apr_b")
    await store.decide(first.approval_id, YES)
    assert await store.create(req()) == await store.get("apr_a")  # decided, not consumed
    await store.consume("apr_a")
    assert (await store.create(req(approval_id="apr_b"))).approval_id == "apr_b"


async def test_consume_is_a_test_and_set_and_never_takes_a_pending_record(store):
    record = await store.create(req())
    assert await store.consume(record.approval_id) is False  # only a decision is used up
    assert not (await store.get(record.approval_id)).consumed
    await store.decide(record.approval_id, NO)
    assert await store.consume(record.approval_id) is True
    assert await store.consume(record.approval_id) is False
    after = await store.get(record.approval_id)
    assert (after.status, after.consumed, after.decision) == ("denied", True, NO)


async def test_list_filters_newest_first_and_limits(store):
    a = await store.create(req(run_id="r1", call_id="a"))
    b = await store.create(req(run_id="r1", call_id="b"))
    c = await store.create(req(run_id="r2", call_id="c"))
    await store.decide(b.approval_id, YES)

    async def ids(**filters):
        return [r.approval_id for r in await store.list(**filters)]

    assert await ids() == [c.approval_id, b.approval_id, a.approval_id]
    assert await ids(run_id="r1") == [b.approval_id, a.approval_id]
    assert await ids(status="pending") == [c.approval_id, a.approval_id]
    assert await ids(status="approved") == [b.approval_id]
    assert await ids(run_id="r1", status="approved") == [b.approval_id]
    assert await ids(run_id="r1", status="denied") == [] and await ids(run_id="r9") == []
    assert await ids(limit=1) == [c.approval_id] and await store.list(limit=0) == []


async def test_list_pages_past_records_the_filters_skip(store):
    made = [await store.create(req(run_id="r1", call_id=f"c{i}")) for i in range(105)]
    await store.create(req(run_id="r2", call_id="other"))
    for record in made[:3]:
        await store.decide(record.approval_id, YES)
    found = await store.list(run_id="r1", status="approved", limit=5)
    assert [r.approval_id for r in found] == [r.approval_id for r in reversed(made[:3])]
    assert len(await store.list(run_id="r1", limit=200)) == 105


async def test_list_rejects_bad_status_and_limit(store):
    with pytest.raises(ConfigError, match="status"):
        await store.list(status="maybe")
    with pytest.raises(ConfigError, match="limit"):
        await store.list(limit=-1)


async def test_records_round_trip_every_field(store):
    request = ApprovalRequest(
        approval_id="",
        run_id="rün \ud800",  # non-ASCII and a lone surrogate survive
        call_id="call:1:x",
        tool="node:flow.review",
        kind="node",
        arguments={"nested": [{"a": 1.5, "b": None, "c": True}], "big": 10**30, "s": "\ud83d"},
        hint="check twice",
        metadata={"flow": "f", "visit": 2},
    )
    record = await store.create(request)
    assert await store.get(record.approval_id) == record
    assert await store.find(request.run_id, request.call_id) == record
    edit = ApprovalDecision(approved=True, edited_args={"path": "ü"}, reason="ok", decided_by="a")
    decided = await store.decide(record.approval_id, edit)
    assert await store.get(record.approval_id) == decided
    assert [r.approval_id for r in await store.list(run_id=request.run_id)] == [record.approval_id]


@pytest.mark.parametrize(
    "edited", [{"a": (1,)}, {"b": b"x"}, {"c": {1, 2}}, {"d": float("nan")}, {"e": {1: 2}}]
)
async def test_edited_args_that_are_not_plain_json_are_refused_before_any_write(store, edited):
    record = await store.create(req())
    with pytest.raises(ToolUsageError, match="not JSON") as info:
        await store.decide(record.approval_id, ApprovalDecision(approved=True, edited_args=edited))
    assert info.value.kind == ToolUsageError.kind
    assert await store.get(record.approval_id) == record
    assert await store.consume(record.approval_id) is False


async def test_ids_that_share_a_prefix_or_separator_stay_apart(store):
    one = await store.create(req(run_id="a:b", call_id="c"))
    two = await store.create(req(run_id="a", call_id="b:c"))
    assert one.approval_id != two.approval_id
    assert (await store.find("a:b", "c")).approval_id == one.approval_id
    assert (await store.find("a", "b:c")).approval_id == two.approval_id


async def test_a_second_store_over_the_same_backend_shares_every_record(kit):
    other = kit.other()
    record = await kit.store.create(req())
    assert await other.find("run_1", "call_1") == record
    assert await other.create(req()) == record
    await other.decide(record.approval_id, YES)
    assert await kit.store.consume(record.approval_id) is True
    assert await other.consume(record.approval_id) is False
    assert (await other.get(record.approval_id)).consumed


async def test_concurrent_creates_in_one_loop_store_one_record(store):
    records = await asyncio.gather(*(store.create(req()) for _ in range(40)))
    assert len({r.approval_id for r in records}) == 1
    assert len(await store.list()) == 1


def test_concurrent_creates_across_two_run_sync_loops_store_one_record(store):
    async def burst() -> list[str]:
        records = await asyncio.gather(*(store.create(req()) for _ in range(30)))
        return [r.approval_id for r in records]

    ids = ids_from_threads(burst)
    assert len(ids) == 60 and len(set(ids)) == 1
    assert len(run_sync(store.list())) == 1


async def test_concurrent_decides_have_exactly_one_winner(store):
    record = await store.create(req())

    async def attempt(i: int) -> str:
        try:
            await store.decide(record.approval_id, ApprovalDecision(approved=i % 2 == 0))
        except ApprovalAlreadyDecided:
            return "lost"
        return "won"

    outcomes = await asyncio.gather(*(attempt(i) for i in range(30)))
    assert outcomes.count("won") == 1 and outcomes.count("lost") == 29


async def test_concurrent_consumes_in_one_loop_have_exactly_one_winner(store):
    record = await store.create(req())
    await store.decide(record.approval_id, YES)
    won = await asyncio.gather(*(store.consume(record.approval_id) for _ in range(10)))
    assert won.count(True) == 1 and won.count(False) == 9


def test_concurrent_consumes_across_two_run_sync_loops_have_one_winner(store):
    record = run_sync(store.create(req()))
    run_sync(store.decide(record.approval_id, YES))

    async def burst() -> list[str]:
        won = await asyncio.gather(*(store.consume(record.approval_id) for _ in range(20)))
        return [str(flag) for flag in won]

    flags = ids_from_threads(burst)
    assert len(flags) == 40 and flags.count("True") == 1


# --- eviction: the stores bounded by max_records -------------------------------------------


@pytest.fixture(params=EVICTING)
def small(request: pytest.FixtureRequest, tmp_path: Path) -> ApprovalStore:
    return make(request.param, tmp_path, max_records=3).store


async def test_eviction_drops_oldest_consumed_then_decided_then_pending(small, caplog):
    p1 = await small.create(req(call_id="p1"))
    d1 = await small.create(req(call_id="d1"))
    c1 = await small.create(req(call_id="c1"))
    await small.decide(d1.approval_id, YES)
    await small.decide(c1.approval_id, YES)
    await small.consume(c1.approval_id)
    await small.create(req(call_id="n1"))  # evicts the consumed record
    assert {r.request.call_id for r in await small.list()} == {"p1", "d1", "n1"}
    with caplog.at_level("WARNING"):
        await small.create(req(call_id="n2"))  # then the oldest decided one, loudly
    assert f"evicted decided approval {d1.approval_id!r}" in caplog.text.replace('"', "'")
    caplog.clear()
    assert {r.request.call_id for r in await small.list()} == {"p1", "n1", "n2"}
    with caplog.at_level("WARNING"):
        await small.create(req(call_id="n3"))  # only pending left: the oldest goes, loudly
    assert {r.request.call_id for r in await small.list()} == {"n1", "n2", "n3"}
    assert "evicted pending approval" in caplog.text
    with pytest.raises(ApprovalNotFound):
        await small.get(p1.approval_id)
    assert await small.find("run_1", "p1") is None


async def test_eviction_prefers_the_earliest_consumed_and_keeps_the_call_findable(small):
    a = await small.create(req(call_id="a"))
    b = await small.create(req(call_id="b"))
    for record in (a, b):
        await small.decide(record.approval_id, YES)
    await small.consume(b.approval_id)
    await small.consume(a.approval_id)
    newer = await small.create(req(call_id="b"))
    await small.create(req(call_id="d"))  # evicts b (consumed first), not a
    assert {r.approval_id for r in await small.list()} == {
        a.approval_id,
        newer.approval_id,
        (await small.find("run_1", "d")).approval_id,
    }
    assert await small.find("run_1", "b") == newer
