"""Every BudgetStore: holds, settles, releases, expiry, windows and racing reservations."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.builtin.budgets import BudgetStore, BudgetUsage
from agent_adapter.runtime.loop_local import LoopLocal
from tests.hooks.builtin.budgets.stores import BACKENDS, CLOCKED, Clock, make_store

HOLD = 60.0


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture(params=BACKENDS)
async def store(request, tmp_path, clock) -> AsyncIterator[BudgetStore]:
    yield make_store(request.param, tmp_path, clock)
    await LoopLocal.aclose_all()  # engines and clients bound to this test's loop


async def reserve(store: BudgetStore, rid: str, usd: float, limit: float | None = 1.0, **kw):
    kw.setdefault("hold_s", HOLD)
    kw.setdefault("expire_at", None)
    return await store.reserve("k", rid, usd, limit_usd=limit, **kw)


async def test_every_backend_is_a_budget_store(store):
    assert isinstance(store, BudgetStore)


async def test_an_unknown_counter_is_empty(store):
    assert await store.usage("nobody") == BudgetUsage(0.0, 0.0)


async def test_a_hold_that_fits_is_kept_and_counted(store):
    first = await reserve(store, "r1", 0.4)
    assert first.accepted and first.usage == BudgetUsage(0.0, 0.0)
    second = await reserve(store, "r2", 0.5)
    assert second.accepted and second.usage.reserved_usd == pytest.approx(0.4)
    assert (await store.usage("k")).reserved_usd == pytest.approx(0.9)


async def test_a_hold_that_does_not_fit_is_refused_and_not_kept(store):
    assert (await reserve(store, "r1", 0.7)).accepted
    refused = await reserve(store, "r2", 0.4)
    assert not refused.accepted
    assert refused.usage.committed_usd == pytest.approx(0.7)
    assert (await store.usage("k")).reserved_usd == pytest.approx(0.7)


async def test_at_the_limit_even_a_free_hold_is_refused(store):
    assert (await reserve(store, "r1", 1.0)).accepted  # exactly fits
    assert not (await reserve(store, "r2", 0.0)).accepted


async def test_no_limit_always_holds(store):
    for i in range(3):
        assert (await reserve(store, f"r{i}", 5.0, limit=None)).accepted
    assert (await store.usage("k")).reserved_usd == pytest.approx(15.0)


async def test_settle_swaps_the_hold_for_the_real_cost(store):
    await reserve(store, "r1", 0.5)
    await store.settle("k", "r1", 0.2, expire_at=None)
    assert await store.usage("k") == BudgetUsage(pytest.approx(0.2), 0.0)
    assert (await reserve(store, "r2", 0.8)).accepted  # 0.2 spent + 0.8 fits 1.0


async def test_settle_counts_a_cost_whose_hold_is_gone(store):
    await reserve(store, "r1", 0.5)
    await store.release("k", "r1")
    await store.settle("k", "r1", 0.3, expire_at=None)
    assert (await store.usage("k")).spent_usd == pytest.approx(0.3)


async def test_release_frees_the_hold_and_spends_nothing(store):
    await reserve(store, "r1", 0.9)
    await store.release("k", "r1")
    await store.release("k", "unknown")  # no error
    assert await store.usage("k") == BudgetUsage(0.0, 0.0)


async def test_counters_are_separate(store):
    await store.reserve("a", "r1", 1.0, limit_usd=1.0, hold_s=HOLD, expire_at=None)
    assert (
        await store.reserve("b", "r1", 1.0, limit_usd=1.0, hold_s=HOLD, expire_at=None)
    ).accepted
    assert (await store.usage("a")).reserved_usd == pytest.approx(1.0)


async def test_racing_reservations_never_overspend(store):
    async def one(i: int) -> bool:
        return (await reserve(store, f"r{i}", 0.1)).accepted

    results = await asyncio.gather(*(one(i) for i in range(25)))
    assert sum(results) == 10
    assert (await store.usage("k")).committed_usd == pytest.approx(1.0)


async def test_an_ended_window_drops_the_counter(store, clock):
    past = min(clock.now, time.time()) - 1  # redis judges by its own clock
    await store.settle("k", "r1", 0.6, expire_at=past)
    assert await store.usage("k") == BudgetUsage(0.0, 0.0)


@pytest.mark.parametrize("kind", CLOCKED)
async def test_a_held_estimate_frees_itself_after_hold_s(kind, tmp_path, clock):
    store = make_store(kind, tmp_path, clock)
    await reserve(store, "r1", 0.9, hold_s=10)
    clock.now += 11
    assert (await store.usage("k")).reserved_usd == 0.0
    assert (await reserve(store, "r2", 0.9)).accepted
    await LoopLocal.aclose_all()


@pytest.mark.parametrize("kind", CLOCKED)
async def test_a_counter_resets_once_its_window_ends(kind, tmp_path, clock):
    store = make_store(kind, tmp_path, clock)
    end = clock.now + 100
    await reserve(store, "r1", 0.5, expire_at=end)
    await store.settle("k", "r1", 0.9, expire_at=end)
    assert not (await reserve(store, "r2", 0.5, expire_at=end)).accepted
    clock.now = end + 1
    assert await store.usage("k") == BudgetUsage(0.0, 0.0)
    assert (await reserve(store, "r3", 0.5, expire_at=end + 1000)).accepted
    await LoopLocal.aclose_all()


async def test_redis_holds_expire_on_the_redis_clock(tmp_path, clock):
    store = make_store("redis", tmp_path, clock)
    await reserve(store, "r1", 0.9, hold_s=0.05)
    await asyncio.sleep(0.15)
    assert (await store.usage("k")).reserved_usd == 0.0
    await LoopLocal.aclose_all()


@pytest.mark.parametrize(
    ("key", "rid", "usd", "hold_s", "match"),
    [
        ("", "r", 0.1, 1.0, "key"),
        ("k", "", 0.1, 1.0, "reservation_id"),
        ("k", "r", -0.1, 1.0, "amount_usd"),
        ("k", "r", float("nan"), 1.0, "amount_usd"),
        ("k", "r", True, 1.0, "amount_usd"),
        ("k", "r", 0.1, 0.0, "hold_s"),
        ("k", "r", 0.1, float("inf"), "hold_s"),
    ],
)
async def test_bad_reserve_arguments_are_refused(store, key, rid, usd, hold_s, match):
    with pytest.raises(ConfigError, match=match):
        await store.reserve(key, rid, usd, limit_usd=1.0, hold_s=hold_s, expire_at=None)


async def test_bad_settle_and_release_arguments_are_refused(store):
    with pytest.raises(ConfigError, match="cost_usd"):
        await store.settle("k", "r", -1.0, expire_at=None)
    with pytest.raises(ConfigError, match="key"):
        await store.release("", "r")
