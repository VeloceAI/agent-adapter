import pytest

from agent_adapter.cache.memory import MemoryCache
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.ports import CachePort
from tests.fakes.clock import FakeClock


def test_memory_cache_is_a_cache_port():
    assert isinstance(MemoryCache(), CachePort)


async def test_set_get_delete_round_trip():
    cache = MemoryCache()
    assert await cache.get("k") is None
    await cache.set("k", b"v")
    assert await cache.get("k") == b"v"
    await cache.delete("k")
    assert await cache.get("k") is None
    await cache.delete("missing")  # no error


async def test_entries_expire_after_ttl():
    clock = FakeClock()
    cache = MemoryCache(clock=clock)
    await cache.set("k", b"v", ttl=10)
    clock.advance(9.9)
    assert await cache.get("k") == b"v"
    clock.advance(0.1)
    assert await cache.get("k") is None
    assert len(cache) == 0


async def test_no_ttl_never_expires():
    clock = FakeClock()
    cache = MemoryCache(clock=clock)
    await cache.set("k", b"v", ttl=None)
    clock.advance(1e9)
    assert await cache.get("k") == b"v"


async def test_non_positive_ttl_removes_the_entry():
    cache = MemoryCache()
    await cache.set("k", b"old")
    await cache.set("k", b"new", ttl=0)
    assert await cache.get("k") is None
    await cache.set("j", b"v", ttl=-1)
    assert len(cache) == 0


async def test_least_recently_used_entry_is_evicted():
    cache = MemoryCache(max_entries=2)
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    assert await cache.get("a") == b"1"  # a is now most recent
    await cache.set("c", b"3")
    assert await cache.get("b") is None
    assert await cache.get("a") == b"1"
    assert await cache.get("c") == b"3"


async def test_overwrite_refreshes_value_ttl_and_recency():
    clock = FakeClock()
    cache = MemoryCache(max_entries=2, clock=clock)
    await cache.set("a", b"1", ttl=5)
    await cache.set("b", b"2")
    await cache.set("a", b"1b", ttl=50)
    clock.advance(10)
    await cache.set("c", b"3")
    assert await cache.get("a") == b"1b"
    assert await cache.get("b") is None


async def test_expired_entries_are_purged_before_evicting_live_ones():
    clock = FakeClock()
    cache = MemoryCache(max_entries=2, clock=clock)
    await cache.set("live", b"1")
    await cache.set("short", b"2", ttl=1)
    clock.advance(2)
    await cache.set("new", b"3")
    assert await cache.get("live") == b"1"
    assert await cache.get("new") == b"3"


@pytest.mark.parametrize("bad", [0, -3])
def test_max_entries_must_be_positive(bad):
    with pytest.raises(ConfigError, match="max_entries"):
        MemoryCache(max_entries=bad)


async def test_full_cache_does_not_rescan_for_expired_entries_on_every_set():
    calls = [0]
    clock = FakeClock()

    def counting_clock():
        calls[0] += 1
        return clock()

    cache = MemoryCache(max_entries=100, clock=counting_clock)
    for i in range(100):
        await cache.set(f"k{i}", b"v")
    calls[0] = 0
    for i in range(1000):
        await cache.set(f"n{i}", b"v")  # each set overflows by one
    # One expiry scan per max_entries overflows, not one per set.
    assert calls[0] <= 1000 / 100 + 1
    assert len(cache) == 100


@pytest.mark.parametrize("ttl", [float("inf"), float("nan")])
async def test_non_finite_ttl_never_expires(ttl):
    clock = FakeClock()
    cache = MemoryCache(clock=clock)
    await cache.set("k", b"v", ttl=ttl)
    clock.advance(1e12)
    assert await cache.get("k") == b"v"


async def test_negative_infinite_ttl_deletes_the_key():
    cache = MemoryCache()
    await cache.set("k", b"v")
    await cache.set("k", b"w", ttl=float("-inf"))
    assert await cache.get("k") is None
