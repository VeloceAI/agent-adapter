import logging

from agent_adapter.cache import ToolResultCache, encode, tool_key
from agent_adapter.core.context import ToolContext
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.clock import FakeClock

CTX = ToolContext(run_id="r1", thread_id="t1")


async def test_set_then_get_round_trips_per_scope():
    port = DictCache()
    cache = ToolResultCache(port)
    for scope in ("run", "thread", "global"):
        assert await cache.set("search", {"q": scope}, [scope], ttl=None, scope=scope, ctx=CTX)
        envelope = await cache.get("search", {"q": scope}, scope=scope, ctx=CTX)
        assert envelope is not None and envelope.payload == [scope] and envelope.kind == "tool"
    keys = list(port.data)
    assert keys[0].startswith("aa:tool:v1:run=r1:search:")
    assert keys[1].startswith("aa:tool:v1:thread=t1:search:")
    assert keys[2].startswith("aa:tool:v1:global:search:")


async def test_scopes_isolate_runs_and_threads():
    cache = ToolResultCache(DictCache(), namespace="app")
    await cache.set("t", {}, 1, ttl=None, scope="run", ctx=CTX)
    other_run = ToolContext(run_id="r2", thread_id="t1")
    assert await cache.get("t", {}, scope="run", ctx=other_run) is None
    await cache.set("t", {}, 2, ttl=None, scope="thread", ctx=CTX)
    assert (await cache.get("t", {}, scope="thread", ctx=other_run)).payload == 2


async def test_ttl_is_passed_to_the_port():
    clock = FakeClock()
    cache = ToolResultCache(DictCache(clock))
    await cache.set("t", {}, 1, ttl=10, scope="run", ctx=CTX)
    clock.advance(11)
    assert await cache.get("t", {}, scope="run", ctx=CTX) is None


async def test_uncacheable_calls_skip_the_port():
    port = DictCache()
    cache = ToolResultCache(port)
    no_thread = ToolContext(run_id="r")
    assert not await cache.set("t", {}, 1, ttl=None, scope="thread", ctx=no_thread)
    assert await cache.get("t", {}, scope="thread", ctx=no_thread) is None
    assert not await cache.set("t", {"x": object()}, 1, ttl=None, scope="run", ctx=CTX)
    assert await cache.get("t", {"x": object()}, scope="run", ctx=CTX) is None
    assert not await cache.set("t", {}, object(), ttl=None, scope="run", ctx=CTX)
    assert port.calls == []


async def test_broken_backend_is_a_miss_with_a_warning(caplog):
    port = BrokenCache()
    cache = ToolResultCache(port)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.tool_cache"):
        assert await cache.get("t", {}, scope="run", ctx=CTX) is None
        assert not await cache.set("t", {}, 1, ttl=None, scope="run", ctx=CTX)
    assert port.calls == 2
    assert "cache get failed for tool t" in caplog.text
    assert "cache set failed for tool t" in caplog.text


async def test_corrupt_or_foreign_entries_are_misses():
    port = DictCache()
    cache = ToolResultCache(port)
    key = tool_key("aa", "t", {}, scope="run", run_id="r1", thread_id="")
    await port.set(key, b"garbage")
    assert await cache.get("t", {}, scope="run", ctx=CTX) is None
    await port.set(key, encode("llm", 1))
    assert await cache.get("t", {}, scope="run", ctx=CTX) is None
