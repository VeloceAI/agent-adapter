import logging

import pytest

from agent_adapter import Registry, ToolCall, ToolContext
from agent_adapter.exec import Executor
from agent_adapter.hooks import Hook, ShortCircuit
from tests.exec.helpers import check_events, make
from tests.fakes.cache import BrokenCache, DictCache

calls: list[str] = []
registry = Registry()


@registry.tool(cache_ttl=60, parallel_safe=True)
async def lookup(q: str, limit: int = 3) -> dict:
    """Search."""
    calls.append(q)
    return {"q": q, "limit": limit}


@registry.tool(cache_ttl=60, cache_scope="global")
async def shared(q: str) -> str:
    """Global cache."""
    calls.append(q)
    return q


@registry.tool(cache_ttl=60)
async def opaque(q: str) -> object:
    """Not JSON-encodable."""
    calls.append(q)
    return object()


@registry.tool
async def uncached(q: str) -> str:
    """No cache_ttl."""
    calls.append(q)
    return q


@pytest.fixture(autouse=True)
def _reset():
    calls.clear()


async def test_second_call_is_served_from_the_cache():
    cache = DictCache()
    ex, sink, ledger, _ = make(registry, cache=cache)
    ctx = ex.context(run_id="r")
    first = await ex.run_call(ToolCall(name="lookup", arguments={"q": "80C"}), ctx)
    second = await ex.run_call(ToolCall(name="lookup", arguments={"q": "80C", "limit": 3}), ctx)
    assert first.value == second.value == {"q": "80C", "limit": 3} and calls == ["80C"]
    assert (first.cached, second.cached, second.attempts) == (False, True, 0)
    assert check_events(sink)[-3:] == ["tool.started", "tool.cached", "tool.finished"]
    assert sink.events[-1].data["cached"] is True
    assert [e.cache_hit for e in ledger.entries("r")] == [False, True]
    assert [op for op, _ in cache.calls] == ["get", "set", "get"]
    key = cache.calls[1][1]
    assert key.startswith("aa:tool:") and "run=r" in key


async def test_namespace_prefixes_the_key():
    cache = DictCache()
    ex = Executor(registry, cache=cache, namespace="app")
    await ex.run("lookup", {"q": "x"}, ex.context(run_id="r"))
    assert cache.calls[-1][1].startswith("app:tool:")


async def test_run_scope_is_not_shared_across_runs_but_global_is():
    ex = Executor(registry, cache=DictCache())
    await ex.run("lookup", {"q": "a"}, ex.context(run_id="r1"))
    await ex.run("lookup", {"q": "a"}, ex.context(run_id="r2"))
    await ex.run("shared", {"q": "b"}, ex.context(run_id="r1"))
    await ex.run("shared", {"q": "b"}, ex.context(run_id="r2"))
    assert calls == ["a", "a", "b"]


async def test_a_broken_cache_acts_as_a_miss(caplog):
    cache = BrokenCache()
    ex = Executor(registry, cache=cache)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.tool_cache"):
        assert (await ex.run("lookup", {"q": "z"}))["q"] == "z"
    assert cache.calls == 2 and "cache get failed" in caplog.text


async def test_unencodable_values_are_not_stored():
    cache = DictCache()
    ex = Executor(registry, cache=cache)
    ctx = ex.context(run_id="r")
    await ex.run("opaque", {"q": "o"}, ctx)
    await ex.run("opaque", {"q": "o"}, ctx)
    assert calls == ["o", "o"] and cache.data == {}


async def test_failures_and_short_circuits_are_not_stored():
    class Block(Hook):
        async def before_tool(self, inv):
            return ShortCircuit({"canned": True}) if inv.call.arguments["q"] == "sc" else None

        async def after_tool(self, inv, result):
            if inv.call.arguments["q"] == "bad":
                raise ValueError("after hook")
            return None

    cache = DictCache()
    ex = Executor(registry, cache=cache, hooks=[Block()])
    ctx = ex.context(run_id="r")
    assert (await ex.run_call(ToolCall(name="lookup", arguments={"q": "sc"}), ctx)).ok
    assert not (await ex.run_call(ToolCall(name="lookup", arguments={"q": "bad"}), ctx)).ok
    assert cache.data == {}


async def test_tools_without_cache_ttl_never_touch_the_cache():
    cache = DictCache()
    ex = Executor(registry, cache=cache)
    await ex.run("uncached", {"q": "u"})
    assert cache.calls == []


async def test_after_tool_hooks_also_see_cached_results():
    seen = []

    class Spy(Hook):
        async def after_tool(self, inv, result):
            seen.append(result.cached)

    ex = Executor(registry, cache=DictCache(), hooks=[Spy()])
    ctx = ex.context(run_id="r")
    await ex.run("lookup", {"q": "h"}, ctx)
    await ex.run("lookup", {"q": "h"}, ctx)
    assert seen == [False, True]


async def test_the_context_cache_wins_over_the_executor_cache():
    mine, theirs = DictCache(), DictCache()
    ex = Executor(registry, cache=theirs)
    await ex.run("lookup", {"q": "c"}, ToolContext(run_id="r", cache=mine))
    assert mine.data and not theirs.calls
