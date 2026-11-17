"""Concurrent callers of one cached tool call share a single run."""

import asyncio

import pytest

from agent_adapter import Registry, ToolCall
from agent_adapter.cache._inflight import _FLIGHTS
from agent_adapter.errors import ToolError
from agent_adapter.hooks import Hook
from tests.exec.helpers import check_events, make
from tests.fakes.cache import DictCache

calls: list[str] = []
gate = asyncio.Event()
registry = Registry()


@registry.tool(cache_ttl=60, parallel_safe=True)
async def lookup(q: str) -> dict:
    """Search."""
    calls.append(q)
    await gate.wait()
    return {"q": q}


@registry.tool(cache_ttl=60, cache_scope="thread", parallel_safe=True)
async def per_thread(q: str) -> str:
    """Thread-scoped."""
    calls.append(q)
    await gate.wait()
    return q


@registry.tool(cache_ttl=60, cache_scope="global", parallel_safe=True)
async def public(q: str) -> str:
    """Global."""
    calls.append(q)
    await gate.wait()
    return q


@registry.tool(cache_ttl=60, parallel_safe=True)
async def flaky(q: str) -> str:
    """Fails."""
    calls.append(q)
    await gate.wait()
    raise ToolError("upstream down")


@registry.tool(cache_ttl=60, parallel_safe=True)
async def opaque(q: str) -> object:
    """Not JSON-encodable."""
    calls.append(q)
    await gate.wait()
    return object()


@registry.tool(parallel_safe=True)
async def uncached(q: str) -> str:
    """No cache_ttl."""
    calls.append(q)
    await gate.wait()
    return q


@pytest.fixture(autouse=True)
def _reset():
    global gate
    calls.clear()
    gate = asyncio.Event()  # one per test: an Event binds to the loop that first waits on it


async def _ticks(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def _together(ex, tool, contexts, args=None):
    tasks = [
        asyncio.ensure_future(ex.run_call(ToolCall(name=tool, arguments=args or {"q": "x"}), ctx))
        for ctx in contexts
    ]
    await _ticks()
    gate.set()
    return await asyncio.gather(*tasks)


async def test_five_concurrent_callers_make_one_call():
    ex, sink, ledger, _ = make(registry, cache=DictCache())
    ctx = ex.context(run_id="r")
    results = await _together(ex, "lookup", [ctx] * 5)
    assert calls == ["x"]
    assert all(r.ok and r.value == {"q": "x"} for r in results)
    assert [r.cached for r in results].count(False) == 1
    values = [r.value for r in results]
    assert len({id(v) for v in values}) == 5  # every waiter got its own copy
    shared = [e for e in sink.events if e.type == "inflight.shared"]
    assert len(shared) == 4 and {e.data["kind"] for e in shared} == {"tool"}
    assert {e.data["name"] for e in shared} == {"lookup"}
    assert check_events(sink).count("tool.cached") == 4
    assert [e.cache_hit for e in ledger.entries("r")].count(True) == 4
    assert not _FLIGHTS.get()  # nothing left in flight


async def test_run_scope_keeps_runs_apart():
    ex, *_ = make(registry, cache=DictCache())
    await _together(ex, "lookup", [ex.context(run_id="a"), ex.context(run_id="b")])
    assert calls == ["x", "x"]


async def test_thread_scope_shares_within_a_thread_only():
    ex, *_ = make(registry, cache=DictCache())
    same = [ex.context(run_id="a", thread_id="t"), ex.context(run_id="b", thread_id="t")]
    await _together(ex, "per_thread", [*same, ex.context(run_id="c", thread_id="u")])
    assert calls == ["x", "x"]


async def test_global_scope_shares_across_runs():
    ex, *_ = make(registry, cache=DictCache())
    await _together(ex, "public", [ex.context(run_id="a"), ex.context(run_id="b")])
    assert calls == ["x"]


async def test_other_arguments_are_not_shared():
    ex, *_ = make(registry, cache=DictCache())
    ctx = ex.context(run_id="r")
    first = asyncio.ensure_future(ex.run_call(ToolCall(name="lookup", arguments={"q": "a"}), ctx))
    second = await _together(ex, "lookup", [ctx], {"q": "b"})
    await first
    assert sorted(calls) == ["a", "b"] and second[0].cached is False


async def test_no_cache_or_no_ttl_means_no_sharing():
    ex, *_ = make(registry)
    ctx = ex.context(run_id="r")
    await _together(ex, "lookup", [ctx] * 3)
    assert calls == ["x"] * 3
    calls.clear()
    ex, *_ = make(registry, cache=DictCache())
    await _together(ex, "uncached", [ex.context(run_id="r")] * 3)
    assert calls == ["x"] * 3


async def test_leader_failure_reaches_every_waiter():
    ex, sink, *_ = make(registry, cache=DictCache())
    results = await _together(ex, "flaky", [ex.context(run_id="r")] * 4)
    assert calls == ["x"]
    assert [r.status for r in results] == ["error"] * 4
    assert {r.error.kind for r in results} == {"tool-error"}
    assert not [e for e in sink.events if e.type == "inflight.shared"]
    assert not _FLIGHTS.get()


async def test_cancelled_leader_hands_the_call_to_one_waiter():
    ex, *_ = make(registry, cache=DictCache())
    ctx = ex.context(run_id="r")
    call = ToolCall(name="lookup", arguments={"q": "x"})
    leader = asyncio.ensure_future(ex.run_call(call, ctx))
    await _ticks()
    waiters = [asyncio.ensure_future(ex.run_call(call, ctx)) for _ in range(3)]
    await _ticks()
    leader.cancel()
    await _ticks()
    gate.set()
    results = await asyncio.gather(*waiters)
    with pytest.raises(asyncio.CancelledError):
        await leader
    assert calls == ["x", "x"]  # the cancelled leader's, then one waiter's
    assert all(r.ok and r.value == {"q": "x"} for r in results)
    assert [r.cached for r in results].count(False) == 1
    assert not _FLIGHTS.get()


async def test_cancelled_waiter_leaves_the_others_waiting():
    ex, *_ = make(registry, cache=DictCache())
    ctx = ex.context(run_id="r")
    call = ToolCall(name="lookup", arguments={"q": "x"})
    tasks = [asyncio.ensure_future(ex.run_call(call, ctx)) for _ in range(3)]
    await _ticks()
    tasks[1].cancel()
    await _ticks()
    gate.set()
    first, last = await tasks[0], await tasks[2]
    assert calls == ["x"] and first.ok and last.ok and last.cached


async def test_unencodable_result_is_not_shared():
    ex, *_ = make(registry, cache=DictCache())
    results = await _together(ex, "opaque", [ex.context(run_id="r")] * 3)
    assert calls == ["x"] * 3 and all(r.ok and not r.cached for r in results)


class _SlowAfter(Hook):
    """Holds the first call in its after-tool hook, between the run and the cache store."""

    def __init__(self) -> None:
        self.inside = asyncio.Event()
        self.release = asyncio.Event()

    async def after_tool(self, inv, result):
        if not self.inside.is_set():
            self.inside.set()
            await self.release.wait()
        return None


async def test_a_caller_between_run_and_store_still_shares():
    hook = _SlowAfter()
    ex, *_ = make(registry, cache=DictCache(), hooks=[hook])
    ctx = ex.context(run_id="r")
    call = ToolCall(name="lookup", arguments={"q": "x"})
    gate.set()
    leader = asyncio.ensure_future(ex.run_call(call, ctx))
    await hook.inside.wait()
    late = await ex.run_call(call, ctx)
    hook.release.set()
    assert (await leader).ok and late.ok and late.cached and calls == ["x"]
    assert not _FLIGHTS.get()
