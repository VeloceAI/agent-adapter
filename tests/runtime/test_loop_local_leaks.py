import asyncio
import gc
import threading
import time
import weakref

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.runtime.loop_local import LoopLocal


def _contend(local: LoopLocal[asyncio.Lock]) -> None:
    """Run a loop where the loop-local Lock gets a waiter, so it references its loop."""

    async def main() -> None:
        lock = local.get()
        async with lock:
            waiter = asyncio.ensure_future(lock.acquire())
            await asyncio.sleep(0)
        await waiter
        lock.release()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    refs.append(weakref.ref(loop))


refs: list[weakref.ref[asyncio.AbstractEventLoop]] = []


def test_closed_loops_are_dropped_even_when_the_value_references_the_loop():
    refs.clear()
    local = LoopLocal(asyncio.Lock)
    for _ in range(3):
        _contend(local)
    gc.collect()
    # each get() dropped the earlier closed loops; the last one is held by its Lock value
    assert [ref() is None for ref in refs] == [True, True, False]

    async def touch() -> None:
        local.get()

    asyncio.run(touch())  # get() drops the entries of closed loops
    gc.collect()
    assert [ref() for ref in refs] == [None, None, None]
    assert len(local._values) == 0  # the asyncio.run loop is closed and gone too


def test_bound_and_aclose_outside_a_loop_are_usage_errors():
    local = LoopLocal(object)
    with pytest.raises(ToolUsageError, match="running event loop"):
        local._bound()
    coro = local.aclose()
    with pytest.raises(ToolUsageError, match="running event loop"):
        coro.send(None)  # drive it outside any loop
    coro.close()


def test_get_creates_one_value_under_real_contention_on_one_loop():
    """Many threads see the same running loop at once; the lock allows one factory call."""
    calls = []

    def slow_factory() -> object:
        calls.append(1)
        time.sleep(0.01)  # widen the race window
        return object()

    local = LoopLocal(slow_factory)
    loop = asyncio.new_event_loop()
    start = threading.Barrier(8)
    seen: list[object] = []

    def worker() -> None:
        asyncio._set_running_loop(loop)
        try:
            start.wait(5)
            seen.append(local.get())
        finally:
            asyncio._set_running_loop(None)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    loop.close()
    assert len(calls) == 1
    assert len(seen) == 8 and len({id(v) for v in seen}) == 1
