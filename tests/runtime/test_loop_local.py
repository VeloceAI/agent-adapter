import asyncio
import gc
import logging
import threading

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.runtime.loop_local import LoopLocal


def test_get_outside_a_loop_is_a_usage_error():
    with pytest.raises(ToolUsageError, match="running event loop"):
        LoopLocal(object).get()


async def test_same_loop_gets_the_same_value():
    calls = []
    local = LoopLocal(lambda: calls.append(1) or object())
    assert local.get() is local.get()
    assert calls == [1]
    assert local._bound()


def test_each_loop_gets_its_own_value():
    local = LoopLocal(object)

    async def grab():
        return local.get()

    assert asyncio.run(grab()) is not asyncio.run(grab())


async def test_holds_asyncio_primitives_per_loop():
    local = LoopLocal(asyncio.Lock)
    async with local.get():
        assert local.get().locked()


async def test_aclose_closes_and_forgets_this_loops_value():
    closed = []

    async def close(value):
        closed.append(value)

    local = LoopLocal(object, close=close)
    first = local.get()
    await local.aclose()
    assert closed == [first]
    assert not local._bound()
    assert local.get() is not first
    await local.aclose()
    await local.aclose()  # nothing bound: no-op
    assert len(closed) == 2


async def test_aclose_without_close_just_forgets():
    local = LoopLocal(object)
    first = local.get()
    await local.aclose()
    assert local.get() is not first


async def test_aclose_all_closes_only_bound_instances_and_survives_failures(caplog):
    closed = []

    async def close(value):
        closed.append(value)

    async def broken(value):
        raise RuntimeError("close failed")

    bound = LoopLocal(object, close=close)
    failing = LoopLocal(object, close=broken)
    unbound = LoopLocal(object, close=close)
    value = bound.get()
    failing.get()
    with caplog.at_level(logging.WARNING, logger="agent_adapter.runtime.loop_local"):
        await LoopLocal.aclose_all()
    assert closed == [value]
    assert "closing a loop-local value failed" in caplog.text
    assert not bound._bound() and not failing._bound() and not unbound._bound()


def test_values_are_released_with_their_loop():
    local = LoopLocal(object)

    async def grab():
        local.get()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(grab())
    loop.close()
    assert len(local._values) == 1
    del loop
    gc.collect()
    assert len(local._values) == 0


def test_smoke_threads_with_their_own_loops_get_their_own_values():
    created = []
    local = LoopLocal(lambda: created.append(1) or object())
    start = threading.Barrier(8)
    seen: list[object] = []

    def run():
        async def grab():
            start.wait(5)
            return local.get(), local.get()

        a, b = asyncio.run(grab())
        assert a is b
        seen.append(a)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(created) == 8 and len({id(v) for v in seen}) == 8
