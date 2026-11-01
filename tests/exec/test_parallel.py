import asyncio

import pytest

from agent_adapter import ToolError
from agent_adapter.errors import ToolTimeout, ToolUsageError
from agent_adapter.exec import Backoff, Executor
from tests.exec.helpers import GraphBubbleUp
from tests.fakes.clock import FakeClock


def _executor(clock=None):
    clock = clock or FakeClock()
    return Executor(sleep=clock.sleep, rng=lambda: 0.0, backoff=Backoff(base_s=1.0, jitter=0))


async def value(v, delay=0.0):
    await asyncio.sleep(delay)
    return v


async def test_results_keep_input_order_for_callables_and_awaitables():
    ex = _executor()
    results = await ex.parallel(lambda: value("slow", 0.02), value("fast"), lambda: value(3))
    assert results == ["slow", "fast", 3]


async def test_concurrency_limits_running_items():
    running = peak = 0

    async def job():
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.005)
        running -= 1

    await _executor().parallel(*([job] * 5), concurrency=2)
    assert peak == 2


async def test_each_item_times_out_as_tool_timeout():
    with pytest.raises(ToolTimeout) as info:
        await _executor().parallel(lambda: value("ok"), lambda: value("late", 1), timeout=0.01)
    assert info.value.tool == "parallel[1]" and info.value.timeout == 0.01


async def test_retries_use_the_executor_backoff():
    clock, state = FakeClock(), {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    assert await _executor(clock).parallel(flaky, retries=2) == ["ok"]
    assert clock.sleeps == [1.0, 2.0]


async def test_non_retryable_errors_are_not_retried():
    state = {"n": 0}

    async def bad():
        state["n"] += 1
        raise ToolError("no")

    with pytest.raises(ToolError):
        await _executor().parallel(bad, retries=3)
    assert state["n"] == 1


async def test_collect_keeps_exceptions_in_place():
    async def boom():
        raise ValueError("x")

    results = await _executor().parallel(lambda: value(1), boom, on_error="collect")
    assert results[0] == 1 and isinstance(results[1], ValueError)


async def test_fail_fast_cancels_siblings_and_raises_unwrapped():
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def boom():
        raise ValueError("first")

    pending = value("never started")
    with pytest.raises(ValueError, match="first"):
        await _executor().parallel(slow, boom, pending, concurrency=2)
    assert cancelled == [True]


@pytest.mark.parametrize("mode", ["collect", "fail_fast"])
@pytest.mark.parametrize("signal", [GraphBubbleUp("x"), asyncio.CancelledError()])
async def test_control_flow_escapes_parallel_unwrapped(mode, signal):
    async def raise_signal():
        raise signal

    with pytest.raises(type(signal)) as info:
        await _executor().parallel(raise_signal, lambda: value(1, 1), on_error=mode)
    assert not isinstance(info.value, BaseExceptionGroup)


async def test_awaitables_cannot_be_retried():
    coro = value(1)
    with pytest.raises(ToolUsageError, match="cannot be retried"):
        await _executor().parallel(coro, retries=1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"retries": -1}, "retries"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": True}, "timeout"),
        ({"concurrency": 0}, "concurrency"),
        ({"on_error": "nope"}, "on_error"),
    ],
)
async def test_bad_options_raise_usage_errors_and_close_coroutines(kwargs, match):
    with pytest.raises(ToolUsageError, match=match):
        await _executor().parallel(value(1), **kwargs)


async def test_items_must_be_awaitable_or_callable():
    with pytest.raises(ToolUsageError, match="awaitable or a zero-argument callable"):
        await _executor().parallel(42)  # type: ignore[arg-type]


async def test_a_callable_must_return_an_awaitable():
    results = await _executor().parallel(lambda: 5, on_error="collect")  # type: ignore[arg-type,return-value]
    assert isinstance(results[0], ToolUsageError)
