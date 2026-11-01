import asyncio
import logging
from contextlib import aclosing

import pytest

from agent_adapter import ApprovalPending, Registry, ToolCall, ToolError
from agent_adapter.errors import ToolUsageError
from agent_adapter.exec import Executor, RunLimits
from tests.exec.helpers import DeferredFake, GraphBubbleUp

registry = Registry()
log: list[tuple[str, str]] = []
cancelled: list[str] = []


@registry.tool(parallel_safe=True)
async def wait(name: str, delay: float = 0.0, fail: str = "") -> str:
    """Sleep, then return the name (or raise `fail`)."""
    log.append(("in", name))
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        cancelled.append(name)
        raise
    log.append(("out", name))
    if fail == "error":
        raise ToolError(f"{name} failed")
    if fail == "bubble":
        raise GraphBubbleUp(name)
    if fail == "cancel":
        raise asyncio.CancelledError
    return name


@registry.tool
async def serial(name: str) -> str:
    """Not parallel-safe."""
    log.append(("in", name))
    await asyncio.sleep(0.005)
    log.append(("out", name))
    return name


@registry.tool(confirm=True, parallel_safe=True)
async def guarded(name: str) -> str:
    """Needs approval."""
    return name


@registry.tool(kind="search", parallel_safe=True)
async def search(name: str) -> str:
    """Kind-capped."""
    log.append(("in", name))
    await asyncio.sleep(0.005)
    log.append(("out", name))
    return name


@pytest.fixture(autouse=True)
def _reset():
    log.clear()
    cancelled.clear()


def _calls(*specs):
    return [ToolCall(name="wait", arguments=spec) for spec in specs]


async def test_run_many_keeps_input_order():
    ex = Executor(registry)
    calls = _calls(
        {"name": "slow", "delay": 0.03}, {"name": "fast"}, {"name": "mid", "delay": 0.01}
    )
    results = await ex.run_many(calls)
    assert [r.value for r in results] == ["slow", "fast", "mid"]
    assert len({r.call_id for r in results}) == 3


async def test_run_many_collects_errors_in_place():
    results = await Executor(registry).run_many(
        _calls({"name": "a"}, {"name": "b", "fail": "error"}, {"name": "c"})
    )
    assert [r.status for r in results] == ["ok", "error", "ok"]
    assert results[1].error.kind == "tool-error"


async def test_fail_fast_raises_the_first_error_unwrapped_and_cancels_siblings():
    calls = _calls({"name": "bad", "fail": "error"}, {"name": "slow", "delay": 1})
    with pytest.raises(ToolError, match="bad failed"):
        await Executor(registry).run_many(calls, on_error="fail_fast")
    assert cancelled == ["slow"]


async def test_pending_is_a_result_in_collect_and_raises_in_fail_fast():
    ex = Executor(registry, approval=DeferredFake())
    calls = [ToolCall(name="guarded", arguments={"name": "g"}), *_calls({"name": "w"})]
    results = await ex.run_many(calls)
    assert [r.status for r in results] == ["pending", "ok"]
    with pytest.raises(ApprovalPending):
        await ex.run_many(calls, on_error="fail_fast")


@pytest.mark.parametrize("mode", ["collect", "fail_fast"])
@pytest.mark.parametrize(
    ("fail", "signal"), [("bubble", GraphBubbleUp), ("cancel", asyncio.CancelledError)]
)
async def test_control_flow_escapes_run_many_unwrapped(mode, fail, signal):
    calls = _calls({"name": "sig", "fail": fail}, {"name": "slow", "delay": 1})
    with pytest.raises(signal) as info:
        await Executor(registry).run_many(calls, on_error=mode)
    assert not isinstance(info.value, BaseExceptionGroup)
    assert cancelled == ["slow"]


async def test_concurrency_limits_the_batch():
    calls = _calls(*({"name": f"n{i}", "delay": 0.005} for i in range(4)))
    await Executor(registry).run_many(calls, concurrency=1)
    assert log == [(e, f"n{i}") for i in range(4) for e in ("in", "out")]


async def test_non_parallel_safe_calls_run_one_at_a_time(caplog):
    calls = [ToolCall(name="serial", arguments={"name": f"s{i}"}) for i in range(3)]
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.exec.parallel"):
        await Executor(registry).run_many(calls)
    assert log == [(e, f"s{i}") for i in range(3) for e in ("in", "out")]
    assert "runs one at a time" in caplog.text


async def test_an_exclusive_call_waits_for_parallel_safe_ones():
    calls = [
        *_calls({"name": "p", "delay": 0.01}),
        ToolCall(name="serial", arguments={"name": "x"}),
    ]
    await Executor(registry).run_many(calls)
    assert log.index(("out", "p")) < log.index(("in", "x"))


async def test_per_kind_caps_concurrency_within_a_run():
    ex = Executor(registry, limits=RunLimits(per_kind={"search": 1}))
    calls = [ToolCall(name="search", arguments={"name": f"q{i}"}) for i in range(3)]
    await ex.run_many(calls)
    assert log == [(e, f"q{i}") for i in range(3) for e in ("in", "out")]


async def test_max_calls_limits_distinct_calls_per_run():
    ex = Executor(registry, limits=RunLimits(max_calls=2))
    results = await ex.run_many(_calls({"name": "a"}, {"name": "b"}, {"name": "c"}))
    assert sorted(r.status for r in results) == ["error", "ok", "ok"]
    assert [r.error.kind for r in results if r.error] == ["run-limit-exceeded"]


async def test_one_fresh_run_per_batch_and_it_is_ended():
    ex = Executor(registry, limits=RunLimits(max_calls=2))
    for _ in range(2):  # a second batch starts a new run: the limit applies per batch
        assert all(r.ok for r in await ex.run_many(_calls({"name": "a"}, {"name": "b"})))
    assert len(ex._settings.runs) == 0


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"on_error": "ignore"}, "on_error"), ({"concurrency": 0}, "concurrency")]
)
async def test_bad_batch_options_raise_usage_errors(kwargs, match):
    with pytest.raises(ToolUsageError, match=match):
        await Executor(registry).run_many(_calls({"name": "a"}), **kwargs)


async def test_as_completed_streams_in_completion_order():
    ex = Executor(registry)
    calls = _calls(
        {"name": "slow", "delay": 0.03}, {"name": "fast"}, {"name": "mid", "delay": 0.01}
    )
    names = [r.value async for r in ex.as_completed(calls)]
    assert names == ["fast", "mid", "slow"]


async def test_as_completed_collects_errors():
    ex = Executor(registry)
    results = [r async for r in ex.as_completed(_calls({"name": "a", "fail": "error"}))]
    assert results[0].error.kind == "tool-error"


async def test_leaving_as_completed_early_cancels_the_rest():
    ex = Executor(registry)
    calls = _calls({"name": "fast"}, {"name": "slow", "delay": 1})
    async with aclosing(ex.as_completed(calls)) as stream:
        async for result in stream:
            assert result.value == "fast"
            break
    assert cancelled == ["slow"]


@pytest.mark.parametrize(
    ("fail", "signal"), [("bubble", GraphBubbleUp), ("cancel", asyncio.CancelledError)]
)
async def test_control_flow_escapes_as_completed_unwrapped(fail, signal):
    ex = Executor(registry)
    calls = _calls({"name": "sig", "fail": fail}, {"name": "slow", "delay": 1})
    with pytest.raises(signal):
        async with aclosing(ex.as_completed(calls)) as stream:
            async for _ in stream:
                pass
    assert cancelled == ["slow"]


def test_as_completed_checks_concurrency_eagerly():
    with pytest.raises(ToolUsageError):
        Executor(registry).as_completed([], concurrency=-1)
