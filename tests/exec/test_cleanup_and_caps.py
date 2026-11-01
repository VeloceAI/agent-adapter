"""Batch cleanup under an outer cancel, and the crossed per_kind caps caveat (fix round 2)."""

import asyncio
import contextlib
import gc
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest

from agent_adapter import Registry, ToolCall, ToolContext, ToolError
from agent_adapter.exec import Executor, RunLimits


class _Cleanup:
    """A body that, once cancelled, is still cleaning up when the outer timeout fires."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def body(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.started.set()
            await self.release.wait()
            raise


@contextlib.asynccontextmanager
async def _timeout_during(cleanup: _Cleanup) -> AsyncIterator[None]:
    """Fire the outer timeout exactly while `cleanup` runs, then let it finish (no wall clock)."""
    async with asyncio.timeout(None) as scope:

        async def fire() -> None:
            await asyncio.wait_for(cleanup.started.wait(), 5)
            scope.reschedule(asyncio.get_running_loop().time())
            await asyncio.sleep(0)
            cleanup.release.set()

        firer = asyncio.create_task(fire())
        try:
            yield
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                await firer


async def _boom() -> None:
    await asyncio.sleep(0)
    raise ValueError("boom")


def _recording_handler(messages: list[str]):
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: messages.append(context["message"]))


async def _assert_clean(messages: list[str]) -> None:
    gc.collect()
    await asyncio.sleep(0)
    current = asyncio.current_task()
    assert {t for t in asyncio.all_tasks() if not t.done()} == {current}
    assert messages == []
    assert current is not None and current.cancelling() == 0


async def test_parallel_cleanup_finishes_before_an_outer_timeout_propagates():
    messages: list[str] = []
    _recording_handler(messages)
    cleanup = _Cleanup()
    with pytest.raises(TimeoutError):
        async with _timeout_during(cleanup):
            await Executor().parallel(_boom, cleanup.body)
    await _assert_clean(messages)


async def test_run_many_fail_fast_cleanup_finishes_before_an_outer_timeout_propagates():
    reg = Registry()

    @reg.tool(parallel_safe=True)
    async def boom() -> None:
        """Fails at once."""
        await asyncio.sleep(0)
        raise ToolError("boom")

    cleanup = _Cleanup()

    @reg.tool(parallel_safe=True)
    async def slow() -> None:
        """Slow to clean up when cancelled."""
        await cleanup.body()

    ex = Executor(reg)
    messages: list[str] = []
    _recording_handler(messages)
    with pytest.raises(TimeoutError):
        async with _timeout_during(cleanup):
            await ex.run_many([ToolCall(name="boom"), ToolCall(name="slow")], on_error="fail_fast")
    await _assert_clean(messages)
    assert len(ex._settings.runs) == 0  # the batch's run was ended after its calls stopped


async def test_crossed_nested_kind_caps_deadlock_as_documented():
    reg = Registry()
    ex = Executor(reg, limits=RunLimits(per_kind={"A": 1, "B": 1}))
    both_holding = asyncio.Barrier(2)

    @reg.tool(kind="A", parallel_safe=True)
    async def a(ctx: ToolContext, nest: bool = True) -> str:
        """Holds A, then calls B."""
        if nest:
            await both_holding.wait()
            await ex.run("b", {"nest": False}, ctx)
        return "a"

    @reg.tool(kind="B", parallel_safe=True)
    async def b(ctx: ToolContext, nest: bool = True) -> str:
        """Holds B, then calls A."""
        if nest:
            await both_holding.wait()
            await ex.run("a", {"nest": False}, ctx)
        return "b"

    calls = [ToolCall(name="a"), ToolCall(name="b")]
    with pytest.raises(TimeoutError):  # A waits for B while B waits for A: a lock-order cycle
        async with asyncio.timeout(0.05):
            await ex.run_many(calls, ex.context(run_id="r"))


async def test_closing_as_completed_drains_before_an_outer_timeout_propagates():
    reg = Registry()

    @reg.tool(parallel_safe=True)
    async def fast() -> str:
        """Done at once."""
        return "fast"

    cleanup = _Cleanup()

    @reg.tool(parallel_safe=True)
    async def slow() -> None:
        """Slow to clean up when cancelled."""
        await cleanup.body()

    messages: list[str] = []
    _recording_handler(messages)
    ex = Executor(reg)
    with pytest.raises(TimeoutError):
        async with _timeout_during(cleanup):
            stream = ex.as_completed([ToolCall(name="fast"), ToolCall(name="slow")])
            async with aclosing(stream):
                async for result in stream:
                    assert result.value == "fast"
                    break
    await _assert_clean(messages)
