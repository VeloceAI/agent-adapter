"""Run async work from sync code (Flask, Celery, scripts, notebooks) without nest_asyncio."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.runtime.loop_local import LoopLocal

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Extra seconds the caller waits for a worker loop after `timeout`, in case the work blocks
# the loop synchronously and asyncio.timeout cannot fire.
CALLER_GRACE_S = 5.0


def run_sync(work: Awaitable[T] | Callable[[], Awaitable[T]], *, timeout: float | None = None) -> T:
    """Run `work` to completion on a fresh event loop and return its result.

    Without a running loop in this thread it uses `asyncio.run`. Inside a running loop
    (Jupyter, an async framework) it runs that loop in a one-shot worker thread with a copy
    of the current contextvars and blocks this thread until done. Loop-local values
    (`LoopLocal`) are closed before the loop ends.

    Args:
        work: An awaitable, or a zero-argument callable returning one (called inside the loop).
        timeout: Seconds before the work is cancelled. In the worker-thread case the caller
            also stops waiting `CALLER_GRACE_S` later, even if the work blocks its loop.

    Raises:
        TimeoutError: the work ran past `timeout`.
        ToolUsageError: `work` is neither awaitable nor callable, or `timeout` is not a
            positive number.
    """
    if not (inspect.isawaitable(work) or callable(work)):
        raise ToolUsageError(f"run_sync needs an awaitable or a callable, got {work!r}")
    if timeout is not None and not _positive_number(timeout):
        _discard(work)
        raise ToolUsageError(f"run_sync timeout must be a positive number, got {timeout!r}")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_main(work, timeout))
    return _in_worker_thread(work, timeout)


def _in_worker_thread(work: Awaitable[T] | Callable[[], Awaitable[T]], timeout: float | None) -> T:
    context = contextvars.copy_context()
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="agent_adapter-run_sync"
    )
    try:
        future = pool.submit(context.run, asyncio.run, _main(work, timeout))
        wait = None if timeout is None else timeout + CALLER_GRACE_S
        try:
            return future.result(timeout=wait)
        except concurrent.futures.TimeoutError:
            if future.done():
                # The work finished (or itself raised TimeoutError) after the wait ended.
                return future.result()
            logger.error("run_sync: work blocked its loop past %ss; abandoning the thread", wait)
            raise TimeoutError(f"run_sync: work did not finish within {wait:g}s") from None
    finally:
        pool.shutdown(wait=False)  # never block the caller on a stuck worker


def _positive_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and value > 0


async def _main(work: Awaitable[T] | Callable[[], Awaitable[T]], timeout: float | None) -> T:
    try:
        async with asyncio.timeout(timeout):
            return await _awaitable(work)
    finally:
        await LoopLocal.aclose_all()


def _awaitable(work: Awaitable[T] | Callable[[], Awaitable[T]]) -> Awaitable[T]:
    if inspect.isawaitable(work):
        return work
    result: Any = cast(Callable[[], Any], work)()
    if not inspect.isawaitable(result):
        raise ToolUsageError(f"run_sync: {work!r} returned {type(result).__name__}, not awaitable")
    return cast(Awaitable[T], result)


def _discard(work: object) -> None:
    """Close an unawaited coroutine so it does not warn."""
    if inspect.iscoroutine(work):
        work.close()
