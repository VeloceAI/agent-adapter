"""Batches that never leak ExceptionGroup, and `parallel` for non-tool work."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Any, Literal, TypeVar

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import control_flow_member
from agent_adapter.core.errors import ToolTimeout, ToolUsageError, qualified_name
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.exec._settings import _Settings
from agent_adapter.exec.pipeline import _execute
from agent_adapter.exec.retry import Backoff, _attempt

logger = logging.getLogger(__name__)

T = TypeVar("T")

OnError = Literal["collect", "fail_fast"]
ON_ERROR: tuple[OnError, ...] = ("collect", "fail_fast")


def _check_on_error(on_error: object) -> OnError:
    """`on_error` if valid.

    Raises:
        ToolUsageError: not "collect" or "fail_fast".
    """
    if on_error not in ON_ERROR:
        raise ToolUsageError(f"on_error must be one of {ON_ERROR}, got {on_error!r}")
    return on_error


def _check_concurrency(concurrency: object, default: int) -> int:
    """`concurrency`, or `default` when None.

    Raises:
        ToolUsageError: not an int >= 1.
    """
    if concurrency is None:
        return default
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ToolUsageError(f"concurrency must be an int >= 1, got {concurrency!r}")
    return concurrency


async def _gather_unwrapped(
    factories: Sequence[Callable[[], Awaitable[T]]], *, concurrency: int
) -> list[T]:
    """Run factories concurrently in input order; the first escaping exception cancels the rest
    and is re-raised as itself, never inside an ExceptionGroup."""
    # Plain tasks, not a TaskGroup: on Python 3.12 a TaskGroup whose child fails while the
    # parent waits leaks a cancel request, and the caller's own cancellation (an outer
    # asyncio.timeout) is lost behind the children's errors.
    semaphore = asyncio.Semaphore(concurrency)

    async def one(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    tasks = [asyncio.create_task(one(factory)) for factory in factories]
    try:
        pending: set[asyncio.Task[T]] = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            failed = [task for task in tasks if task in done and _failed(task)]
            if failed:
                raise _escaping(failed[0])
        return [task.result() for task in tasks]
    finally:
        if await _drain(tasks):
            raise asyncio.CancelledError  # the caller was cancelled during cleanup: that wins


async def _drain(tasks: Sequence[asyncio.Task[Any]]) -> bool:
    """Cancel `tasks` and wait until every one has finished, retrieving all their exceptions.

    Returns True when the caller was cancelled while waiting; the wait goes on regardless, so
    no task outlives the batch (still holding a run slot after `end_run`).
    """
    for task in tasks:
        task.cancel()
    if not tasks:
        return False
    gathered = asyncio.gather(*tasks, return_exceptions=True)  # siblings' late errors lost the race
    interrupted = False
    while not gathered.done():
        try:
            await asyncio.shield(gathered)
        except asyncio.CancelledError:
            interrupted = True
    return interrupted


def _failed(task: asyncio.Task[Any]) -> bool:
    return task.cancelled() or task.exception() is not None


def _escaping(task: asyncio.Task[Any]) -> BaseException:
    """What a failed task re-raises: its error, or CancelledError raised by the work itself."""
    if task.cancelled():
        return asyncio.CancelledError()
    error = task.exception()
    assert error is not None
    return control_flow_member(error) or error


async def _run_batch(
    settings: _Settings,
    calls: Sequence[ToolCall],
    ctx: ToolContext,
    *,
    concurrency: int,
    on_error: OnError,
    registry: Registry | None,
) -> list[ToolResult]:
    """Run tool calls concurrently in one run; `fail_fast` raises the first failure."""
    _check_unique_ids(calls)
    _log_serial(calls, registry)

    def factory(call: ToolCall) -> Callable[[], Awaitable[ToolResult]]:
        async def run() -> ToolResult:
            result = await _execute(settings, call, ctx, registry=registry)
            if on_error == "fail_fast" and not result.ok:
                result.unwrap()  # raises; the batch cancels the siblings
            return result

        return run

    return await _gather_unwrapped([factory(c) for c in calls], concurrency=concurrency)


def _check_unique_ids(calls: Sequence[ToolCall]) -> None:
    """One batch may not repeat a call id: both would be the same call.

    Raises:
        ToolUsageError: a non-empty call id appears twice.
    """
    seen: set[str] = set()
    for call in calls:
        if call.id and call.id in seen:
            raise ToolUsageError(f"call id {call.id!r} appears twice in one batch")
        seen.add(call.id)


def _log_serial(calls: Sequence[ToolCall], registry: Registry | None) -> None:
    if len(calls) < 2 or registry is None:
        return
    specs = [registry.get(c.name) for c in calls if c.name in registry]
    if len(specs) == len(calls) and not any(spec.parallel_safe for spec in specs):
        logger.debug("batch of %d non-parallel-safe calls runs one at a time", len(calls))


async def _stream_completed(
    settings: _Settings,
    calls: Sequence[ToolCall],
    ctx: ToolContext,
    *,
    concurrency: int,
    registry: Registry | None,
) -> AsyncGenerator[ToolResult, None]:
    """Yield results in completion order; closing the stream cancels unfinished calls."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(call: ToolCall) -> ToolResult:
        async with semaphore:
            return await _execute(settings, call, ctx, registry=registry)

    tasks = [asyncio.create_task(one(call)) for call in calls]
    try:
        for finished in asyncio.as_completed(tasks):
            yield await finished
    finally:
        if await _drain(tasks):
            raise asyncio.CancelledError  # the consumer was cancelled during cleanup


async def _run_parallel(
    work: Sequence[Callable[[], Awaitable[Any]] | Awaitable[Any]],
    *,
    concurrency: int,
    timeout: float | None,
    retries: int,
    on_error: OnError,
    backoff: Backoff,
    rng: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> list[Any]:
    """Run non-tool work with the executor's timeout, retry and cancellation rules."""
    try:
        _check_parallel(work, timeout, retries)
    except ToolUsageError:
        _close_unstarted(work)
        raise

    def factory(index: int, item: Callable[[], Awaitable[Any]] | Awaitable[Any]) -> Any:
        async def run() -> Any:
            outcome = await _attempt(
                _as_factory(index, item),
                retries=retries,
                timeout=timeout,
                backoff=backoff,
                rng=rng,
                sleep=sleep,
                on_timeout=lambda: ToolTimeout(f"parallel[{index}]", timeout or 0.0),
                on_retry=lambda n, delay, exc: logger.warning(
                    "parallel[%d] attempt %d failed (%s); retrying in %.3gs",
                    index,
                    n,
                    qualified_name(type(exc)),
                    delay,
                ),
            )
            if outcome.error is None:
                return outcome.value
            if on_error == "fail_fast":
                raise outcome.error
            logger.info("parallel[%d] failed: %s", index, qualified_name(type(outcome.error)))
            return outcome.error

        return run

    try:
        return await _gather_unwrapped(
            [factory(i, item) for i, item in enumerate(work)], concurrency=concurrency
        )
    finally:
        _close_unstarted(work)  # items of cancelled siblings never started


def _check_parallel(work: Sequence[Any], timeout: float | None, retries: int) -> None:
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ToolUsageError(f"retries must be an int >= 0, got {retries!r}")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ToolUsageError(f"timeout must be a positive number or None, got {timeout!r}")
    for index, item in enumerate(work):
        if inspect.isawaitable(item):
            if retries:
                raise ToolUsageError(
                    f"parallel[{index}] is an awaitable and cannot be retried; "
                    "pass a zero-argument coroutine function instead"
                )
        elif not callable(item):
            raise ToolUsageError(
                f"parallel[{index}] must be awaitable or a zero-argument callable, got {item!r}"
            )


def _as_factory(index: int, item: Callable[[], Awaitable[T]] | Awaitable[T]) -> Callable[[], Any]:
    if inspect.isawaitable(item):
        return lambda: item

    async def call() -> T:
        result = item()
        if not inspect.isawaitable(result):
            raise ToolUsageError(
                f"parallel[{index}] returned {type(result).__name__}, not an awaitable"
            )
        return await result

    return call


def _close_unstarted(work: Sequence[Any]) -> None:
    """Close coroutines that never ran (cancelled or rejected) so they do not warn."""
    for item in work:
        if inspect.iscoroutine(item) and inspect.getcoroutinestate(item) == inspect.CORO_CREATED:
            item.close()
