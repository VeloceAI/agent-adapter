"""LoopRunner: one long-lived event loop in a daemon thread, for sync code in warm processes."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextvars
import inspect
import logging
import os
import threading
import weakref
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, TypeVar

from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.runtime._fork import ForkSafeLock
from agent_adapter.runtime.loop_local import LoopLocal
from agent_adapter.runtime.sync import CALLER_GRACE_S, _awaitable, _discard, _positive_number

T = TypeVar("T")

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 30.0  # close(): how long in-flight work may finish before it is cancelled
ATEXIT_TIMEOUT_S = 5.0

_RUNNERS: weakref.WeakSet[LoopRunner] = weakref.WeakSet()
_RUNNERS_LOCK = ForkSafeLock()
# A loop inherited by a forked child has no thread: it can neither run nor be closed.
_FORK_ORPHANS: list[asyncio.AbstractEventLoop] = []
_warned_blocking = False


class LoopRunner:
    """Runs async work from sync code on one event loop that lives as long as the process.

    The loop starts in a daemon thread on the first `run()`. `LoopLocal` values (httpx or
    Redis clients) stay warm across calls. A forked child starts its own loop; `close()`, a
    `with` block or interpreter exit closes the `LoopLocal` values and stops the loop.
    """

    def __init__(self, *, name: str = "agent_adapter-loop") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        with _RUNNERS_LOCK:
            _RUNNERS.add(self)

    @property
    def closed(self) -> bool:
        """True once `close()` has begun."""
        return self._closed

    @property
    def running(self) -> bool:
        """True while the loop thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def run(
        self, work: Awaitable[T] | Callable[[], Awaitable[T]], *, timeout: float | None = None
    ) -> T:
        """Run `work` on the runner's loop and block this thread until it finishes.

        Args:
            work: An awaitable, or a zero-argument callable returning one (called in the loop).
            timeout: Seconds before the work is cancelled. The caller also stops waiting
                `CALLER_GRACE_S` later, even if the work blocks the loop.

        Raises:
            TimeoutError: the work ran past `timeout`.
            ToolUsageError: bad `work` or `timeout`, the runner is closed (or closed while
                the work ran), or `run()` was called from the runner's own loop.
        """
        _check(work, timeout)
        if threading.current_thread() is self._thread:
            _discard(work)
            raise ToolUsageError("LoopRunner.run() inside its own loop would deadlock: await")
        _warn_if_blocking_a_loop()
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        tasks: list[asyncio.Task[T]] = []
        context = contextvars.copy_context()

        def start(loop: asyncio.AbstractEventLoop) -> None:
            if not future.set_running_or_notify_cancel():
                _discard(work)
                return
            task = loop.create_task(_bounded(work, timeout), context=context)
            tasks.append(task)
            task.add_done_callback(lambda done: _settle(done, future, work))

        loop = self._schedule(start, work)
        wait = None if timeout is None else timeout + CALLER_GRACE_S
        try:
            return future.result(timeout=wait)
        except concurrent.futures.TimeoutError:
            if future.done():
                return future.result()  # finished (or raised TimeoutError) as the wait ended
            _cancel(loop, future, tasks)
            logger.error("LoopRunner: work blocked the loop past %ss; cancelling it", wait)
            raise TimeoutError(f"LoopRunner: work did not finish within {wait:g}s") from None
        except BaseException:
            if not future.done():  # KeyboardInterrupt in the caller: stop the work too
                _cancel(loop, future, tasks)
            raise

    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> None:
        """Let in-flight work finish for `timeout` s, cancel the rest, close `LoopLocal`
        values and stop the loop; safe to call twice.

        Raises:
            ToolUsageError: called from the runner's own loop, or a bad `timeout`.
        """
        if not _positive_number(timeout):
            raise ToolUsageError(f"LoopRunner.close timeout must be > 0, got {timeout!r}")
        if threading.current_thread() is self._thread:
            raise ToolUsageError("LoopRunner.close() from its own loop: call it from outside")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is None or thread is None or not thread.is_alive():
            return
        shutdown = asyncio.run_coroutine_threadsafe(_shutdown(timeout), loop)
        try:
            shutdown.result(timeout=timeout + CALLER_GRACE_S)
        except concurrent.futures.TimeoutError:
            shutdown.cancel()
            logger.warning("LoopRunner %r: shutdown did not finish; stopping the loop", self._name)
        finally:
            _stop(loop, thread, self._name)

    def __enter__(self) -> LoopRunner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _schedule(
        self, start: Callable[[asyncio.AbstractEventLoop], None], work: object
    ) -> asyncio.AbstractEventLoop:
        """Queue `start` on the loop; under the lock, so a concurrent `close()` runs after it."""
        with self._lock:
            if self._closed:
                _discard(work)
                raise ToolUsageError(f"LoopRunner {self._name!r} is closed")
            loop = self._ensure_loop()
            loop.call_soon_threadsafe(start, loop)
            return loop

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """The running loop, starting a fresh one if there is none. Caller holds _lock."""
        if self._loop is not None and self._thread is not None and self._thread.is_alive():
            return self._loop
        if self._loop is not None:
            logger.warning("LoopRunner %r: its loop thread died; starting a new loop", self._name)
            self._loop.close()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_serve, args=(loop, self._name), name=self._name)
        thread.daemon = True
        thread.start()
        self._loop, self._thread = loop, thread
        logger.debug("LoopRunner %r started its loop", self._name)
        return loop

    def _forget_after_fork(self) -> None:
        """In a forked child: drop the parent's loop and thread; the next `run()` starts anew."""
        self._lock = threading.Lock()
        if self._loop is not None:
            _FORK_ORPHANS.append(self._loop)
        self._loop, self._thread = None, None


def _warn_if_blocking_a_loop() -> None:
    """Warn once when a sync call blocks a thread that runs its own event loop."""
    global _warned_blocking
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    if not _warned_blocking:
        _warned_blocking = True
        logger.warning(
            "LoopRunner.run() called from a running event loop: it blocks that loop until "
            "the work ends (and deadlocks if the work needs it); await the async method"
        )


def _check(work: object, timeout: float | None) -> None:
    if not (inspect.isawaitable(work) or callable(work)):
        raise ToolUsageError(f"LoopRunner.run needs an awaitable or a callable, got {work!r}")
    if timeout is not None and not _positive_number(timeout):
        _discard(work)
        raise ToolUsageError(f"LoopRunner.run timeout must be a positive number, got {timeout!r}")


def _serve(loop: asyncio.AbstractEventLoop, name: str) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    except BaseException as exc:
        # The thread ends either way; the next run() sees it dead and starts a new loop.
        log_failure(logger, logging.ERROR, exc, "LoopRunner %r: its loop crashed", name)


async def _bounded(work: Awaitable[T] | Callable[[], Awaitable[T]], timeout: float | None) -> T:
    async with asyncio.timeout(timeout):
        return await _awaitable(work)


def _settle(task: asyncio.Task[T], future: concurrent.futures.Future[T], work: object) -> None:
    """Copy a finished task's outcome to the caller's future."""
    if future.done():
        return
    if task.cancelled():
        _discard(work)  # a task cancelled before its first step never awaited `work`
        future.set_exception(ToolUsageError("the LoopRunner was closed while this work ran"))
        return
    error = task.exception()
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(task.result())


def _cancel(
    loop: asyncio.AbstractEventLoop,
    future: concurrent.futures.Future[Any],
    tasks: list[asyncio.Task[Any]],
) -> None:
    """Cancel the work: before it started, or on the loop once it runs."""
    if future.cancel():
        return

    def cancel_all() -> None:
        for task in tasks:
            task.cancel()

    try:
        loop.call_soon_threadsafe(cancel_all)
    except RuntimeError:  # the loop closed in the meantime: nothing left to cancel
        logger.debug("LoopRunner: loop already closed; work not cancelled")


async def _shutdown(timeout: float) -> None:
    """On the loop: wait for in-flight work, cancel what is left, close loop-local values."""
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current]
    if pending:
        _, late = await asyncio.wait(pending, timeout=timeout)
        for task in late:
            task.cancel()
        await asyncio.gather(*late, return_exceptions=True)
        if late:
            logger.warning("LoopRunner.close cancelled %d unfinished task(s)", len(late))
    await LoopLocal.aclose_all()
    await loop.shutdown_asyncgens()
    try:
        await loop.shutdown_default_executor()
    except RuntimeError as exc:  # at interpreter exit no new thread may start
        logger.debug("LoopRunner: default executor not shut down: %s", exc)


def _stop(loop: asyncio.AbstractEventLoop, thread: threading.Thread, name: str) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=CALLER_GRACE_S)  # a loop blocked in sync code must not hold up exit
    if thread.is_alive():
        logger.warning("LoopRunner %r: loop thread did not stop in %.0f s", name, CALLER_GRACE_S)
        return
    loop.close()
    logger.debug("LoopRunner %r closed", name)


def _close_at_exit() -> None:
    with _RUNNERS_LOCK:
        runners = list(_RUNNERS)
    for runner in runners:
        try:
            runner.close(timeout=ATEXIT_TIMEOUT_S)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "closing a LoopRunner at exit failed")


def _after_fork_in_child() -> None:
    for runner in list(_RUNNERS):
        runner._forget_after_fork()


atexit.register(_close_at_exit)
if hasattr(os, "register_at_fork"):  # POSIX only
    os.register_at_fork(after_in_child=_after_fork_in_child)
