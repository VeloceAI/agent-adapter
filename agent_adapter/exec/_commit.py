"""Calls that change things: finish them when the caller is cancelled, and drop repeats."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ToolResult
from agent_adapter.exec._canon import digest

logger = logging.getLogger(__name__)

T = TypeVar("T")
RepeatKey = tuple[str, str, str]
_MAX_KEYS = 10_000


def changes_things(spec: ToolSpec) -> bool:
    """True when the tool declares an effect (`additive` or `destructive`)."""
    return spec.effects in ("additive", "destructive")


async def finish_once_started(work: Awaitable[T], started: Callable[[], bool]) -> tuple[T, bool]:
    """Await `work`; once `started()` is true a cancelled caller waits for it to end.

    Returns (value, True when the caller was cancelled meanwhile); the caller re-raises
    `asyncio.CancelledError` after that. Before `started()`, a cancellation cancels `work`.
    """
    task = asyncio.ensure_future(work)
    interrupted = False
    while True:
        try:
            return await asyncio.shield(task), interrupted
        except asyncio.CancelledError:
            if task.done():
                raise  # the work itself was cancelled, not its caller
            if not started():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            interrupted = True


def repeat_key(run_id: str, spec: ToolSpec, arguments: Any) -> RepeatKey | None:
    """The key a repeated call shares (run, tool, argument digest), or None if not deduped."""
    if not changes_things(spec) or spec.idempotent:
        return None
    try:
        return (run_id, spec.name, digest(arguments))
    except Exception:  # arguments no digest can encode run as fresh calls
        logger.debug("tool %s: arguments cannot be digested; repeats are not dropped", spec.name)
        return None


class RepeatWindow:
    """Returns the first result for a repeat of a finished or running call within `seconds`."""

    def __init__(self, seconds: float, clock: Callable[[], float]) -> None:
        self.seconds = seconds
        self._clock = clock
        self._done: OrderedDict[RepeatKey, tuple[float, ToolResult]] = OrderedDict()
        self._running: dict[RepeatKey, asyncio.Future[ToolResult]] = {}

    def _prune(self, now: float) -> None:
        while self._done:
            key, (at, _) = next(iter(self._done.items()))
            if now - at <= self.seconds and len(self._done) <= _MAX_KEYS:
                break
            del self._done[key]

    async def run(
        self, key: RepeatKey, work: Callable[[], Awaitable[tuple[ToolResult, Any]]]
    ) -> tuple[ToolResult, Any, ToolResult | None]:
        """Run `work` once per key; returns (result, stored, earlier result for a repeat)."""
        self._prune(self._clock())
        done = self._done.get(key)
        if done is not None:
            return done[1], None, done[1]
        running = self._running.get(key)
        if running is not None:
            try:
                earlier = await asyncio.shield(running)
            except Exception:
                earlier = None
            if earlier is not None and earlier.ok:
                return earlier, None, earlier
        return await self._first(key, work)

    async def _first(
        self, key: RepeatKey, work: Callable[[], Awaitable[tuple[ToolResult, Any]]]
    ) -> tuple[ToolResult, Any, ToolResult | None]:
        future: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()
        self._running[key] = future
        try:
            result, stored = await work()
        except BaseException as exc:
            future.set_exception(exc if isinstance(exc, Exception) else RuntimeError("stopped"))
            future.exception()  # marks it retrieved when nobody waits
            raise
        else:
            future.set_result(result)
            if result.ok:
                self._done[key] = (self._clock(), result)
            return result, stored, None
        finally:
            self._running.pop(key, None)
