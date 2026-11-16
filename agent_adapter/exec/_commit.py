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


REPEAT_NOTE = "already done earlier in this window; not run again"


def changes_things(spec: ToolSpec) -> bool:
    """True for an `additive` or `destructive` effect; with no `effects`, for `confirm` or
    `once_per_run` tools."""
    if spec.effects is not None:
        return spec.effects in ("additive", "destructive")
    return bool(spec.confirm) or spec.once_per_run


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
                if task.cancelled() or task.exception() is not None:
                    raise  # the work itself was cancelled or failed as its caller was cancelled
                return task.result(), True  # it finished as its caller was cancelled
            if not started():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            interrupted = True


def repeat_key(run_id: str, spec: ToolSpec, arguments: Any) -> RepeatKey | None:
    """The key a repeated call shares (run, tool, validated-argument digest), or None."""
    if not changes_things(spec) or spec.idempotent:
        return None
    try:
        validated = spec.validate_args(arguments)
        dumped = validated.model_dump(mode="json", by_alias=True, exclude_unset=True)
        return (run_id, spec.name, digest(dumped))
    except Exception:  # invalid or undigestable arguments run as fresh calls
        logger.debug("tool %s: no repeat key for these arguments; not deduped", spec.name)
        return None


def mark_repeat(value: Any) -> Any:
    """`value` with `REPEAT_NOTE`, so the model can tell a dropped repeat from a new run."""
    if value is None:
        return REPEAT_NOTE
    if isinstance(value, str):
        return f"{value} ({REPEAT_NOTE})"
    if isinstance(value, list | tuple):
        return [REPEAT_NOTE, *value]
    return [REPEAT_NOTE, value]


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
        while True:
            self._prune(self._clock())
            done = self._done.get(key)
            if done is not None:
                return done[1], None, done[1]
            running = self._running.get(key)
            if running is None:
                return await self._first(key, work)
            try:
                earlier = await asyncio.shield(running)
            except Exception:
                continue  # the first failed: one waiter becomes the new first
            if earlier.ok:
                return earlier, None, earlier

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
            if self._running.get(key) is future:
                del self._running[key]
