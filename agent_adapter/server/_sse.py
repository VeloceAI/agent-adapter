"""Server-sent events for one run: frames, keep-alives and the rules for closing the stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable, Collection
from typing import Any

from agent_adapter.core.types import Event, to_jsonable
from agent_adapter.exec.control import is_control_flow
from agent_adapter.tracing.events import EventType

logger = logging.getLogger(__name__)

KEEP_ALIVE = b": keep-alive\n\n"
END = b"event: end\ndata: {}\n\n"
_DONE = object()
_QUEUE = 1000


def frame(seq: int, event: Event, run_key: str) -> bytes | None:
    """`id/event/data` for one event, run id shown as the client's key; None if unencodable."""
    data = event.to_dict()
    data["run_id"] = run_key
    name = "".join(ch for ch in event.type if ch not in "\r\n")  # no frame injection
    try:
        text = json.dumps(to_jsonable(data), ensure_ascii=False, allow_nan=False)
    except ValueError:
        logger.warning("skipping an SSE event that is not JSON (%s)", name)
        return None
    return f"id: {seq}\nevent: {name}\ndata: {text}\n\n".encode("utf-8", "replace")


async def stream(
    source: AsyncIterator[tuple[int, Event]],
    *,
    run_key: str,
    close_on: Collection[str],
    keep_alive_s: float,
    idle_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[bytes]:
    """SSE bytes: each event, `: keep-alive` every `keep_alive_s`, `event: end` then close after
    a `close_on` event, `run.finished`, the source ending, or `idle_s` without events."""
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE)
    pump = asyncio.ensure_future(_pump(source, queue))
    try:
        last = clock()
        while True:
            left = idle_s - (clock() - last)
            if left <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), min(keep_alive_s, left))
            except TimeoutError:
                if clock() - last < idle_s:
                    yield KEEP_ALIVE
                continue
            if item is _DONE:
                break
            seq, event = item
            last = clock()
            encoded = frame(seq, event, run_key)
            if encoded is not None:
                yield encoded
            if event.type in close_on or event.type == EventType.RUN_FINISHED:
                break
        yield END
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump


async def _pump(source: AsyncIterator[tuple[int, Event]], queue: asyncio.Queue[Any]) -> None:
    try:
        async for item in source:
            await queue.put(item)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        logger.warning("event stream subscription failed: %s", type(exc).__name__)
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()
    await queue.put(_DONE)


class Streams:
    """Open event streams per principal and in total; a slot is taken until its stream ends."""

    def __init__(self) -> None:
        self._open: dict[str, int] = {}
        self._total = 0
        self._lock = threading.Lock()

    def acquire(self, owner: str, *, per_principal: int, total: int) -> Callable[[], None] | None:
        """A release function for a new slot, or None when a cap is reached."""
        with self._lock:
            if self._total >= total or self._open.get(owner, 0) >= per_principal:
                return None
            self._open[owner] = self._open.get(owner, 0) + 1
            self._total += 1

        def release() -> None:
            with self._lock:
                self._total -= 1
                left = self._open[owner] - 1
                if left:
                    self._open[owner] = left
                else:
                    del self._open[owner]

        return release


class Guarded:
    """`inner` that calls `release` exactly once: at its end, on close, or when collected."""

    def __init__(self, inner: AsyncIterator[bytes], release: Callable[[], None]) -> None:
        self._inner = inner
        self._release = weakref.finalize(self, release)

    def __aiter__(self) -> Guarded:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await anext(self._inner)
        except BaseException:
            self._release()
            raise

    async def aclose(self) -> None:
        try:
            aclose = getattr(self._inner, "aclose", None)
            if aclose is not None:
                await aclose()
        finally:
            self._release()
