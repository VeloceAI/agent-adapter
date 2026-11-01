"""The event channel of one AG-UI run: a thread-safe queue the SSE response drains."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

MAX_EVENTS = 10_000
_CLOSE = object()


class EventChannel:
    """AG-UI events in emission order; past `max_events`, droppable (trace) events are dropped.

    Must be created inside the event loop that drains it; `put` may be called from any thread.
    """

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._max = max_events
        self._lock = threading.Lock()
        self._count = 0
        self._closed = False
        self.dropped = 0

    def put(self, event: Any, *, droppable: bool = False) -> None:
        """Queue `event`; after `close`, or for a droppable event over the bound, drop it."""
        with self._lock:
            if self._closed:
                logger.debug("ag_ui: event after the run closed dropped (%s)", type(event).__name__)
                return
            if droppable and self._count >= self._max:
                self.dropped += 1
                if self.dropped == 1:
                    logger.warning("ag_ui: run reached %d events; dropping trace events", self._max)
                return
            self._count += 1
            self._send(event)

    def close(self) -> None:
        """End the stream after the events already queued."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._send(_CLOSE)

    def _send(self, item: Any) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._queue.put_nowait(item)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def __aiter__(self) -> AsyncIterator[Any]:
        while True:
            item = await self._queue.get()
            if item is _CLOSE:
                return
            yield item
