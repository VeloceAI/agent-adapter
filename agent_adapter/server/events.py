"""Run event streams: the executor's events kept per run for SSE subscribers."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event

logger = logging.getLogger(__name__)


@runtime_checkable
class EventStream(EventSink, Protocol):
    """An event sink that replays and follows the events of one run."""

    def subscribe(self, run_id: str, *, after: int = 0) -> AsyncIterator[tuple[int, Event]]:
        """`(seq, event)` for every event of `run_id` with `seq > after`, then new ones as
        they arrive; `seq` starts at 1 per run."""
        ...


class _Waiter:
    """Wakes one subscriber on its own loop; safe to call from any thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.flag = asyncio.Event()

    def wake(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.flag.set)
        except RuntimeError:  # the subscriber's loop is closed; it will never read again
            logger.debug("event subscriber's loop is closed")


class MemoryEventStream:
    """In-memory `EventStream`: the last `max_events_per_run` events of the last `max_runs` runs.

    Events without a `run_id` are dropped. Subscribing never creates a run, so readers
    cannot evict the buffers of other runs.
    """

    def __init__(self, *, max_events_per_run: int = 1000, max_runs: int = 1000) -> None:
        if max_events_per_run < 1 or max_runs < 1:
            raise ConfigError("MemoryEventStream limits must be >= 1")
        self._max_events = max_events_per_run
        self._max_runs = max_runs
        self._runs: OrderedDict[str, deque[tuple[int, Event]]] = OrderedDict()
        self._next: dict[str, int] = {}
        self._waiters: dict[str, set[_Waiter]] = {}
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        """Store `event` under its run and wake that run's subscribers; never raises."""
        try:
            self._store(event)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "event stream failed to keep %s", event.type)

    def _store(self, event: Event) -> None:
        if not event.run_id:
            return
        with self._lock:
            run = self._runs.get(event.run_id)
            if run is None:
                run = self._runs[event.run_id] = deque(maxlen=self._max_events)
                self._evict(keep=event.run_id)
            self._runs.move_to_end(event.run_id)
            seq = self._next.get(event.run_id, 1)
            self._next[event.run_id] = seq + 1
            run.append((seq, event))
            waiters = list(self._waiters.get(event.run_id, ()))
        for waiter in waiters:
            waiter.wake()

    def _evict(self, *, keep: str) -> None:
        """Drop the least recent runs beyond `max_runs`: unfollowed ones first, never `keep`."""
        while len(self._runs) > self._max_runs:
            idle = (rid for rid in self._runs if rid != keep and rid not in self._waiters)
            victim = next(idle, None) or next(rid for rid in self._runs if rid != keep)
            del self._runs[victim]
            if victim not in self._waiters:
                self._next.pop(victim, None)  # a followed run keeps counting from where it was

    def _since(self, run_id: str, after: int) -> list[tuple[int, Event]]:
        with self._lock:
            return [item for item in self._runs.get(run_id, ()) if item[0] > after]

    async def subscribe(self, run_id: str, *, after: int = 0) -> AsyncIterator[tuple[int, Event]]:
        """Replay the kept events after `after`, then follow the run until closed."""
        waiter = _Waiter(asyncio.get_running_loop())
        with self._lock:
            self._waiters.setdefault(run_id, set()).add(waiter)
        try:
            last = after
            while True:
                waiter.flag.clear()
                batch = self._since(run_id, last)
                if not batch:
                    await waiter.flag.wait()
                    continue
                for seq, event in batch:
                    last = seq
                    yield seq, event
        finally:
            with self._lock:
                waiters = self._waiters.get(run_id)
                if waiters is not None:
                    waiters.discard(waiter)
                    if not waiters:
                        del self._waiters[run_id]
                        if run_id not in self._runs:
                            self._next.pop(run_id, None)
