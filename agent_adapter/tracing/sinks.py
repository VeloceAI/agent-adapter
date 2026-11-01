"""Event sinks: a bounded in-memory log, standard logging, a callback, a fan-out and a hub."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event
from agent_adapter.runtime._fork import ForkSafeLock
from agent_adapter.tracing.events import emit

logger = logging.getLogger(__name__)

# Event data keys that may carry prompts, arguments or results: logged or exported only
# with capture_content=True (and logged only at DEBUG).
CONTENT_KEYS = frozenset({"arguments", "result", "input_messages", "output_messages", "updates"})


class EventLog:
    """Keeps the latest `max_events` events in memory, oldest dropped first."""

    def __init__(self, *, max_events: int = 10_000) -> None:
        if max_events < 1:
            raise ConfigError(f"max_events must be >= 1, got {max_events}")
        self.max_events = max_events
        self.events: list[Event] = []
        self._lock = threading.Lock()

    def emit(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)
            overflow = len(self.events) - self.max_events
            if overflow > 0:
                del self.events[:overflow]

    def of_type(self, prefix: str) -> list[Event]:
        """Events whose type starts with `prefix`, in emission order."""
        with self._lock:
            return [event for event in self.events if event.type.startswith(prefix)]


class LoggingSink:
    """Logs each event at `level`; content keys only with `capture_content` and at DEBUG."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.DEBUG,
        capture_content: bool = False,
    ) -> None:
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._level = level
        self._content = capture_content and level <= logging.DEBUG

    @property
    def logger(self) -> logging.Logger:
        """The logger events are written to."""
        return self._logger

    @property
    def level(self) -> int:
        """The level events are logged at."""
        return self._level

    def emit(self, event: Event) -> None:
        if not self._logger.isEnabledFor(self._level):
            return
        try:
            data = dict(event.data)
            omitted = [] if self._content else sorted(CONTENT_KEYS & data.keys())
            for key in omitted:
                del data[key]
            suffix = f" omitted={omitted}" if omitted else ""
            # Formatted here, not by the handler, so a bad repr is caught below.
            text = (
                f"event {event.type} run={event.run_id} call={event.call_id} "
                f"parent={event.parent_id} data={data!r}{suffix}"
            )
            self._logger.log(self._level, "%s", text)
        except Exception as exc:
            # A sink never raises into the traced call.
            log_failure(logger, logging.WARNING, exc, "logging sink failed for %s", event.type)


class CallbackEventSink:
    """Calls `fn(event)`; a failing callback is logged, never raised."""

    def __init__(self, fn: Callable[[Event], None]) -> None:
        self._fn = fn

    def emit(self, event: Event) -> None:
        try:
            self._fn(event)
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "event callback failed for %s (run %s)",
                event.type,
                event.run_id,
            )


class FanoutSink:
    """Sends each event to every sink in order; a failing sink does not stop the others."""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """The sinks each event goes to, in order."""
        return self._sinks

    def emit(self, event: Event) -> None:
        for sink in self._sinks:
            emit(sink, event)


class SinkHub:
    """An executor's stable sink: `base` (the constructor's sink), then every added sink.

    Contexts and models keep the hub itself, so `add` and `remove` reach them all. Members are
    a copy-on-write tuple: `emit` never locks.
    """

    def __init__(self, base: EventSink | None = None, extra: tuple[EventSink, ...] = ()) -> None:
        self._base = base
        self._extra = extra
        self._lock = ForkSafeLock()

    @property
    def base(self) -> EventSink | None:
        """The constructor's sink (fixed for the hub's life), or None."""
        return self._base

    @property
    def extra(self) -> tuple[EventSink, ...]:
        """The sinks added after construction, in order."""
        return self._extra

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """Every member, base first; empty when there is nothing to send to."""
        return self._extra if self._base is None else (self._base, *self._extra)

    def add(self, sink: EventSink) -> bool:
        """Add `sink` once; False when it is already a member."""
        with self._lock:
            if sink is self._base or any(sink is member for member in self._extra):
                return False
            self._extra = (*self._extra, sink)
            return True

    def remove(self, sink: EventSink) -> bool:
        """Remove an added `sink`; False when it was not added (the base stays)."""
        with self._lock:
            kept = tuple(member for member in self._extra if member is not sink)
            if len(kept) == len(self._extra):
                return False
            self._extra = kept
            return True

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            emit(sink, event)

    def __repr__(self) -> str:
        return f"SinkHub({', '.join(type(s).__name__ for s in self.sinks)})"


def members(sink: EventSink | None) -> Iterator[EventSink]:
    """`sink` and, through hubs and fan-outs, every sink inside it, each once (depth first)."""
    seen: set[int] = set()
    todo = [] if sink is None else [sink]
    while todo:
        current = todo.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, SinkHub | FanoutSink):
            todo.extend(reversed(current.sinks))


def first_sink(own: EventSink | None, fallback: EventSink | None) -> EventSink | None:
    """`own`, else `fallback`; a hub with no members counts as no sink, so events still flow."""
    if own is None or (isinstance(own, SinkHub) and not own.sinks):
        return fallback
    return own
