"""Collect each case run's events; read the tools it called and what it cost."""

from __future__ import annotations

import logging
import math
import threading

from agent_adapter.core.ports import LedgerPort
from agent_adapter.core.types import Event
from agent_adapter.tracing.events import EventType

logger = logging.getLogger(__name__)

MAX_EVENTS = 10_000  # per case run, to bound memory
_COST_KEYS = {
    EventType.LLM_FINISHED: "cost_usd",
    EventType.EMBEDDING_FINISHED: "cost_usd",
    EventType.RERANK_FINISHED: "cost_usd",
    EventType.CONTEXT_COMPRESSED: "summary_cost_usd",
}


class Collector:
    """An event sink that keeps the events of the runs it was told to open."""

    def __init__(self) -> None:
        self._runs: dict[str, list[Event]] = {}
        self._capped: set[str] = set()
        self._lock = threading.Lock()

    def open(self, run_id: str) -> None:
        """Start keeping `run_id`'s events."""
        with self._lock:
            self._runs[run_id] = []

    def close(self, run_id: str) -> tuple[Event, ...]:
        """Stop keeping `run_id`'s events; the events kept so far."""
        with self._lock:
            self._capped.discard(run_id)
            return tuple(self._runs.pop(run_id, ()))

    def emit(self, event: Event) -> None:
        """Keep `event` when its run is open and under the cap."""
        with self._lock:
            kept = self._runs.get(event.run_id)
            if kept is None:
                return
            if len(kept) < MAX_EVENTS:
                kept.append(event)
                return
            first_drop = event.run_id not in self._capped
            self._capped.add(event.run_id)
        if first_drop:
            logger.warning(
                "eval run %s: over %d events, the rest are dropped", event.run_id, MAX_EVENTS
            )


def tools_of(events: tuple[Event, ...]) -> tuple[str, ...]:
    """The tools the run started, in order (a tool called twice appears twice)."""
    return tuple(str(e.data.get("tool", "")) for e in events if e.type == EventType.TOOL_STARTED)


def case_cost(ledger: LedgerPort | None, run_id: str, events: tuple[Event, ...]) -> float:
    """The run's cost: the ledger's total, or the events' total when that is larger."""
    from_events = 0.0
    for event in events:
        key = _COST_KEYS.get(event.type)
        value = event.data.get(key) if key else None
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            from_events += float(value)
    from_ledger = ledger.total_usd(run_id) if ledger is not None else 0.0
    return max(from_events, from_ledger)
