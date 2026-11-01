"""A run summary folded line by line, so the runs list never keeps a run's events in memory."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.viewer.lanes import END_TYPES, SPAN_FAMILIES, span_identity

LIVE_S = 120.0  # a run without an end, written to this recently, is "running"
_PENDING = frozenset({"tool.pending", "node.paused"})
_FAILED = frozenset({"tool.failed", "llm.failed", "node.failed", "embedding.failed"})


def valid_event(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The record as an event (with dict `data`), or None when it is not a usable event."""
    if record.get("kind") != "event" or not isinstance(record.get("type"), str):
        return None
    ts = record.get("ts")
    if not isinstance(ts, int | float) or isinstance(ts, bool):
        return None
    data = record.get("data")
    return {**record, "data": data if isinstance(data, dict) else {}}


@dataclass
class Fold:
    """The summary state of one segment."""

    meta: dict[str, Any] | None = None
    events: int = 0
    first: float | None = None
    last: float | None = None
    cost: float | None = None
    errors: int = 0
    top_failures: int = 0
    flows: set[str] = field(default_factory=set)
    framework: str = ""
    providers: set[str] = field(default_factory=set)
    finished: list[bool] = field(default_factory=list)
    scoped: bool = False
    pending: dict[str, bool] = field(default_factory=dict)  # call -> waiting (last state)
    spans: dict[str, bool] = field(default_factory=dict)  # span -> still open (last state)

    def add(self, record: Mapping[str, Any]) -> None:
        """Fold one parsed line in."""
        if record.get("kind") == "meta" and self.meta is None:
            self.meta = dict(record)
            return
        event = valid_event(record)
        if event is not None:
            self._event(event)

    def _event(self, event: dict[str, Any]) -> None:
        kind, data, ts = str(event["type"]), event["data"], float(event["ts"])
        call = str(event.get("call_id") or "")
        self.events += 1
        self.first = ts if self.first is None else min(self.first, ts)
        self.last = ts if self.last is None else max(self.last, ts)
        for key in ("cost_usd", "summary_cost_usd"):
            value = data.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                self.cost = (self.cost or 0.0) + float(value)
        if data.get("error_kind"):
            self.errors += 1
        if kind in _FAILED and not event.get("parent_id"):
            self.top_failures += 1
        if kind == "flow.started":
            self.flows.add(str(data.get("flow", "")))
        if not self.framework:
            self.framework = str(data.get("framework") or data.get("backend") or "")
        if kind == "llm.started" and data.get("provider"):
            self.providers.add(str(data["provider"]))
        if kind == "run.started":
            self.scoped = True
        elif kind == "run.finished":
            self.finished.append(bool(data.get("ok")))
        if kind in _PENDING:
            self.pending[call] = True
        elif call and kind.startswith(("tool.", "node.")):
            self.pending[call] = False
        family, span = span_identity(kind, call, data)
        if family in SPAN_FAMILIES:
            if kind.endswith(".started"):
                self.spans[span] = True
            elif kind in END_TYPES:
                self.spans[span] = False


def summarise(key: str, folds: list[Fold], mtime: float, pids: set[int]) -> dict[str, Any]:
    """One run's summary from its segments' folds, oldest segment first."""
    pending: dict[str, bool] = {}
    spans: dict[str, bool] = {}
    for fold in folds:
        pending.update(fold.pending)
        spans.update(fold.spans)
    stamps = [f.first for f in folds if f.first is not None]
    ends = [f.last for f in folds if f.last is not None]
    started = min(stamps) if stamps else 0.0
    ended = max(ends) if ends else started
    costs = [f.cost for f in folds if f.cost is not None]
    meta = next((f.meta for f in folds if f.meta is not None), {})
    return {
        "key": key,
        "run_id": str(meta.get("run_id", key)),
        "started": started,
        "ended": ended,
        "duration_ms": round((ended - started) * 1000, 3),
        "status": _status(folds, pending, spans, mtime),
        "cost_usd": round(sum(costs), 9) if costs else None,
        "framework": _framework(folds),
        "flows": sorted(set().union(*(f.flows for f in folds))),
        "events": sum(f.events for f in folds),
        "errors": sum(f.errors for f in folds),
        "processes": len(pids),
    }


def _status(
    folds: list[Fold], pending: dict[str, bool], spans: dict[str, bool], mtime: float
) -> str:
    """paused, ok, failed, running (still written to) or unfinished."""
    if any(pending.values()):
        return "paused"
    finished = [ok for fold in folds for ok in fold.finished]
    if finished:
        return "ok" if all(finished) else "failed"
    scoped = any(fold.scoped for fold in folds)
    if not scoped and spans and not any(spans.values()):
        # Calls outside a run scope: the run is over once its last call ended.
        return "failed" if any(fold.top_failures for fold in folds) else "ok"
    return "running" if time.time() - mtime < LIVE_S else "unfinished"


def _framework(folds: list[Fold]) -> str:
    """The recorder's `source`, else a `framework` / `backend` an event names, else providers."""
    for fold in folds:
        if fold.meta and fold.meta.get("source"):
            return str(fold.meta["source"])
    for fold in folds:
        if fold.framework:
            return fold.framework
    return ", ".join(sorted(set().union(*(f.providers for f in folds))))
