"""Turns events into the lines of each run's segment file (used by the recorder's writer)."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_adapter._version import __version__
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import Event
from agent_adapter.tracing.events import EventType
from agent_adapter.viewer._files import (
    FORMAT_VERSION,
    NO_RUN,
    dumps,
    rotate,
    run_key,
    segment_name,
)
from agent_adapter.viewer._scrub import catalogue, scrub_data
from agent_adapter.viewer.lanes import END_TYPES, SPAN_FAMILIES, span_identity

logger = logging.getLogger(__name__)

MAX_OPEN = 4_096  # runs tracked per process; the oldest is forgotten first
ROTATE_EVERY_S = 1.0
# Written past the per-run cap, so a capped run still ends.
_ALWAYS = frozenset({EventType.RUN_FINISHED, EventType.FLOW_FINISHED})


@dataclass
class Segment:
    """What this process wrote of one run."""

    path: Path
    events: int = 0
    flows: set[str] = field(default_factory=set)
    truncated: bool = False
    scoped: bool = False  # saw run.started: run.finished ends it
    open: set[str] = field(default_factory=set)  # spans started, not yet ended
    trace_state: bool = False  # a flow of this run asked for state values on node.finished


@dataclass
class Options:
    """The recorder's settings the writer needs."""

    folder: Path
    capture_content: bool
    keep_runs: int
    max_events_per_run: int
    registry: Callable[[], Registry | None]
    source: str


class Segments:
    """Per-run segment state of one process; only the writer thread touches it."""

    def __init__(self, options: Options) -> None:
        self.options = options
        self.open: OrderedDict[str, Segment] = OrderedDict()
        self._last_rotation = 0.0
        self._catalogue: tuple[tuple[int, ...], bytes] | None = None

    def event(self, event: Event, out: dict[Path, list[bytes]]) -> None:
        """Add the lines for `event` to `out`, by file."""
        key = run_key(event.run_id)
        segment = self._segment(key, event.run_id, event.ts, out)
        lines = out.setdefault(segment.path, [])
        if segment.events >= self.options.max_events_per_run and event.type not in _ALWAYS:
            if not segment.truncated:
                segment.truncated = True
                lines.append(dumps({"kind": "truncated", "ts": event.ts}))
            return
        if event.type == EventType.FLOW_STARTED and event.data.get("trace_state") is True:
            if not segment.trace_state and not self.options.capture_content:
                # State values reach disk although content capture is off: say so once per run.
                logger.info("run %s traces flow state: node state values are recorded", key)
            segment.trace_state = True
        clean, omitted = scrub_data(
            event.data,
            capture_content=self.options.capture_content,
            keep_state=segment.trace_state,
        )
        record: dict[str, Any] = {"kind": "event", **event.to_dict(), "data": clean}
        if omitted:
            record["omitted"] = omitted
        lines.append(dumps(record))
        segment.events += 1
        if event.type == EventType.RUN_STARTED:
            segment.scoped = True
        if event.type == EventType.RUN_FINISHED:
            self.open.pop(key, None)
        elif self._settled(segment, event) and key != NO_RUN:
            # A call outside a run scope has no run.finished: its run ends with its last call.
            lines.append(dumps({"kind": "end", "ts": event.ts, "inferred": True}))
            self.open.pop(key, None)

    def graph(self, run_id: str, flow: str, graph: dict[str, Any], ts: float, out: Any) -> None:
        """Add one graph line per flow and segment."""
        segment = self._segment(run_key(run_id), run_id, ts, out)
        if flow in segment.flows:
            return
        segment.flows.add(flow)
        out.setdefault(segment.path, []).append(
            dumps({"kind": "graph", "flow": flow, "graph": graph})
        )

    def _settled(self, segment: Segment, event: Event) -> bool:
        """Track open spans; True when an unscoped run has just closed its last one."""
        family, key = span_identity(event.type, event.call_id, event.data)
        if family not in SPAN_FAMILIES:
            return False
        if event.type.endswith(".started"):
            segment.open.add(key)
            return False
        if event.type not in END_TYPES:
            return False
        segment.open.discard(key)
        return not segment.scoped and not segment.open

    def _segment(self, key: str, run_id: str, ts: float, out: dict[Path, list[bytes]]) -> Segment:
        found = self.open.get(key)
        if found is not None:
            self.open.move_to_end(key)
            return found
        path = self.options.folder / segment_name(int(ts * 1000), key, os.getpid())
        segment = Segment(path)
        self.open[key] = segment
        while len(self.open) > MAX_OPEN:
            self.open.popitem(last=False)
        out.setdefault(path, []).extend([dumps(self._meta(run_id, ts)), *self._tools()])
        self._maybe_rotate()
        return segment

    def _meta(self, run_id: str, ts: float) -> dict[str, Any]:
        return {
            "kind": "meta",
            "v": FORMAT_VERSION,
            "run_id": run_id,
            "pid": os.getpid(),
            "started": ts,
            "source": self.options.source,
            "capture_content": self.options.capture_content,
            "agent_adapter": __version__,
            "python": sys.version.split()[0],
        }

    def _tools(self) -> list[bytes]:
        registry = self.options.registry()
        if registry is None:
            return []
        version = tuple(id(spec) for spec in registry)
        if self._catalogue is None or self._catalogue[0] != version:
            line = dumps({"kind": "tools", "tools": catalogue(registry)})
            self._catalogue = (version, line)
        return [self._catalogue[1]]

    def _maybe_rotate(self) -> None:
        now = time.monotonic()
        if now - self._last_rotation < ROTATE_EVERY_S:
            return
        self._last_rotation = now
        rotate(self.options.folder, self.options.keep_runs, set(self.open))
