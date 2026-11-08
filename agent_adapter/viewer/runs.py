"""Reads a runs folder (or one run file) incrementally: small summaries, details on demand."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.viewer._files import RunFile, group_runs, list_segments, parse_name
from agent_adapter.viewer._fold import Fold, summarise, valid_event
from agent_adapter.viewer._search import SearchFold
from agent_adapter.viewer.lanes import build_spans
from agent_adapter.viewer.layout import layout

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 64 * 1024 * 1024  # read at most this much of one segment
CHUNK = 1024 * 1024
EVENT_PAGE = 5_000  # events per detail response
MAX_SPANS = 5_000  # spans per detail response: flows and nodes, then the latest calls
DETAILS_KEPT = 4  # runs whose events stay in memory (least recently viewed dropped)


@dataclass
class _Tail:
    """How far one segment has been read, and what it added up to."""

    offset: int = 0
    rest: bytes = b""
    bad_lines: int = 0
    mtime: float = 0.0
    fold: Fold = field(default_factory=Fold)
    search: SearchFold = field(default_factory=SearchFold)


@dataclass
class _Detail:
    """One run's records in arrival order, read from where they stopped."""

    tails: dict[Path, _Tail] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    built: tuple[int, dict[str, Any]] | None = None  # (events seen, spans/graphs/tools)
    placed: dict[int, dict[str, Any] | None] = field(default_factory=dict)  # by record index


class RunFolder:
    """A runs folder, or a single run file, read again on each `refresh` from where it stopped.

    Raises:
        ConfigError: `target` is a file that is not a run segment.
    """

    def __init__(self, target: str | Path) -> None:
        self.target = Path(target).resolve()
        self._single = self.target.is_file()
        if self._single and parse_name(self.target) is None:
            raise ConfigError(f"{self.target} is not a run file (<start>.<run>.<pid>.jsonl)")
        self._tails: dict[Path, _Tail] = {}
        self._segments: list[RunFile] = []
        self._cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self._details: OrderedDict[str, _Detail] = OrderedDict()
        self._lock = threading.Lock()

    def refresh(self) -> None:
        """Pick up new runs and new lines; deleted (rotated) runs are dropped.

        Raises:
            OSError: the folder cannot be listed.
        """
        with self._lock:
            if self._single:
                found = parse_name(self.target)
                segments = [found] if found is not None and self.target.exists() else []
            else:
                segments = list_segments(self.target)
            alive = {segment.path for segment in segments}
            for path in [p for p in self._tails if p not in alive]:
                del self._tails[path]
            for segment in segments:
                tail = self._tails.setdefault(segment.path, _Tail())
                for record in _read(segment.path, tail):
                    tail.fold.add(record)
                    tail.search.add(record)
            self._segments = segments

    def summaries(self) -> list[dict[str, Any]]:
        """One summary per run, newest first."""
        bucket = int(time.time() // 10)  # "running" turns "unfinished" with time alone
        with self._lock:
            found = []
            for key, files in group_runs(self._segments).items():  # oldest first
                tails = [self._tails[f.path] for f in files if f.path in self._tails]
                signature = (tuple(t.offset for t in tails), bucket)
                cached = self._cache.get(key)
                if cached is None or cached[0] != signature:
                    mtime = max((t.mtime for t in tails), default=0.0)
                    summary = summarise(key, [t.fold for t in tails], mtime, {f.pid for f in files})
                    cached = self._cache[key] = (signature, summary)
                found.append(cached[1])
            for key in set(self._cache) - {s["key"] for s in found}:
                del self._cache[key]
        # Equal start stamps (a coarse clock) keep file order: the later file is newer.
        order = {id(s): n for n, s in enumerate(found)}
        return sorted(found, key=lambda s: (s["started"], order[id(s)]), reverse=True)

    @property
    def tags_folder(self) -> Path:
        """Where the run tag sidecars live: the runs folder, or the run file's folder."""
        return self.target.parent if self._single else self.target

    def search_folds(self) -> dict[str, list[SearchFold]]:
        """Copies of the search folds of each run's segments, by run key (safe to read later)."""
        with self._lock:
            return {
                key: [self._tails[f.path].search.copy() for f in files if f.path in self._tails]
                for key, files in group_runs(self._segments).items()
            }

    def detail(self, key: str, since: int = 0, *, paged: bool = True) -> dict[str, Any] | None:
        """One run: summary, graphs, tools, spans with lanes, and events `since` (paged).

        None for an unknown run.
        """
        with self._lock:
            files = group_runs(self._segments).get(key)
            if not files:
                return None
            detail = self._load(key, files)
            built = self._build(detail)
            events = detail.events
        summary = next((s for s in self.summaries() if s["key"] == key), None)
        start = max(0, since)
        stop = start + EVENT_PAGE if paged else len(events)
        return {
            "summary": summary,
            **built,
            "events": events[start:stop],
            "events_from": start,
            "events_total": len(events),
        }

    def _load(self, key: str, files: list[RunFile]) -> _Detail:
        detail = self._details.get(key)
        paths = {f.path for f in files}
        if detail is None or any(path not in paths for path in detail.tails):
            detail = _Detail()  # first view, or a segment was rotated away: read again
        self._details[key] = detail
        self._details.move_to_end(key)
        while len(self._details) > DETAILS_KEPT:
            self._details.popitem(last=False)
        for segment in files:
            tail = detail.tails.setdefault(segment.path, _Tail())
            for record in _read(segment.path, tail):
                detail.records.append(record)
                event = valid_event(record)
                if event is not None:
                    detail.events.append(event)
        return detail

    def _build(self, detail: _Detail) -> dict[str, Any]:
        if detail.built is not None and detail.built[0] == len(detail.records):
            return detail.built[1]
        records = detail.records
        graphs = {}
        for index, record in enumerate(records):
            if _is(record, "graph"):
                if index not in detail.placed:  # laid out (or warned about) once per record
                    detail.placed[index] = _placed(record)
                placed = detail.placed[index]
                if placed is not None:
                    graphs[str(record.get("flow", ""))] = placed
        tools = next((r.get("tools") for r in reversed(records) if _is(r, "tools")), None)
        spans = build_spans(detail.events)
        built = {
            "meta": [r for r in records if _is(r, "meta")],
            "graphs": graphs,
            "tools": tools,
            "spans": _cap_spans(spans),
            "spans_total": len(spans),
            "truncated": any(_is(r, "truncated") for r in records),
        }
        detail.built = (len(records), built)
        return built


def _read(path: Path, tail: _Tail) -> Iterator[dict[str, Any]]:
    """New complete lines of `path` since `tail.offset`, parsed; a partial last line waits."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    tail.mtime = stat.st_mtime
    end = min(stat.st_size, MAX_FILE_BYTES)
    if end <= tail.offset:
        return
    try:
        with path.open("rb") as handle:
            handle.seek(tail.offset)
            while tail.offset < end:
                chunk = handle.read(min(CHUNK, end - tail.offset))
                if not chunk:
                    break
                tail.offset += len(chunk)
                lines = (tail.rest + chunk).split(b"\n")
                tail.rest = lines.pop()  # a line still being written has no newline yet
                yield from _parse(lines, tail, path)
    except FileNotFoundError:
        return  # rotated away between listing and reading


def _parse(lines: list[bytes], tail: _Tail, path: Path) -> Iterator[dict[str, Any]]:
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            tail.bad_lines += 1
            logger.debug("skipped a malformed line in %s", path.name)
            continue
        if isinstance(record, dict):
            yield record


def _cap_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At most MAX_SPANS: every flow and node span, then the latest calls."""
    if len(spans) <= MAX_SPANS:
        return spans
    frame = [s for s in spans if s["family"] in ("flow", "node")][:MAX_SPANS]
    calls = [s for s in spans if s["family"] not in ("flow", "node")]
    room = MAX_SPANS - len(frame)
    keep = {id(s) for s in [*frame, *(calls[-room:] if room else [])]}
    return [s for s in spans if id(s) in keep]


def _placed(record: dict[str, Any]) -> dict[str, Any] | None:
    """The record's graph with its layout; None (logged) for a malformed graph."""
    try:
        return layout(record["graph"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        logger.warning(
            "skipped the malformed graph of flow %r (%s)", record.get("flow"), type(exc).__name__
        )
        return None


def _is(record: dict[str, Any], kind: str) -> bool:
    return record.get("kind") == kind
