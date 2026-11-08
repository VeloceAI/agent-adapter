"""Check signed records one at a time: canonical form, MACs, sequence, chain, open and seal."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.audit._chain import canonical, chain_hash, check_mac

MAX_PROBLEMS = 50  # per check; past this the log is plainly not intact
OnRecord = Callable[[dict[str, Any], bytes], None]


@dataclass
class Cursor:
    """Where the chain stands: the next seq and the hash the next record must carry."""

    seq: int
    prev: str


@dataclass
class SegmentCheck:
    """What checking one segment (or stream) found; counts only, nothing is kept."""

    problems: list[str] = field(default_factory=list)
    records: int = 0
    events: int = 0
    sealed: bool = False
    lines: int = 0
    tail: bytes = b""  # the last complete line, for the head check


def check_segment(
    lines: Iterable[bytes],
    key: bytes,
    cursor: Cursor,
    *,
    name: str,
    index: int,
    on_record: OnRecord | None = None,
) -> SegmentCheck:
    """Check segment `name` line by line; `cursor` advances, `on_record` sees each record."""
    out = SegmentCheck()
    for number, line in enumerate(lines, 1):
        out.lines, out.tail = number, line
        where = f"{name} line {number}"
        record = _parse(line)
        if record is None:
            _problem(out, f"{where}: not a signed record (edited or damaged)")
            cursor.prev = chain_hash(line)
            continue
        out.records += 1
        _check_record(out, record, line, key, cursor, where)
        kind = record.get("kind")
        if out.sealed:
            _problem(out, f"{where}: a record after the segment's seal")
        if number == 1 and (
            kind != "open" or record.get("segment") != name or record.get("index") != index
        ):
            _problem(out, f"{where}: the segment does not start with its own open record")
        elif number > 1 and kind == "open":
            _problem(out, f"{where}: a second open record")
        if kind == "event":
            out.events += 1
        elif kind == "seal":
            out.sealed = True
            if record.get("events") != out.events or record.get("segment") != name:
                _problem(out, f"{where}: the seal does not match the segment's records")
        if on_record is not None:
            on_record(record, line)
    if not out.lines:
        _problem(out, f"{name}: empty segment")
    return out


def check_streams(records: Iterable[Mapping[str, Any]], key: bytes) -> SegmentCheck:
    """Check shipped records, which may interleave several streams; each from its open."""
    out = SegmentCheck()
    open_streams: dict[str, tuple[Cursor, list[int]]] = {}  # stream: (cursor, [events])
    sealed: set[str] = set()
    for number, given in enumerate(records, 1):
        record = dict(given)
        stream = str(record.get("stream", ""))
        where = f"stream {stream or '?'} record {number}"
        line = canonical(record)
        out.records += 1
        if stream not in open_streams:
            open_streams[stream] = (Cursor(_int(record.get("seq")), str(record.get("prev"))), [0])
            if record.get("kind") != "open":
                _problem(out, f"{where}: the stream does not start with an open record")
        cursor, events = open_streams[stream]
        _check_record(out, record, line, key, cursor, where)
        if stream in sealed:
            _problem(out, f"{where}: a record after the stream's seal")
        kind = record.get("kind")
        if kind == "event":
            out.events += 1
            events[0] += 1
        elif kind == "seal":
            sealed.add(stream)
            if record.get("events") != events[0]:
                _problem(out, f"{where}: the seal does not match the stream's records")
    unsealed = sorted(set(open_streams) - sealed)
    for stream in unsealed:
        _problem(out, f"stream {stream or '?'}: no seal (cut, or the writer did not close)")
    out.sealed = bool(open_streams) and not unsealed
    out.lines = len(open_streams)  # the number of streams
    return out


def _check_record(
    out: SegmentCheck,
    record: dict[str, Any],
    line: bytes,
    key: bytes,
    cursor: Cursor,
    where: str,
) -> None:
    if not check_mac(key, record):
        _problem(out, f"{where}: MAC mismatch (edited, or another key)")
    seq = _int(record.get("seq"))
    if seq != cursor.seq:
        _problem(out, f"{where}: seq {seq}, expected {cursor.seq} (missing or reordered)")
    if record.get("prev") != cursor.prev:
        _problem(out, f"{where}: hash chain broken (a record before it was removed or changed)")
    cursor.seq = seq + 1  # resynchronise, so one gap is reported once
    cursor.prev = chain_hash(line)


def _parse(line: bytes) -> dict[str, Any] | None:
    try:
        record = json.loads(line)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    try:
        same = canonical(record) == line
    except (TypeError, ValueError):
        return None
    return record if same else None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _problem(out: SegmentCheck, text: str) -> None:
    if len(out.problems) < MAX_PROBLEMS:
        out.problems.append(text)
