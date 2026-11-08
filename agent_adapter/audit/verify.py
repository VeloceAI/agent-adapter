"""Check an audit log end to end, streaming: edits, deletions, reordering and cuts all show."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_adapter.audit._chain import GENESIS, chain_hash, resolve_key
from agent_adapter.audit._check import (
    MAX_PROBLEMS,
    Cursor,
    OnRecord,
    check_segment,
    check_streams,
)
from agent_adapter.audit._files import Head, Lines, read_head, segments
from agent_adapter.core.errors import ConfigError


@dataclass(frozen=True)
class AuditVerification:
    """The result of `verify`: `ok` only when nothing at all is wrong."""

    ok: bool
    segments: int
    records: int
    events: int
    sealed: bool  # the last segment (every stream) is sealed: the writer closed cleanly
    problems: tuple[str, ...] = ()
    log_id: str = ""
    last_seq: int = -1
    last_hash: str = ""

    @property
    def anchor(self) -> str:
        """`log_id:seq:hash` of the end; keep it elsewhere and pass it to a later `verify`."""
        return f"{self.log_id}:{self.last_seq}:{self.last_hash}"

    def summary(self) -> str:
        """One line for a terminal or a CI log."""
        state = "intact" if self.ok else f"NOT intact ({len(self.problems)} problem(s))"
        seal = "sealed" if self.sealed else "open"
        return f"{state}: {self.segments} segment(s), {self.events} event(s), last segment {seal}"


def verify(
    path: str | os.PathLike[str], key: str | bytes | None = None, *, anchor: str | None = None
) -> AuditVerification:
    """Check the audit folder `path` line by line; `anchor` (an earlier one) catches rollbacks.

    Raises:
        ConfigError: no key, a short key, `path` is not a folder, or a malformed `anchor`.
    """
    return walk(_folder(path), resolve_key(key), anchor=_anchor(anchor))


def read_entries(
    path: str | os.PathLike[str], key: str | bytes | None = None
) -> Iterator[dict[str, Any]]:
    """The event records of a verified log, oldest first, one at a time.

    Raises:
        ConfigError: no key, a bad folder, or a log that is not intact (checked before the
            first record, and again while reading).
    """
    for record, _ in signed_events(path, key):
        yield record


def signed_events(
    path: str | os.PathLike[str], key: str | bytes | None
) -> Iterator[tuple[dict[str, Any], bytes]]:
    """Each event record with its stored line, after a full check; a later change raises."""
    folder, secret = _folder(path), resolve_key(key)
    checked = walk(folder, secret)
    if not checked.ok:
        raise ConfigError(f"the audit log is not intact: {checked.problems[0]}")
    found: list[tuple[dict[str, Any], bytes]] = []
    cursor = Cursor(0, GENESIS)

    def keep(record: dict[str, Any], line: bytes) -> None:
        if record.get("kind") == "event" and record.get("seq", -1) <= checked.last_seq:
            found.append((record, line))

    for index, file in segments(folder):
        result = check_segment(
            Lines(file), secret, cursor, name=file.name, index=index, on_record=keep
        )
        if result.problems:
            raise ConfigError("the audit log changed while it was read")
        yield from found
        found.clear()


class _Tracker:
    """What `walk` needs from each record: the end, log ids, and hashes at given seqs."""

    def __init__(self, wanted: set[int], on_record: OnRecord | None) -> None:
        self.wanted = wanted
        self.hashes: dict[int, str] = {}
        self.log_ids: set[str] = set()
        self.last_seq, self.last_hash = -1, ""
        self._next = on_record

    def __call__(self, record: dict[str, Any], line: bytes) -> None:
        seq = record.get("seq")
        if isinstance(seq, int):
            self.last_seq, self.last_hash = seq, chain_hash(line)
            if seq in self.wanted:
                self.hashes[seq] = self.last_hash
        if record.get("kind") == "open":
            self.log_ids.add(str(record.get("log_id", "")))
        if self._next is not None:
            self._next(record, line)


def walk(
    folder: Path,
    key: bytes,
    on_record: OnRecord | None = None,
    anchor: tuple[str, int, str] | None = None,
) -> AuditVerification:
    """Check `folder` streaming; the head is read first, so a live writer's growth is fine."""
    problems: list[str] = []
    head, head_problem = _head(folder, key)
    problems += head_problem
    wanted = {head.seq} if head is not None else set()
    if anchor is not None:
        wanted.add(anchor[1])
    seen = _Tracker(wanted, on_record)
    found = segments(folder)
    cursor = Cursor(0, GENESIS)
    records = events = 0
    sealed = False
    if not found:
        problems.append("no audit segments found (nothing written, or everything removed)")
    for position, (index, file) in enumerate(found):
        if index != position:
            problems.append(f"segment {position} is missing (found {file.name})")
        lines = Lines(file)
        checked = check_segment(lines, key, cursor, name=file.name, index=index, on_record=seen)
        if lines.partial and position < len(found) - 1:
            problems.append(f"{file.name}: ends in a partial record (cut)")
        problems += checked.problems
        if not checked.sealed and position < len(found) - 1:
            problems.append(f"{file.name}: not sealed, but a later segment exists")
        records += checked.records
        events += checked.events
        sealed = checked.sealed
    log_id = head.log_id if head is not None else ""
    problems += _end_problems(head, seen, found)
    if anchor is not None:
        problems += _anchor_problems(anchor, log_id, seen)
    return AuditVerification(
        ok=not problems,
        segments=len(found),
        records=records,
        events=events,
        sealed=sealed,
        problems=tuple(problems[:MAX_PROBLEMS]),
        log_id=log_id,
        last_seq=seen.last_seq,
        last_hash=seen.last_hash,
    )


def _head(folder: Path, key: bytes) -> tuple[Head | None, list[str]]:
    try:
        return read_head(folder, key), []
    except ConfigError as exc:
        return None, [str(exc)]


def _end_problems(head: Head | None, seen: _Tracker, found: list[tuple[int, Path]]) -> list[str]:
    if head is None:
        return ["audit.head is missing (the end of the chain cannot be checked)"] if found else []
    if not found:
        return ["audit.head names a segment, but there are none (all removed)"]
    out = []
    if seen.log_ids - {head.log_id}:
        out.append("a segment belongs to another log (log id differs from audit.head)")
    if seen.last_seq < head.seq:
        out.append(f"the log ends at seq {seen.last_seq}, but audit.head says {head.seq} (cut)")
    elif seen.hashes.get(head.seq) != head.last or head.index > found[-1][0]:
        out.append(f"the record at seq {head.seq} does not match audit.head (changed)")
    return out


def _anchor_problems(anchor: tuple[str, int, str], log_id: str, seen: _Tracker) -> list[str]:
    want_log, seq, digest = anchor
    if want_log != log_id:
        return ["this is another log than the anchor names (folder replaced)"]
    if seen.last_seq < seq:
        return [f"the log ends at seq {seen.last_seq}, before the anchor's {seq} (rolled back)"]
    if seen.hashes.get(seq) != digest:
        return [f"the record at seq {seq} differs from the anchor (rewritten)"]
    return []


def _anchor(text: str | None) -> tuple[str, int, str] | None:
    if text is None:
        return None
    parts = str(text).split(":")
    if len(parts) != 3 or not parts[1].isdigit() or len(parts[2]) != 64:
        raise ConfigError("an anchor is log_id:seq:hash, as AuditVerification.anchor gives it")
    return parts[0], int(parts[1]), parts[2]


def _folder(path: str | os.PathLike[str]) -> Path:
    folder = Path(path)
    if not folder.is_dir():
        raise ConfigError(f"{folder} is not an audit folder")
    return folder


def verify_records(
    records: Iterable[Mapping[str, Any]], key: str | bytes | None = None
) -> AuditVerification:
    """Check shipped records in the order received; interleaved streams are checked apart.

    Raises:
        ConfigError: no key, or a short one.
    """
    checked = check_streams(records, resolve_key(key))
    return AuditVerification(
        ok=not checked.problems,
        segments=checked.lines,
        records=checked.records,
        events=checked.events,
        sealed=checked.sealed,
        problems=tuple(checked.problems),
    )
