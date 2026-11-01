"""Run-file names, appends and rotation; each process writes only its own files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FOLDER = ".agent_adapter/runs"
SUFFIX = ".jsonl"
FORMAT_VERSION = 1
NO_RUN = "no-run"
GRACE_S = 900.0  # a run written to this recently and not ended is live, in any process
_TAIL_BYTES = 8_192
_END_MARKERS = (b'"type":"run.finished"', b'"kind":"end"')
_SAFE_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")
# <start ms>.<run key>.<pid>.jsonl: one segment per run and process, so no two processes
# ever append to the same file.
_NAME = re.compile(r"^(\d{13})\.([A-Za-z0-9_-]{1,64})\.(\d{1,10})\.jsonl$")


@dataclass(frozen=True)
class RunFile:
    """One segment of a run: the part one process wrote."""

    path: Path
    start_ms: int
    key: str
    pid: int


def run_key(run_id: str) -> str:
    """A file-name-safe key for a run id: the id itself, or a hash of an unusual one."""
    if not run_id:
        return NO_RUN
    if _SAFE_KEY.fullmatch(run_id):
        return run_id
    return "h" + hashlib.sha256(run_id.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def segment_name(start_ms: int, key: str, pid: int) -> str:
    """The file name of one run segment."""
    return f"{start_ms:013d}.{key}.{pid}{SUFFIX}"


def parse_name(path: Path) -> RunFile | None:
    """The segment a path names, or None for any other file."""
    match = _NAME.match(path.name)
    if match is None:
        return None
    return RunFile(path, int(match.group(1)), match.group(2), int(match.group(3)))


def list_segments(folder: Path) -> list[RunFile]:
    """Every run segment in `folder`, oldest first; a missing folder has none.

    Raises:
        OSError: the folder exists but cannot be listed.
    """
    try:
        entries = list(os.scandir(folder))
    except FileNotFoundError:
        return []
    found = [parse_name(Path(entry.path)) for entry in entries if entry.is_file()]
    return sorted((f for f in found if f is not None), key=lambda f: (f.start_ms, f.path.name))


def group_runs(segments: Iterable[RunFile]) -> dict[str, list[RunFile]]:
    """Segments by run key, keys ordered by their first segment."""
    groups: dict[str, list[RunFile]] = {}
    for segment in sorted(segments, key=lambda f: (f.start_ms, f.path.name)):
        groups.setdefault(segment.key, []).append(segment)
    return groups


def dumps(record: Mapping[str, Any]) -> bytes:
    """One JSONL line; values JSON cannot carry are written as their `str`."""
    text = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
    return text.encode("utf-8", "surrogatepass") + b"\n"


def append(path: Path, lines: list[bytes]) -> None:
    """Append whole lines in one write; the file is opened per call, so rotation can delete it.

    Raises:
        OSError: the file cannot be written.
    """
    with path.open("ab") as handle:
        handle.write(b"".join(lines))


def rotate(folder: Path, keep_runs: int, active: set[str], *, grace_s: float = GRACE_S) -> int:
    """Delete the oldest runs beyond `keep_runs`; return files deleted.

    A run is kept while this process has it open (`active`), or while any process wrote it in
    the last `grace_s` seconds and it has not ended, so other workers' live runs survive.

    Raises:
        OSError: the folder cannot be listed.
    """
    groups = group_runs(list_segments(folder))
    extra = len(groups) - keep_runs
    deleted = 0
    now = time.time()
    for key, segments in groups.items():
        if extra <= 0:
            break
        if key in active or _live(segments, now, grace_s):
            continue
        extra -= 1
        for segment in segments:
            try:
                segment.path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue  # another process rotated it first
            except OSError as exc:
                # Windows refuses to delete a file another process holds open; next time.
                logger.debug("run rotation skipped %s: %s", segment.path.name, exc)
    if deleted:
        logger.debug("run rotation deleted %d file(s) in %s", deleted, folder)
    return deleted


def _live(segments: list[RunFile], now: float, grace_s: float) -> bool:
    """True for a run written to within `grace_s` whose newest segment has not ended."""
    stamped = []
    for segment in segments:
        try:
            stamped.append((segment.path.stat().st_mtime, segment.path))
        except FileNotFoundError:
            continue
    if not stamped:
        return False
    mtime, newest = max(stamped)
    return now - mtime < grace_s and not ended(newest)


def ended(path: Path) -> bool:
    """True when the file's last lines hold `run.finished` or an inferred end."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _TAIL_BYTES))
            tail = handle.read()
    except FileNotFoundError:
        return True
    return any(marker in tail for marker in _END_MARKERS)
