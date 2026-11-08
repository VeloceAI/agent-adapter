"""The audit folder on disk: segment names, the signed head file and the one-writer lock."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from agent_adapter.audit._chain import canonical, check_mac, mac
from agent_adapter.core.errors import ConfigError, log_failure

logger = logging.getLogger(__name__)

HEAD = "audit.head"
LOCK = "audit.lock"
_SEGMENT = re.compile(r"^audit-(\d{8})-(\d{6})\.jsonl$")
_REPLACE_TRIES = 5  # a reader holding the head open on Windows blocks os.replace briefly


def segment_name(index: int, ts: float) -> str:
    """`audit-YYYYMMDD-000123.jsonl` for a segment opened at `ts` (UTC day)."""
    return f"audit-{day(ts)}-{index:06d}.jsonl"


def day(ts: float) -> str:
    """The UTC day of `ts` as `YYYYMMDD`."""
    return datetime.fromtimestamp(ts, UTC).strftime("%Y%m%d")


def segments(folder: Path) -> list[tuple[int, Path]]:
    """Every segment file in `folder`, by index."""
    found = []
    for path in folder.iterdir() if folder.is_dir() else []:
        match = _SEGMENT.match(path.name)
        if match is not None and path.is_file():
            found.append((int(match.group(2)), path))
    return sorted(found)


class Lines:
    """The complete lines of a segment, read one at a time; `partial` once a cut tail is seen."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.partial = False

    def __iter__(self) -> Iterator[bytes]:
        with self.path.open("rb") as source:
            for raw in source:
                if raw.endswith(b"\n"):
                    yield raw[:-1]
                else:
                    self.partial = True  # no newline: a write that never finished, or a cut


@dataclass(frozen=True)
class Head:
    """Where the chain ends: the last record's seq and hash, its segment, and if it is sealed."""

    seq: int
    last: str
    index: int
    segment: str
    sealed: bool
    log_id: str = ""  # names this log, so segments or heads of another log show

    def body(self) -> dict[str, Any]:
        """The fields the head file signs."""
        return {
            "v": 1,
            "seq": self.seq,
            "last": self.last,
            "index": self.index,
            "segment": self.segment,
            "sealed": self.sealed,
            "log_id": self.log_id,
        }


def write_head(folder: Path, key: bytes, head: Head, *, fsync: bool) -> None:
    """Replace the head file atomically with a signed copy of `head`."""
    body = head.body()
    body["mac"] = mac(key, body)
    temp = folder / f"{HEAD}.tmp"
    with temp.open("wb") as out:
        out.write(canonical(body) + b"\n")
        out.flush()
        if fsync:
            os.fsync(out.fileno())
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(temp, folder / HEAD)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(0.02)


def read_head(folder: Path, key: bytes) -> Head | None:
    """The signed head, or None when there is none.

    Raises:
        ConfigError: the head is unreadable or its MAC fails (another key, or an edit).
    """
    path = folder / HEAD
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"the audit head {path} is unreadable") from exc
    if not isinstance(body, dict) or not check_mac(key, body):
        raise ConfigError(f"the audit head {path} does not match this key (edited, or another key)")
    try:
        return Head(
            int(body["seq"]),
            str(body["last"]),
            int(body["index"]),
            str(body["segment"]),
            bool(body["sealed"]),
            str(body["log_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"the audit head {path} is malformed") from exc


class FolderLock:
    """An exclusive OS lock on `audit.lock`: one writer per folder, across processes."""

    def __init__(self, folder: Path) -> None:
        self._file: IO[bytes] | None = None
        path = folder / LOCK
        handle = path.open("a+b")
        try:
            _lock(handle)
        except OSError as exc:
            handle.close()
            raise ConfigError(f"another writer holds the audit folder {folder}") from exc
        self._file = handle

    def release(self) -> None:
        """Give the folder up; safe to call twice."""
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            _unlock(handle)
        except OSError as exc:
            log_failure(logger, logging.WARNING, exc, "audit lock release failed")
        finally:
            handle.close()


if sys.platform == "win32":
    import msvcrt

    def _lock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
