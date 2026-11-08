"""The audit store port, and FileAuditStore: signed, chained JSONL segments on disk."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import weakref
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

from agent_adapter.audit._actions import AuditEntry
from agent_adapter.audit._chain import Chain, chain_hash, key_id, keyed_digest, resolve_key
from agent_adapter.audit._check import Cursor, check_segment
from agent_adapter.audit._files import (
    FolderLock,
    Head,
    Lines,
    day,
    read_head,
    segment_name,
    segments,
    write_head,
)
from agent_adapter.core.errors import ConfigError, IntegrationError, ToolUsageError, log_failure
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 10_000_000
_OPEN: weakref.WeakSet[FileAuditStore] = weakref.WeakSet()


@runtime_checkable
class AuditStore(Protocol):
    """Keeps audit entries signed and in order."""

    def write(self, entry: AuditEntry) -> None:
        """Sign and keep one entry; raises on failure (the sink logs it)."""
        ...

    def close(self) -> None:
        """Seal what was written; later writes are refused."""
        ...

    def digest(self, value: Any) -> str:
        """A keyed digest of `value` (a scope, edited arguments): linkable, not reversible."""
        ...


class FileAuditStore:
    """Append-only, chained, signed JSONL segments in `folder`; one writer per folder.

    Raises:
        ConfigError: no key or a short one, a bad limit, another writer, or a folder whose
            end does not check out (run `verify`).
    """

    def __init__(
        self,
        folder: str | os.PathLike[str],
        *,
        key: str | bytes | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        rotate_daily: bool = True,
        fsync: bool = True,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1024:
            raise ConfigError(f"max_bytes must be an int >= 1024, got {max_bytes!r}")
        self._key = resolve_key(key)
        self.folder = Path(folder).resolve()
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"cannot create the audit folder {self.folder}: {exc}") from exc
        self.max_bytes = max_bytes
        self.rotate_daily = bool(rotate_daily)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()
        self._file: IO[bytes] | None = None
        self._index = 0  # the active segment, or the next one when none is open
        self._name = ""
        self._day = ""
        self._size = 0
        self._events = 0
        self._closed = False
        self._broken = False
        self.log_id = ""
        self._folder_lock = FolderLock(self.folder)
        try:
            self._chain = self._resume()
        except BaseException:
            self._folder_lock.release()
            raise
        _OPEN.add(self)

    @property
    def key_id(self) -> str:
        """The public id of the signing key."""
        return key_id(self._key)

    def write(self, entry: AuditEntry) -> None:
        """Append one signed entry, rotating the segment first when due.

        Raises:
            ToolUsageError: the store is closed.
            IntegrationError: the store failed earlier and stopped, to keep the chain whole.
            OSError: the disk write failed (rolled back).
        """
        with self._lock:
            if self._closed:
                raise ToolUsageError("the audit store is closed")
            if self._broken:
                raise IntegrationError(
                    "the audit store stopped after a write failure", integration="audit"
                )
            now = time.time()
            if self._file is not None and self._due(now):
                self._seal()
            if self._file is None:
                self._open(now)
            self._append(entry.body())

    def close(self) -> None:
        """Seal the active segment and release the folder; safe to call twice."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._file is not None and not self._broken:
                    self._seal()
            finally:
                if self._file is not None:
                    self._file.close()
                    self._file = None
                self._folder_lock.release()

    def digest(self, value: Any) -> str:
        """A keyed digest of `value` under this store's key."""
        return keyed_digest(self._key, value)

    def _due(self, now: float) -> bool:
        if self.rotate_daily and day(now) != self._day:
            return True
        return self._events > 0 and self._size >= self.max_bytes

    def _open(self, now: float) -> None:
        self._name = segment_name(self._index, now)
        self._day = day(now)
        self._file = (self.folder / self._name).open("xb")
        self._size = 0
        self._events = 0
        self._append(
            {
                "kind": "open",
                "segment": self._name,
                "index": self._index,
                "key_id": self.key_id,
                "log_id": self.log_id,
                "ts": now,
            }
        )

    def _seal(self) -> None:
        body = {
            "kind": "seal",
            "segment": self._name,
            "index": self._index,
            "events": self._events,
            "ts": time.time(),
        }
        self._append(body)

    def _append(self, body: dict[str, Any]) -> None:
        handle = self._file
        assert handle is not None
        _, line = self._chain.sign(body)
        start = self._size
        try:
            handle.write(line + b"\n")
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        except OSError:
            self._roll_back(handle, start)
            raise
        # The record is stored: account for it before anything else can fail.
        self._size += len(line) + 1
        self._chain.advance(line)
        sealed = body["kind"] == "seal"
        if body["kind"] == "event":
            self._events += 1
        head = Head(
            self._chain.seq - 1, self._chain.prev, self._index, self._name, sealed, self.log_id
        )
        if sealed:
            handle.close()
            self._file = None
            self._index += 1
        try:
            write_head(self.folder, self._key, head, fsync=self.fsync)
        except OSError as exc:
            # The record is safe; a lagging head is repaired by the next write or open.
            log_failure(logger, logging.WARNING, exc, "audit head not updated")

    def _roll_back(self, handle: IO[bytes], start: int) -> None:
        # A fresh handle: an append handle that was truncated may still write past the cut.
        path = self.folder / self._name
        self._file = None
        try:
            handle.close()
        except OSError as exc:
            log_failure(logger, logging.WARNING, exc, "audit segment close failed")
        try:
            os.truncate(path, start)
            self._file = path.open("ab")
        except OSError:
            self._broken = True
            logger.error(
                "audit store %s stopped: a failed write could not be rolled back", self.folder
            )

    def _resume(self) -> Chain:
        head = read_head(self.folder, self._key)
        found = segments(self.folder)
        if head is None:
            if found:
                raise ConfigError(f"{self.folder} has audit segments but no head: run verify")
            self.log_id = new_id("log_")
            return Chain(self._key)
        self.log_id = head.log_id
        index, path = found[-1] if found else (-1, self.folder / head.segment)
        if index != head.index or path.name != head.segment:
            raise ConfigError(
                f"the audit head in {self.folder} does not match its last segment: run verify"
            )
        lines = Lines(path)
        first = next(iter(lines), None)
        if first is None or lines.partial:
            raise ConfigError(f"{path.name} ends in a partial record: run verify")
        cursor = _start_of(first)
        finder = _HeadFinder(head.last)
        checked = check_segment(
            lines, self._key, cursor, name=path.name, index=index, on_record=finder
        )
        if checked.problems or lines.partial or finder.at is None:
            raise ConfigError(f"{path.name} does not match the audit head: run verify")
        chain = Chain(self._key, seq=cursor.seq, prev=cursor.prev)
        if finder.at != checked.records - 1:
            # A crash between a line and its head: the extra lines are signed, so keep them.
            logger.warning(
                "audit store %s: the head lagged its last segment; repaired", self.folder
            )
            fixed = Head(chain.seq - 1, chain.prev, index, path.name, checked.sealed, head.log_id)
            write_head(self.folder, self._key, fixed, fsync=self.fsync)
        if checked.sealed:
            self._index = index + 1
            return chain
        self._index, self._name = index, path.name
        self._day = path.name.split("-")[1]
        self._size = path.stat().st_size
        self._events = checked.events
        self._file = path.open("ab")
        return chain


class _HeadFinder:
    """Notes which record of a segment the head names."""

    def __init__(self, last: str) -> None:
        self.last = last
        self.at: int | None = None
        self._seen = 0

    def __call__(self, record: dict[str, Any], line: bytes) -> None:
        if chain_hash(line) == self.last:
            self.at = self._seen
        self._seen += 1


def _start_of(line: bytes) -> Cursor:
    """The cursor a segment's first record claims (its MAC is checked next)."""
    try:
        first = json.loads(line)
        return Cursor(int(first["seq"]), str(first["prev"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(
            "the last audit segment starts with a damaged record: run verify"
        ) from exc


def close_all() -> None:
    """Seal every open store (runs at interpreter exit); a failure is logged, never raised."""
    for store in list(_OPEN):
        try:
            store.close()
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "audit store %s not sealed", store.folder)


atexit.register(close_all)
