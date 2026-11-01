"""Ledger sinks: memory, callback and JSON lines, plus a reader for JSON-lines files."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.types import LedgerEntry

logger = logging.getLogger(__name__)


@runtime_checkable
class LedgerSink(Protocol):
    """Receives every entry the ledger records; may raise (the ledger logs and continues)."""

    def write(self, entry: LedgerEntry) -> None: ...


class MemorySink:
    """Keeps every written entry, duplicates included, in `entries`."""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def write(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


class CallbackSink:
    """Calls `fn(entry)` for every entry, e.g. to insert into a database."""

    def __init__(self, fn: Callable[[LedgerEntry], None]) -> None:
        self._fn = fn

    def write(self, entry: LedgerEntry) -> None:
        self._fn(entry)


class JsonlSink:
    """Appends one JSON line per entry (UTF-8), flushed and closed on every write."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, entry: LedgerEntry) -> None:
        line = entry.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)


def read_jsonl(path: str | os.PathLike[str]) -> list[LedgerEntry]:
    """Entries from a JSON-lines file; the last line per (run_id, entry_id) wins.

    A torn final line (a crash mid-write) is skipped with a WARNING.

    Raises:
        ConfigError: the file cannot be read, or a line other than the last is invalid.
    """
    file = Path(path)
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log_failure(logger, logging.ERROR, exc, "cannot read ledger file %s", file)
        raise ConfigError(f"cannot read ledger file {file}: {exc}") from exc
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    found: dict[tuple[str, str], LedgerEntry] = {}
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = LedgerEntry.model_validate_json(line)
        except ValidationError as exc:
            if number == last:
                logger.warning("skipping incomplete last line %d of %s", number + 1, file)
                break
            logger.error("invalid ledger line %d of %s", number + 1, file)
            raise ConfigError(f"invalid ledger entry at {file} line {number + 1}") from exc
        found[(entry.run_id, entry.entry_id)] = entry
    return list(found.values())
