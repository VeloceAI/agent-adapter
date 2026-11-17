"""SQLiteBoardStore: boards in one SQLite file, shared by every process on a host."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from typing import Any, TypeVar

from agent_adapter.board.entries import Entry, dumps, loads
from agent_adapter.core.errors import ConfigError, IntegrationError, log_failure
from agent_adapter.runtime._sqlite_wal import enable_wal

logger = logging.getLogger(__name__)

T = TypeVar("T")
SCHEMA_VERSION = 1
ENTRIES = "aa_board_entries"
CLAIMS = "aa_board_claims"
META = "aa_board_meta"
_SCHEMA = (
    f"CREATE TABLE {ENTRIES} (scope TEXT NOT NULL, seq INTEGER NOT NULL, size INTEGER NOT NULL,"
    " body TEXT NOT NULL, PRIMARY KEY (scope, seq))",
    f"CREATE TABLE {CLAIMS} (scope TEXT NOT NULL, key TEXT NOT NULL, holder TEXT NOT NULL,"
    " PRIMARY KEY (scope, key))",
)
_RESERVED = (":memory:", "")


class SQLiteBoardStore:
    """A `BoardStore` in a SQLite file (WAL), safe across threads, loops and processes.

    Each call runs in a worker thread on its own connection; `append` and `claim` hold
    `BEGIN IMMEDIATE`, so the cap check and the write are one step for every process.
    `timeout` is how long one waits for the file lock (seconds).

    Raises:
        ConfigError: a bad `path` or `timeout` (here); on first use, a database without WAL
            support or with another schema version.
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5.0) -> None:
        if not isinstance(path, str | os.PathLike):
            raise ConfigError(f"path must be a str or os.PathLike, got {type(path).__name__}")
        self.path = os.fspath(path)
        if self.path in _RESERVED or self.path.startswith("file:"):
            raise ConfigError("SQLiteBoardStore needs a file path (not :memory:, '' or a URI)")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigError(f"timeout must be a finite number > 0, got {timeout!r}")
        self.timeout = float(timeout)
        self._ready = False
        self._ready_lock = threading.Lock()

    async def append(
        self, scope: str, entry: Entry, *, max_entries: int, max_bytes: int
    ) -> Entry | None:
        """Add `entry` as the board's next one, or None when a cap would be passed."""
        return await self._run("append", self._append, scope, entry, max_entries, max_bytes)

    async def read(self, scope: str, *, since: int = 0) -> list[Entry]:
        """The board's entries after `seq` `since`, oldest first."""
        return await self._run("read", self._read, scope, since)

    async def claim(self, scope: str, key: str, holder: str) -> tuple[str, bool]:
        """The key's holder: `holder` when it was free (True) or already theirs."""
        return await self._run("claim", self._claim, scope, key, holder)

    async def drop(self, scope: str) -> None:
        """Forget a board's entries and claims."""
        await self._run("drop", self._drop, scope)

    # --- worker-thread side ----------------------------------------------------------------

    async def _run(self, op: str, fn: Callable[..., T], *args: Any) -> T:
        try:
            return await asyncio.to_thread(self._call, fn, *args)
        except (sqlite3.Error, OSError) as exc:
            log_failure(logger, logging.ERROR, exc, "sqlite board store %s failed", op)
            raise

    def _call(self, fn: Callable[..., T], *args: Any) -> T:
        self._ensure_schema()
        with closing(self._connect()) as conn:
            return fn(conn, *args)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        try:
            conn.execute("PRAGMA synchronous=FULL")
        except BaseException:
            conn.close()
            raise
        return conn

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._ready_lock:
            if not self._ready:
                _create_private(self.path)
                with closing(self._connect()) as conn:
                    _migrate(conn)
                self._ready = True

    def _append(
        self, conn: sqlite3.Connection, scope: str, entry: Entry, max_entries: int, max_bytes: int
    ) -> Entry | None:
        size = entry.size()
        with _transaction(conn, "BEGIN IMMEDIATE"):
            count, used = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(size), 0) FROM {ENTRIES} WHERE scope = ?",
                (scope,),
            ).fetchone()
            if count >= max_entries or used + size > max_bytes:
                return None
            stored = dataclasses.replace(entry, seq=count + 1)
            conn.execute(
                f"INSERT INTO {ENTRIES} (scope, seq, size, body) VALUES (?, ?, ?, ?)",
                (scope, stored.seq, size, dumps(stored)),
            )
            return stored

    def _read(self, conn: sqlite3.Connection, scope: str, since: int) -> list[Entry]:
        rows = conn.execute(
            f"SELECT body FROM {ENTRIES} WHERE scope = ? AND seq > ? ORDER BY seq",
            (scope, since),
        ).fetchall()
        return [_entry(body) for (body,) in rows]

    def _claim(
        self, conn: sqlite3.Connection, scope: str, key: str, holder: str
    ) -> tuple[str, bool]:
        with _transaction(conn, "BEGIN IMMEDIATE"):
            made = conn.execute(
                f"INSERT OR IGNORE INTO {CLAIMS} (scope, key, holder) VALUES (?, ?, ?)",
                (scope, key, holder),
            ).rowcount
            (found,) = conn.execute(
                f"SELECT holder FROM {CLAIMS} WHERE scope = ? AND key = ?", (scope, key)
            ).fetchone()
            return str(found), made == 1

    def _drop(self, conn: sqlite3.Connection, scope: str) -> None:
        with _transaction(conn, "BEGIN IMMEDIATE"):
            conn.execute(f"DELETE FROM {ENTRIES} WHERE scope = ?", (scope,))
            conn.execute(f"DELETE FROM {CLAIMS} WHERE scope = ?", (scope,))


def _entry(body: str) -> Entry:
    try:
        return loads(body)
    except ValueError as exc:
        logger.error("sqlite board store holds an unreadable entry")
        raise IntegrationError(
            "the board store holds an unreadable entry", integration="sqlite"
        ) from exc


@contextmanager
def _transaction(conn: sqlite3.Connection, begin: str) -> Iterator[None]:
    conn.execute(begin)
    try:
        yield
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _create_private(path: str) -> None:
    """Create the database file owner-only (0600) unless it exists."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.close(fd)


def _migrate(conn: sqlite3.Connection) -> None:
    """Switch the file to WAL and create (or check) schema version `SCHEMA_VERSION`."""
    mode = enable_wal(conn)
    if mode != "wal":
        raise ConfigError(f"SQLiteBoardStore needs WAL journal mode; the file reports {mode!r}")
    with _transaction(conn, "BEGIN IMMEDIATE"):
        conn.execute(f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value)")
        row = conn.execute(f"SELECT value FROM {META} WHERE key = 'schema'").fetchone()
        if row is not None:
            if row[0] != SCHEMA_VERSION:
                raise ConfigError(
                    f"board database schema {row[0]!r} is not {SCHEMA_VERSION}; "
                    "use the agent_adapter version that wrote it, or a new file"
                )
            return
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.execute(f"INSERT INTO {META} (key, value) VALUES ('schema', ?)", (SCHEMA_VERSION,))
