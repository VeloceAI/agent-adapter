"""SQLiteBudgetStore: budget counters in one SQLite file, shared by every process on a host."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from typing import Any, TypeVar

from agent_adapter.core.errors import ConfigError, IntegrationError, log_failure
from agent_adapter.hooks.builtin.budgets.store import (
    BudgetUsage,
    Reserved,
    admits,
    check_ids,
    check_reserve,
    check_usd,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
SCHEMA_VERSION = 1
COUNTERS = "aa_budget_counters"
HOLDS = "aa_budget_holds"
META = "aa_budget_meta"
_SCHEMA = (
    f"CREATE TABLE {COUNTERS} (key TEXT PRIMARY KEY, spent REAL NOT NULL, expire_at REAL)",
    f"CREATE INDEX {COUNTERS}_expiry ON {COUNTERS} (expire_at)",
    f"CREATE TABLE {HOLDS} (key TEXT NOT NULL, reservation_id TEXT NOT NULL,"
    " amount REAL NOT NULL, until REAL NOT NULL, PRIMARY KEY (key, reservation_id))",
    f"CREATE INDEX {HOLDS}_until ON {HOLDS} (until)",
)
_RESERVED = (":memory:", "")


class SQLiteBudgetStore:
    """A `BudgetStore` in a SQLite file (WAL), safe across threads, loops and processes.

    Each call runs in a worker thread on its own connection. `reserve` and `settle` hold
    `BEGIN IMMEDIATE`, so the check and the hold are one step for every process; `timeout`
    is how long one waits for the file lock (seconds). Ended windows and expired holds are
    deleted as new reservations arrive.

    Raises:
        ConfigError: a bad `path` or `timeout` (here); on first use, a database without WAL
            support or with another schema version.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        timeout: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(path, str | os.PathLike):
            raise ConfigError(f"path must be a str or os.PathLike, got {type(path).__name__}")
        self.path = os.fspath(path)
        if self.path in _RESERVED or self.path.startswith("file:"):
            raise ConfigError("SQLiteBudgetStore needs a file path (not :memory:, '' or a URI)")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigError(f"timeout must be a finite number > 0, got {timeout!r}")
        self.timeout = float(timeout)
        self._clock = clock
        self._ready = False
        self._ready_lock = threading.Lock()

    async def reserve(
        self,
        key: str,
        reservation_id: str,
        amount_usd: float,
        *,
        limit_usd: float | None,
        hold_s: float,
        expire_at: float | None,
    ) -> Reserved:
        """Hold `amount_usd` when it fits `limit_usd` (None: always), under the file lock."""
        check_reserve(key, reservation_id, amount_usd, hold_s)
        args = (key, reservation_id, float(amount_usd), limit_usd, hold_s, expire_at)
        return await self._run("reserve", self._reserve, *args)

    async def settle(
        self, key: str, reservation_id: str, cost_usd: float, *, expire_at: float | None
    ) -> None:
        """Drop the hold and add the real cost (even when the hold already expired)."""
        check_ids(key, reservation_id)
        check_usd(cost_usd, "cost_usd")
        await self._run("settle", self._settle, key, reservation_id, float(cost_usd), expire_at)

    async def release(self, key: str, reservation_id: str) -> None:
        """Drop the hold; nothing was spent."""
        check_ids(key, reservation_id)
        await self._run("release", self._release, key, reservation_id)

    async def usage(self, key: str) -> BudgetUsage:
        """The counter's spend and live holds; zero for an unknown or ended key."""
        return await self._run("usage", self._usage, key)

    # --- worker-thread side ----------------------------------------------------------------

    async def _run(self, op: str, fn: Callable[..., T], *args: Any) -> T:
        try:
            return await asyncio.to_thread(self._call, fn, *args)
        except (sqlite3.Error, OSError) as exc:
            log_failure(logger, logging.ERROR, exc, "sqlite budget store %s failed", op)
            raise

    def _call(self, fn: Callable[..., T], *args: Any) -> T:
        self._ensure_schema()
        with closing(self._connect()) as conn:
            return fn(conn, *args)

    def _connect(self) -> sqlite3.Connection:
        # Autocommit: transactions are only the explicit BEGIN blocks below.
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

    def _reserve(
        self,
        conn: sqlite3.Connection,
        key: str,
        rid: str,
        amount: float,
        limit: float | None,
        hold_s: float,
        expire_at: float | None,
    ) -> Reserved:
        now = self._clock()
        with _transaction(conn, "BEGIN IMMEDIATE"):
            conn.execute(f"DELETE FROM {HOLDS} WHERE until <= ?", (now,))
            ended = conn.execute(
                f"SELECT key FROM {COUNTERS} WHERE expire_at <= ?", (now,)
            ).fetchall()
            conn.executemany(f"DELETE FROM {HOLDS} WHERE key = ?", ended)
            conn.execute(f"DELETE FROM {COUNTERS} WHERE expire_at <= ?", (now,))
            usage = _read(conn, key, now)
            if not admits(usage, amount, limit):
                return Reserved(False, usage)
            _upsert(conn, key, 0.0, expire_at)
            conn.execute(
                f"INSERT OR REPLACE INTO {HOLDS} (key, reservation_id, amount, until)"
                " VALUES (?, ?, ?, ?)",
                (key, rid, amount, now + hold_s),
            )
            return Reserved(True, usage)

    def _settle(
        self,
        conn: sqlite3.Connection,
        key: str,
        rid: str,
        cost: float,
        expire_at: float | None,
    ) -> None:
        now = self._clock()
        with _transaction(conn, "BEGIN IMMEDIATE"):
            conn.execute(f"DELETE FROM {HOLDS} WHERE key = ? AND reservation_id = ?", (key, rid))
            conn.execute(f"DELETE FROM {COUNTERS} WHERE key = ? AND expire_at <= ?", (key, now))
            _upsert(conn, key, cost, expire_at)

    def _release(self, conn: sqlite3.Connection, key: str, rid: str) -> None:
        conn.execute(f"DELETE FROM {HOLDS} WHERE key = ? AND reservation_id = ?", (key, rid))

    def _usage(self, conn: sqlite3.Connection, key: str) -> BudgetUsage:
        now = self._clock()
        with _transaction(conn, "BEGIN"):  # one snapshot for both reads
            row = conn.execute(f"SELECT expire_at FROM {COUNTERS} WHERE key = ?", (key,)).fetchone()
            if row is not None and row[0] is not None and row[0] <= now:
                return BudgetUsage()
            return _read(conn, key, now)


def _read(conn: sqlite3.Connection, key: str, now: float) -> BudgetUsage:
    row = conn.execute(f"SELECT spent FROM {COUNTERS} WHERE key = ?", (key,)).fetchone()
    spent = _number(row[0] if row is not None else 0.0, "spent")
    held = conn.execute(
        f"SELECT COALESCE(SUM(amount), 0) FROM {HOLDS} WHERE key = ? AND until > ?", (key, now)
    ).fetchone()
    return BudgetUsage(spent, _number(held[0], "held"))


def _upsert(conn: sqlite3.Connection, key: str, add: float, expire_at: float | None) -> None:
    conn.execute(
        f"INSERT INTO {COUNTERS} (key, spent, expire_at) VALUES (?, ?, ?)"
        " ON CONFLICT(key) DO UPDATE SET spent = spent + excluded.spent,"
        " expire_at = COALESCE(excluded.expire_at, expire_at)",
        (key, add, expire_at),
    )


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        logger.error("budget store %s value is not a finite number", name)
        raise IntegrationError(f"budget store {name} is not a finite number", integration="sqlite")
    return float(value)


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
    """Create the database file owner-only (0600) unless it exists; POSIX SQLite gives
    its -wal and -shm files the same mode. Windows keeps the folder's ACL."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return  # an operator's or another process's file keeps its mode
    os.close(fd)


def _migrate(conn: sqlite3.Connection) -> None:
    """Switch the file to WAL and create (or check) schema version `SCHEMA_VERSION`."""
    (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if str(mode).lower() != "wal":
        raise ConfigError(f"SQLiteBudgetStore needs WAL journal mode; the file reports {mode!r}")
    with _transaction(conn, "BEGIN IMMEDIATE"):
        conn.execute(f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value)")
        row = conn.execute(f"SELECT value FROM {META} WHERE key = 'schema'").fetchone()
        if row is not None:
            if row[0] != SCHEMA_VERSION:
                raise ConfigError(
                    f"budget database schema {row[0]!r} is not {SCHEMA_VERSION}; "
                    "use the agent_adapter version that wrote it, or a new file"
                )
            return
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.execute(f"INSERT INTO {META} (key, value) VALUES ('schema', ?)", (SCHEMA_VERSION,))
