"""SQLiteCheckpointStore: checkpoints in one SQLite file, shared by every process on one host."""

from __future__ import annotations

import asyncio
import builtins
import logging
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from typing import Any, TypeVar

from agent_adapter.core.errors import CheckpointStoreCorrupt, ConfigError, log_failure
from agent_adapter.exec.checkpoints._port import (
    CallRecord,
    Checkpoint,
    StoredCheckpoint,
    check_id,
    check_limit,
    check_save,
    conflict,
    decode_call,
    encode_call,
    encode_checkpoint,
    stale_segment,
    stored,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
SCHEMA_VERSION = 1
RUNS = "aa_checkpoints"
CALLS = "aa_checkpoint_calls"
META = "aa_checkpoint_meta"
_SCHEMA = (
    f"CREATE TABLE {RUNS} ("
    " run_id BLOB PRIMARY KEY,"
    " step INTEGER NOT NULL,"
    " name TEXT NOT NULL,"
    " status TEXT NOT NULL,"
    " segment BLOB NOT NULL,"
    " saved_at REAL NOT NULL,"
    " record TEXT NOT NULL)",
    f"CREATE INDEX {RUNS}_status ON {RUNS} (status, saved_at)",
    f"CREATE TABLE {CALLS} ("
    " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
    " run_id BLOB NOT NULL,"
    " segment BLOB NOT NULL,"
    " call_key BLOB NOT NULL,"
    " record TEXT NOT NULL,"
    " UNIQUE (run_id, segment, call_key))",
)
_RESERVED = (":memory:", "")


class SQLiteCheckpointStore:
    """CheckpointStore in a SQLite file (WAL), safe across threads, loops and processes.

    Each call runs in a worker thread on its own connection; writes hold `BEGIN IMMEDIATE`,
    so a step's compare-and-set is atomic for every process; `timeout` is the lock wait (s).

    Raises:
        ConfigError: a bad `path` or `timeout` (here); on first use, a database without WAL
            support, foreign tables with these names, or another schema version.
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5.0) -> None:
        if not isinstance(path, str | os.PathLike):
            raise ConfigError(f"path must be a str or os.PathLike, got {type(path).__name__}")
        self.path = os.fspath(path)
        if self.path in _RESERVED or self.path.startswith("file:"):
            raise ConfigError("SQLiteCheckpointStore needs a file path (not :memory:, '' or a URI)")
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

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        """Store the run's next checkpoint (compare-and-set on `step`).

        Raises:
            CheckpointConflict: `step` is not 0 for a new run or one past the stored step.
            ToolUsageError: a bad run id, step or checkpoint.
        """
        check_save(run_id, step, checkpoint)
        if not checkpoint.saved_at:
            checkpoint = checkpoint.model_copy(update={"saved_at": time.time()})
        await self._run("save", self._save, run_id, step, checkpoint)

    async def latest(self, run_id: str) -> StoredCheckpoint | None:
        """The run's latest checkpoint, or None.

        Raises:
            CheckpointStoreCorrupt: the stored row is unreadable or disagrees with itself.
        """
        check_id(run_id, "run_id")
        return await self._run("latest", self._latest, run_id)

    async def list_unfinished(
        self, *, name: str | None = None, limit: int = 100
    ) -> builtins.list[StoredCheckpoint]:
        """Runs whose latest checkpoint is `running`, oldest first, optionally of one name.

        Raises:
            ConfigError: a negative `limit`.
        """
        check_limit(limit)
        return await self._run("list", self._list, name, limit)

    async def delete(self, run_id: str) -> bool:
        """Forget the run and its call log; False when nothing was stored."""
        check_id(run_id, "run_id")
        return await self._run("delete", self._delete, run_id)

    async def record_call(self, run_id: str, segment: str, call: CallRecord) -> None:
        """Keep a call record in the run's current segment (latest record per key wins).

        Raises:
            CheckpointConflict: `segment` is not the run's current one.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        check_id(call.key, "call key")
        await self._run("record_call", self._record_call, run_id, segment, call, encode_call(call))

    async def calls(self, run_id: str, segment: str) -> builtins.list[CallRecord]:
        """The call records of one segment, in the order first written.

        Raises:
            CheckpointStoreCorrupt: a stored record is unreadable or has another key.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        return await self._run("calls", self._calls, run_id, segment)

    # --- worker-thread side ----------------------------------------------------------------

    async def _run(self, op: str, fn: Callable[..., T], *args: Any) -> T:
        try:
            return await asyncio.to_thread(self._call, fn, *args)
        except (sqlite3.Error, OSError) as exc:
            log_failure(logger, logging.ERROR, exc, "sqlite checkpoint store %s failed", op)
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

    def _save(
        self, conn: sqlite3.Connection, run_id: str, step: int, checkpoint: Checkpoint
    ) -> None:
        run = _key(run_id)
        with _immediate(conn):
            row = conn.execute(f"SELECT step FROM {RUNS} WHERE run_id = ?", (run,)).fetchone()
            have = None if row is None else row[0]
            if have is not None and (isinstance(have, bool) or not isinstance(have, int)):
                raise _corrupt(run_id)
            if step != (0 if have is None else have + 1):
                raise conflict(run_id, step, have)
            conn.execute(
                f"INSERT OR REPLACE INTO {RUNS} (run_id, step, name, status, segment, saved_at,"
                " record) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run,
                    step,
                    checkpoint.name,
                    checkpoint.status,
                    _key(checkpoint.segment),
                    checkpoint.saved_at,
                    encode_checkpoint(checkpoint),
                ),
            )
            conn.execute(
                f"DELETE FROM {CALLS} WHERE run_id = ? AND segment <> ?",
                (run, _key(checkpoint.segment)),
            )

    def _latest(self, conn: sqlite3.Connection, run_id: str) -> StoredCheckpoint | None:
        row = conn.execute(
            f"SELECT run_id, step, status, record FROM {RUNS} WHERE run_id = ?", (_key(run_id),)
        ).fetchone()
        return None if row is None else stored(row[0], row[1], row[3], row[2])

    def _list(
        self, conn: sqlite3.Connection, name: str | None, limit: int
    ) -> builtins.list[StoredCheckpoint]:
        query = f"SELECT run_id, step, status, record FROM {RUNS} WHERE status = 'running'"
        params: tuple[Any, ...] = ()
        if name is not None:
            query += " AND name = ?"
            params = (name,)
        rows = conn.execute(f"{query} ORDER BY saved_at, run_id LIMIT ?", (*params, limit))
        return [stored(r[0], r[1], r[3], r[2]) for r in rows.fetchall()]

    def _delete(self, conn: sqlite3.Connection, run_id: str) -> bool:
        with _immediate(conn):
            conn.execute(f"DELETE FROM {CALLS} WHERE run_id = ?", (_key(run_id),))
            gone = conn.execute(f"DELETE FROM {RUNS} WHERE run_id = ?", (_key(run_id),))
            return gone.rowcount == 1

    def _record_call(
        self, conn: sqlite3.Connection, run_id: str, segment: str, call: CallRecord, text: str
    ) -> None:
        run, seg = _key(run_id), _key(segment)
        with _immediate(conn):
            row = conn.execute(f"SELECT segment FROM {RUNS} WHERE run_id = ?", (run,)).fetchone()
            if row is None or row[0] != seg:
                raise stale_segment(run_id)
            conn.execute(
                f"INSERT INTO {CALLS} (run_id, segment, call_key, record) VALUES (?, ?, ?, ?)"
                " ON CONFLICT (run_id, segment, call_key) DO UPDATE SET record = excluded.record",
                (run, seg, _key(call.key), text),
            )

    def _calls(
        self, conn: sqlite3.Connection, run_id: str, segment: str
    ) -> builtins.list[CallRecord]:
        rows = conn.execute(
            f"SELECT call_key, record FROM {CALLS} WHERE run_id = ? AND segment = ? ORDER BY seq",
            (_key(run_id), _key(segment)),
        ).fetchall()
        return [_call_record(run_id, raw_key, raw) for raw_key, raw in rows]


def _call_record(run_id: str, raw_key: object, raw: object) -> CallRecord:
    """A stored call; its key column must be the record's own key."""
    record = decode_call(raw, repr(run_id[:64]))
    if not isinstance(raw_key, bytes) or raw_key != _key(record.key):
        logger.error("checkpoint call row of run %r disagrees with its record", run_id[:64])
        raise CheckpointStoreCorrupt(f"a call row of run {run_id[:64]!r} disagrees with its record")
    return record


def _key(text: str) -> bytes:
    return text.encode("utf-8", "surrogatepass")


def _corrupt(run_id: str) -> CheckpointStoreCorrupt:
    logger.error("checkpoint row of run %r has a step that is not an integer", run_id[:64])
    return CheckpointStoreCorrupt(f"the checkpoint row of run {run_id[:64]!r} is unreadable")


@contextmanager
def _immediate(conn: sqlite3.Connection) -> Iterator[None]:
    """One write transaction that takes the database lock before it reads."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _create_private(path: str) -> None:
    """Create the database file owner-only (0600) unless it exists (Windows: folder ACL)."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.close(fd)


def _migrate(conn: sqlite3.Connection) -> None:
    """Switch the file to WAL and create (or check) schema version `SCHEMA_VERSION`."""
    (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if str(mode).lower() != "wal":
        raise ConfigError(
            f"SQLiteCheckpointStore needs WAL journal mode; the file reports {mode!r}"
        )
    with _immediate(conn):
        conn.execute(f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value)")
        row = conn.execute(f"SELECT value FROM {META} WHERE key = 'schema'").fetchone()
        if row is not None:
            if row[0] != SCHEMA_VERSION:
                raise ConfigError(
                    f"checkpoint database schema {row[0]!r} is not {SCHEMA_VERSION}; "
                    "use the agent_adapter version that wrote it, or a new file"
                )
            return
        clash = conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN (?, ?, ?)",
            (RUNS, CALLS, f"{RUNS}_status"),
        ).fetchone()
        if clash is not None:
            raise ConfigError(
                f"the database already has {clash[0]!r} without checkpoint store metadata"
            )
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.execute(f"INSERT INTO {META} (key, value) VALUES ('schema', ?)", (SCHEMA_VERSION,))
    logger.info("checkpoint database created (schema %d)", SCHEMA_VERSION)
