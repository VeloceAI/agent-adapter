"""SQLiteApprovalStore: approvals in one SQLite file, shared by every process on one host."""

from __future__ import annotations

import asyncio
import builtins
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from typing import Any, TypeVar

from agent_adapter.approval.persist import (
    ApprovalStoreCorrupt,
    check_list,
    decode,
    encode,
    filled,
    key,
    read_row,
    shown,
    stamped,
)
from agent_adapter.approval.store import ApprovalRecord, ApprovalStatus
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")
SCHEMA_VERSION = 1
TABLE = "aa_approvals"
META = "aa_approval_meta"
# Eviction tiers, lowest first: consumed, then decided, then pending (as MemoryApprovalStore).
_CONSUMED, _DECIDED, _PENDING = 0, 1, 2
_COLUMNS = "approval_id, run_id, call_id, status, consumed, record"
_SCHEMA = (
    f"CREATE TABLE {TABLE} ("
    " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
    " approval_id BLOB NOT NULL UNIQUE,"
    " run_id BLOB NOT NULL,"
    " call_id BLOB NOT NULL,"
    " status TEXT NOT NULL,"
    " consumed INTEGER NOT NULL,"
    " args_digest TEXT NOT NULL,"
    " tier INTEGER NOT NULL,"
    " tier_seq INTEGER NOT NULL,"
    " record TEXT NOT NULL)",
    f"CREATE UNIQUE INDEX {TABLE}_open ON {TABLE} (run_id, call_id) WHERE consumed = 0",
    f"CREATE INDEX {TABLE}_run ON {TABLE} (run_id, seq)",
    f"CREATE INDEX {TABLE}_status ON {TABLE} (status, seq)",
    f"CREATE INDEX {TABLE}_evict ON {TABLE} (tier, tier_seq)",
)
_RESERVED = (":memory:", "")
_EVICTED = {  # keyed by "still pending": a lost decision or a lost request, both loud
    False: "evicted decided approval %s before it was used (store full at %d records)",
    True: "evicted pending approval %s (store full at %d records)",
}


class SQLiteApprovalStore:
    """ApprovalStore in a SQLite file (WAL), safe across threads, loops and processes.

    Every call runs in a worker thread (`asyncio.to_thread`) on its own connection.
    `create`, `decide` and `consume` hold `BEGIN IMMEDIATE`, so writers on every process
    are serialised; `timeout` is how long one waits for the file lock (seconds). Past
    `max_records` it evicts like `MemoryApprovalStore`: the earliest consumed record,
    else the oldest decided, else the oldest pending (WARNING).

    Raises:
        ConfigError: a bad `path`, `max_records` or `timeout` (here); on first use, a
            database without WAL support, a foreign `aa_approvals` table or another schema.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, max_records: int = 10_000, timeout: float = 5.0
    ) -> None:
        if not isinstance(path, str | os.PathLike):
            raise ConfigError(f"path must be a str or os.PathLike, got {type(path).__name__}")
        self.path = os.fspath(path)
        if self.path in _RESERVED or self.path.startswith("file:"):
            raise ConfigError("SQLiteApprovalStore needs a file path (not :memory:, '' or a URI)")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ConfigError(f"max_records must be a positive int, got {max_records!r}")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigError(f"timeout must be a finite number > 0, got {timeout!r}")
        self.max_records = max_records
        self.timeout = float(timeout)
        self._ready = False
        self._ready_lock = threading.Lock()

    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        """Store a pending record (empty id / `created_at` filled in), or return the open one.

        Raises:
            ToolUsageError: another call's record already has this approval id, or the
                record holds a value JSON cannot carry.
        """
        record = ApprovalRecord(request=filled(request), status="pending")
        text = encode(record)
        found, created = await self._run("create", self._create, record, text)
        if created:
            logger.debug("approval %s created for tool %r", found.approval_id, request.tool)
        return found

    async def get(self, approval_id: str) -> ApprovalRecord:
        """The record. Raises: ApprovalNotFound."""
        return await self._run("get", self._get, approval_id)

    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        """The unconsumed record of the call, or None."""
        return await self._run("find", self._find, run_id, call_id)

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> ApprovalRecord:
        """Record the decision; only the first decide of a record succeeds.

        Raises:
            ApprovalNotFound: unknown id.
            ApprovalAlreadyDecided: the record is no longer pending.
            ToolUsageError: `edited_args` is not plain JSON.
        """
        decision = stamped(decision)
        status: ApprovalStatus = "approved" if decision.approved else "denied"
        record = await self._run("decide", self._decide, approval_id, decision, status)
        logger.info("approval %s %s", approval_id, status)
        return record

    async def consume(self, approval_id: str) -> bool:
        """Mark a decided record used; True only for the caller that flipped it.

        Raises:
            ApprovalNotFound: unknown id.
        """
        return await self._run("consume", self._consume, approval_id)

    async def list(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]:
        """Records newest first, filtered by run and status.

        Raises:
            ConfigError: unknown `status` or a negative `limit`.
        """
        check_list(status, limit)
        return await self._run("list", self._list, run_id, status, limit)

    # --- worker-thread side ----------------------------------------------------------------

    async def _run(self, op: str, fn: Callable[..., T], *args: Any) -> T:
        try:
            return await asyncio.to_thread(self._call, fn, *args)
        except (sqlite3.Error, OSError) as exc:
            log_failure(logger, logging.ERROR, exc, "sqlite approval store %s failed", op)
            raise

    def _call(self, fn: Callable[..., T], *args: Any) -> T:
        self._ensure_schema()
        with closing(self._connect()) as conn:
            return fn(conn, *args)

    def _connect(self) -> sqlite3.Connection:
        # Autocommit: transactions are only the explicit BEGIN IMMEDIATE blocks below.
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

    def _create(
        self, conn: sqlite3.Connection, record: ApprovalRecord, text: str
    ) -> tuple[ApprovalRecord, bool]:
        request = record.request
        with _immediate(conn):
            open_record = self._find(conn, request.run_id, request.call_id)
            if open_record is not None:
                return open_record, False
            taken = conn.execute(
                f"SELECT 1 FROM {TABLE} WHERE approval_id = ?", (key(request.approval_id),)
            ).fetchone()
            if taken is not None:
                raise ToolUsageError(f"approval {request.approval_id!r} already exists")
            conn.execute(
                f"INSERT INTO {TABLE} (approval_id, run_id, call_id, status, consumed,"
                " args_digest, tier, tier_seq, record) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)",
                (
                    key(request.approval_id),
                    key(request.run_id),
                    key(request.call_id),
                    record.args_digest,
                    _PENDING,
                    _tick(conn),
                    text,
                ),
            )
            self._evict(conn)
        return decode(text, shown(request.approval_id)), True

    def _get(self, conn: sqlite3.Connection, approval_id: str) -> ApprovalRecord:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM {TABLE} WHERE approval_id = ?", (key(approval_id),)
        ).fetchone()
        if row is None:
            raise ApprovalNotFound(approval_id)
        return _record(row)

    def _find(self, conn: sqlite3.Connection, run_id: str, call_id: str) -> ApprovalRecord | None:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM {TABLE} WHERE run_id = ? AND call_id = ? AND consumed = 0",
            (key(run_id), key(call_id)),
        ).fetchone()
        return None if row is None else _record(row)

    def _decide(
        self,
        conn: sqlite3.Connection,
        approval_id: str,
        decision: ApprovalDecision,
        status: ApprovalStatus,
    ) -> ApprovalRecord:
        with _immediate(conn):
            record = self._get(conn, approval_id)
            if record.status != "pending":
                raise ApprovalAlreadyDecided(approval_id, status=record.status)
            record = record.model_copy(update={"status": status, "decision": decision})
            conn.execute(
                f"UPDATE {TABLE} SET status = ?, tier = ?, tier_seq = ?, record = ?"
                " WHERE approval_id = ?",
                (status, _DECIDED, _tick(conn), encode(record), key(approval_id)),
            )
        return record

    def _consume(self, conn: sqlite3.Connection, approval_id: str) -> bool:
        with _immediate(conn):
            record = self._get(conn, approval_id)
            if record.consumed or record.status == "pending":
                return False
            text = encode(record.model_copy(update={"consumed": True}))
            done = conn.execute(
                f"UPDATE {TABLE} SET consumed = 1, tier = ?, tier_seq = ?, record = ?"
                " WHERE approval_id = ? AND consumed = 0 AND status <> 'pending'",
                (_CONSUMED, _tick(conn), text, key(approval_id)),
            )
            return done.rowcount == 1

    def _list(
        self, conn: sqlite3.Connection, run_id: str | None, status: str | None, limit: int
    ) -> builtins.list[ApprovalRecord]:
        where: builtins.list[str] = []
        params: builtins.list[object] = []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(key(run_id))
        if status is not None:
            where.append("status = ?")
            params.append(status)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM {TABLE}{clause} ORDER BY seq DESC LIMIT ?", (*params, limit)
        ).fetchall()
        return [_record(row) for row in rows]

    def _evict(self, conn: sqlite3.Connection) -> None:
        # Caller holds BEGIN IMMEDIATE.
        (count,) = conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()
        excess = count - self.max_records
        if excess <= 0:
            return
        victims = conn.execute(
            f"SELECT seq, approval_id, status, tier FROM {TABLE} ORDER BY tier, tier_seq LIMIT ?",
            (excess,),
        ).fetchall()
        conn.executemany(f"DELETE FROM {TABLE} WHERE seq = ?", [(row[0],) for row in victims])
        for _, approval_id, status, tier in victims:
            if tier != _CONSUMED:
                logger.warning(_EVICTED[status == "pending"], shown(approval_id), self.max_records)


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
    """Create the database file owner-only (0600) unless it exists; SQLite's -wal and -shm
    files copy its mode. Windows ignores the mode: the file takes its folder's ACL."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return  # an operator's or another process's file keeps its mode
    os.close(fd)


def _tick(conn: sqlite3.Connection) -> int:
    # Caller holds BEGIN IMMEDIATE: the counter orders decides and consumes for eviction.
    conn.execute(f"UPDATE {META} SET value = value + 1 WHERE key = 'tick'")
    row = conn.execute(f"SELECT value FROM {META} WHERE key = 'tick'").fetchone()
    if row is None or isinstance(row[0], bool) or not isinstance(row[0], int):
        logger.error("approval store meta row 'tick' is missing or not an integer")
        raise ApprovalStoreCorrupt("approval store meta row 'tick' is missing or not an integer")
    return row[0]


def _record(row: tuple[Any, ...]) -> ApprovalRecord:
    approval_id, run_id, call_id, status, consumed, raw = row
    if not isinstance(approval_id, bytes):  # a foreign writer's TEXT id
        approval_id = None
    return read_row(
        raw,
        approval_id=approval_id,
        run_id=run_id if isinstance(run_id, bytes) else None,
        call_id=call_id if isinstance(call_id, bytes) else None,
        status=status,
        consumed=consumed,
    )


def _migrate(conn: sqlite3.Connection) -> None:
    """Switch the file to WAL and create (or check) schema version `SCHEMA_VERSION`."""
    (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if str(mode).lower() != "wal":
        raise ConfigError(f"SQLiteApprovalStore needs WAL journal mode; the file reports {mode!r}")
    with _immediate(conn):
        conn.execute(f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value)")
        row = conn.execute(f"SELECT value FROM {META} WHERE key = 'schema'").fetchone()
        if row is not None:
            if row[0] != SCHEMA_VERSION:
                raise ConfigError(
                    f"approval database schema {row[0]!r} is not {SCHEMA_VERSION}; "
                    "use the agent_adapter version that wrote it, or a new file"
                )
            return
        names = {TABLE, *(f"{TABLE}_{suffix}" for suffix in ("open", "run", "status", "evict"))}
        clash = conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN (?, ?, ?, ?, ?)", tuple(sorted(names))
        ).fetchone()
        if clash is not None:
            raise ConfigError(
                f"the database already has {clash[0]!r} without approval store metadata"
            )
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.executemany(
            f"INSERT INTO {META} (key, value) VALUES (?, ?)",
            [("schema", SCHEMA_VERSION), ("tick", 0)],
        )
    logger.info("approval database created (schema %d)", SCHEMA_VERSION)
