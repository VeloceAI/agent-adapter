"""SQLAlchemyApprovalStore: approvals in Postgres (or SQLite) through SQLAlchemy 2 async."""

from __future__ import annotations

import asyncio
import builtins
import logging
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from agent_adapter.approval import sqlalchemy_schema as schema
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
    AdapterError,
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime.loop_local import LoopLocal

if TYPE_CHECKING:
    from sqlalchemy import Executable, Table
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

logger = logging.getLogger(__name__)

DIALECTS = frozenset({"postgresql", "sqlite"})  # both have partial indexes, ON CONFLICT, RETURNING
_ATTEMPTS = 3  # a record changes at most twice (decided, consumed)
_COLUMNS = ("approval_id", "run_id", "call_id", "status", "consumed", "record")


class SQLAlchemyApprovalStore:
    """ApprovalStore on a SQLAlchemy async engine (`agent-adapter[sqlalchemy]`), every host.

    `engine` is a URL (`"postgresql+asyncpg://..."`) or a callable returning an
    `AsyncEngine`; one engine is made per event loop (`LoopLocal`). `create` is
    `INSERT ... ON CONFLICT DO NOTHING` against a partial unique index on the call's
    unconsumed row; `decide` and `consume` are one `UPDATE ... WHERE ... RETURNING` each.
    `metadata` holds the table, for Alembic. Backend errors propagate (fail closed).

    Raises:
        ConfigError: a bad `engine` or `table`; on first use, a dialect other than
            Postgres or SQLite, or a table without the store's columns or its partial
            unique index on the call's unconsumed row.
        MissingDependency: SQLAlchemy is not installed.
    """

    def __init__(
        self,
        engine: str | Callable[[], AsyncEngine],
        *,
        table: str = "aa_approvals",
        create_schema: bool = True,
    ) -> None:
        factory = _engine_factory(engine)
        if not isinstance(table, str) or not table.isidentifier():
            raise ConfigError(f"table must be a plain SQL identifier, got {table!r}")
        self.table_name = table
        self.create_schema = create_schema
        self.metadata, self._t = schema.table_schema(table)
        self._engines: LoopLocal[AsyncEngine] = LoopLocal(factory, close=_dispose)
        self._setup_locks: LoopLocal[asyncio.Lock] = LoopLocal(asyncio.Lock)
        self._ready = False

    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        """Store a pending record (empty id / `created_at` filled in), or return the open one.

        Raises:
            ToolUsageError: another call's record already has this approval id, or the
                record holds a value JSON cannot carry.
        """
        record = ApprovalRecord(request=filled(request), status="pending")
        text = encode(record)
        req = record.request
        values = {
            "approval_id": key(req.approval_id),
            "run_id": key(req.run_id),
            "call_id": key(req.call_id),
            "status": "pending",
            "consumed": 0,
            "args_digest": record.args_digest,
            "created_at": req.created_at,
            "record": text,
        }
        for _ in range(_ATTEMPTS):
            engine = await self._engine()
            async with _Logged("create"), engine.begin() as conn:
                inserted = await conn.execute(insert_new(self._t, engine.dialect.name, values))
                if inserted.first() is not None:
                    logger.debug("approval %s created for tool %r", req.approval_id, req.tool)
                    return decode(text, shown(req.approval_id))
                found = await self._one(conn, self._open_query(req.run_id, req.call_id))
                if found is not None:
                    return found
                if await self._one(conn, self._by_id(req.approval_id)) is not None:
                    raise ToolUsageError(f"approval {req.approval_id!r} already exists")
        raise _unsettled(req.approval_id)

    async def get(self, approval_id: str) -> ApprovalRecord:
        """The record. Raises: ApprovalNotFound."""
        found = await self._read("get", self._by_id(approval_id))
        if found is None:
            raise ApprovalNotFound(approval_id)
        return found

    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        """The unconsumed record of the call, or None."""
        return await self._read("find", self._open_query(run_id, call_id))

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> ApprovalRecord:
        """Record the decision; only the first decide of a record succeeds.

        Raises:
            ApprovalNotFound: unknown id.
            ApprovalAlreadyDecided: the record is no longer pending.
            ToolUsageError: `edited_args` is not plain JSON.
        """
        decision = stamped(decision)
        status: ApprovalStatus = "approved" if decision.approved else "denied"
        for _ in range(_ATTEMPTS):
            record = await self.get(approval_id)
            if record.status != "pending":
                raise ApprovalAlreadyDecided(approval_id, status=record.status)
            record = record.model_copy(update={"status": status, "decision": decision})
            change = {"status": status, "record": encode(record)}
            if await self._write("decide", decide_update(self._t, key(approval_id), change)):
                logger.info("approval %s %s", approval_id, status)
                return record
        raise _unsettled(approval_id)

    async def consume(self, approval_id: str) -> bool:
        """Mark a decided record used; True only for the caller that flipped it.

        Raises:
            ApprovalNotFound: unknown id.
        """
        record = await self.get(approval_id)
        if record.consumed or record.status == "pending":
            return False
        change = {"consumed": 1, "record": encode(record.model_copy(update={"consumed": True}))}
        return await self._write("consume", consume_update(self._t, key(approval_id), change))

    async def list(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]:
        """Records newest first, filtered by run and status.

        Raises:
            ConfigError: unknown `status` or a negative `limit`.
        """
        check_list(status, limit)
        from sqlalchemy import select

        t = self._t
        query = select(*(t.c[name] for name in _COLUMNS)).order_by(t.c.seq.desc()).limit(limit)
        if run_id is not None:
            query = query.where(t.c.run_id == key(run_id))
        if status is not None:
            query = query.where(t.c.status == status)
        engine = await self._engine()
        async with _Logged("list"), engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [_record(tuple(row)) for row in rows]

    async def prune(self, *, older_than_s: float) -> int:
        """Delete records created more than `older_than_s` seconds ago; the count deleted.

        Raises:
            ConfigError: `older_than_s` is not a number >= 0.
        """
        if isinstance(older_than_s, bool) or not isinstance(older_than_s, int | float):
            raise ConfigError(f"older_than_s must be a number >= 0, got {older_than_s!r}")
        if not older_than_s >= 0:
            raise ConfigError(f"older_than_s must be a number >= 0, got {older_than_s!r}")
        from sqlalchemy import delete, func, select

        t = self._t
        cutoff = time.time() - older_than_s
        engine = await self._engine()
        async with _Logged("prune"), engine.begin() as conn:
            open_query = select(func.count()).where(t.c.created_at < cutoff, t.c.consumed == 0)
            unused = (await conn.execute(open_query)).scalar_one()
            gone = await conn.execute(delete(t).where(t.c.created_at < cutoff))
        if unused:
            logger.warning("pruned %d approval(s) that were never used", unused)
        return int(gone.rowcount)

    async def aclose(self) -> None:
        """Dispose of the running loop's engine."""
        await self._engines.aclose()

    # --- database side -------------------------------------------------------------------------

    async def _engine(self) -> AsyncEngine:
        engine = self._engines.get()
        if not self._ready:
            async with self._setup_locks.get():  # first uses in one loop wait for one setup
                if not self._ready:
                    await self._ensure_schema(engine)
                    self._ready = True
        return engine

    async def _ensure_schema(self, engine: AsyncEngine) -> None:
        name = engine.dialect.name
        if name not in DIALECTS:
            raise ConfigError(f"SQLAlchemyApprovalStore supports {sorted(DIALECTS)}, not {name!r}")
        async with _Logged("schema"):
            if self.create_schema:
                try:
                    await schema.create_tables(engine, self._t)
                except Exception as exc:  # e.g. no DDL rights: the check below decides
                    log_failure(
                        logger,
                        logging.WARNING,
                        exc,
                        "approval table creation failed; checking table %r as found",
                        self.table_name,
                    )
            await schema.check_table(engine, self._t)

    async def _read(self, op: str, query: Executable) -> ApprovalRecord | None:
        engine = await self._engine()
        async with _Logged(op), engine.connect() as conn:
            return await self._one(conn, query)

    async def _write(self, op: str, statement: Executable) -> bool:
        engine = await self._engine()
        async with _Logged(op), engine.begin() as conn:
            return (await conn.execute(statement)).first() is not None

    async def _one(self, conn: AsyncConnection, query: Executable) -> ApprovalRecord | None:
        row = (await conn.execute(query)).first()
        return None if row is None else _record(tuple(row))

    def _by_id(self, approval_id: str) -> Executable:
        from sqlalchemy import select

        t = self._t
        return select(*(t.c[name] for name in _COLUMNS)).where(t.c.approval_id == key(approval_id))

    def _open_query(self, run_id: str, call_id: str) -> Executable:
        from sqlalchemy import select

        t = self._t
        return select(*(t.c[name] for name in _COLUMNS)).where(
            t.c.run_id == key(run_id), t.c.call_id == key(call_id), t.c.consumed == 0
        )


class _Logged:
    """Logs a backend failure at ERROR (type only) and lets it propagate unchanged."""

    def __init__(self, op: str) -> None:
        self.op = op

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, kind: type[BaseException] | None, exc: Any, tb: Any) -> None:
        if exc is not None and isinstance(exc, Exception) and not isinstance(exc, AdapterError):
            log_failure(logger, logging.ERROR, exc, "sqlalchemy approval store %s failed", self.op)


def insert_new(table: Table, dialect: str, values: dict[str, Any]) -> Executable:
    """`INSERT ... ON CONFLICT DO NOTHING RETURNING seq` for `dialect`."""
    from sqlalchemy.dialects import postgresql, sqlite

    if dialect == "postgresql":
        return (
            postgresql.insert(table).values(values).on_conflict_do_nothing().returning(table.c.seq)
        )
    return sqlite.insert(table).values(values).on_conflict_do_nothing().returning(table.c.seq)


def decide_update(table: Table, approval_id: bytes, change: dict[str, Any]) -> Executable:
    """`UPDATE ... WHERE status = 'pending' RETURNING seq`: only one decide wins."""
    from sqlalchemy import update

    t = table
    return (
        update(t)
        .where(t.c.approval_id == approval_id, t.c.status == "pending")
        .values(change)
        .returning(t.c.seq)
    )


def consume_update(table: Table, approval_id: bytes, change: dict[str, Any]) -> Executable:
    """`UPDATE ... WHERE consumed = 0 AND status <> 'pending' RETURNING seq`: a test-and-set."""
    from sqlalchemy import update

    t = table
    return (
        update(t)
        .where(t.c.approval_id == approval_id, t.c.consumed == 0, t.c.status != "pending")
        .values(change)
        .returning(t.c.seq)
    )


def _engine_factory(engine: object) -> Callable[[], AsyncEngine]:
    """One engine per loop from a URL, or the caller's own factory."""
    if isinstance(engine, str) and engine:

        def from_url() -> AsyncEngine:
            from sqlalchemy.ext.asyncio import create_async_engine

            return create_async_engine(engine, pool_pre_ping=True)

        return from_url
    if callable(engine) and not isinstance(engine, str):
        made: Callable[[], AsyncEngine] = engine
        return made
    raise ConfigError("engine must be a database URL or a callable returning an AsyncEngine")


async def _dispose(engine: AsyncEngine) -> None:
    await engine.dispose()


def _record(row: Sequence[Any]) -> ApprovalRecord:
    approval_id, run_id, call_id, status, consumed, raw = row
    return read_row(
        raw,
        approval_id=approval_id,
        run_id=run_id,
        call_id=call_id,
        status=status,
        consumed=consumed,
    )


def _unsettled(approval_id: str) -> ApprovalStoreCorrupt:
    logger.error("approval %s kept changing under an update: failing closed", approval_id)
    return ApprovalStoreCorrupt(f"approval {approval_id!r} kept changing; try again")
