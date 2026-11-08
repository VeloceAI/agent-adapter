"""SQLAlchemyCheckpointStore: checkpoints in Postgres (or SQLite) through SQLAlchemy 2 async."""

from __future__ import annotations

import asyncio
import builtins
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import (
    AdapterError,
    CheckpointStoreCorrupt,
    ConfigError,
    MissingDependency,
    log_failure,
)
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
from agent_adapter.runtime.loop_local import LoopLocal

if TYPE_CHECKING:
    from sqlalchemy import Executable, MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

logger = logging.getLogger(__name__)

DIALECTS = frozenset({"postgresql", "sqlite"})  # both have ON CONFLICT and RETURNING
INSTALL = 'pip install "agent-adapter[sqlalchemy]"'


class SQLAlchemyCheckpointStore:
    """CheckpointStore on a SQLAlchemy async engine (`agent-adapter[sqlalchemy]`), every host.

    `engine` is a URL (`"postgresql+asyncpg://..."`) or a callable returning an
    `AsyncEngine`; one engine is made per event loop. A step's compare-and-set is one
    `INSERT ... ON CONFLICT DO NOTHING` (step 0) or `UPDATE ... WHERE step = n - 1`.
    `metadata` holds both tables, for Alembic. Backend errors propagate (fail closed).

    Raises:
        ConfigError: a bad `engine` or `table`; on first use, another dialect or a table
            without the store's columns.
        MissingDependency: SQLAlchemy is not installed.
    """

    def __init__(
        self,
        engine: str | Callable[[], AsyncEngine],
        *,
        table: str = "aa_checkpoints",
        create_schema: bool = True,
    ) -> None:
        factory = _engine_factory(engine)
        if not isinstance(table, str) or not table.isidentifier():
            raise ConfigError(f"table must be a plain SQL identifier, got {table!r}")
        self.table_name = table
        self.create_schema = create_schema
        self.metadata, self._runs, self._calls = _tables(table)
        self._engines: LoopLocal[AsyncEngine] = LoopLocal(factory, close=_dispose)
        self._setup_locks: LoopLocal[asyncio.Lock] = LoopLocal(asyncio.Lock)
        self._ready = False

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        """Store the run's next checkpoint (compare-and-set on `step`).

        Raises:
            CheckpointConflict: `step` is not 0 for a new run or one past the stored step.
            ToolUsageError: a bad run id, step or checkpoint.
        """
        check_save(run_id, step, checkpoint)
        if not checkpoint.saved_at:
            checkpoint = checkpoint.model_copy(update={"saved_at": time.time()})
        from sqlalchemy import delete, update

        r, c = self._runs, self._calls
        values = {
            "step": step,
            "name": checkpoint.name,
            "status": checkpoint.status,
            "segment": _key(checkpoint.segment),
            "saved_at": checkpoint.saved_at,
            "record": encode_checkpoint(checkpoint),
        }
        engine = await self._engine()
        async with _Logged("save"), engine.begin() as conn:
            if step == 0:
                statement = _insert(r, engine.dialect.name, {"run_id": _key(run_id), **values})
            else:
                statement = (
                    update(r)
                    .where(r.c.run_id == _key(run_id), r.c.step == step - 1)
                    .values(values)
                    .returning(r.c.run_id)
                )
            if (await conn.execute(statement)).first() is None:
                raise conflict(run_id, step, await self._step(conn, run_id))
            await conn.execute(
                delete(c).where(c.c.run_id == _key(run_id), c.c.segment != values["segment"])
            )

    async def latest(self, run_id: str) -> StoredCheckpoint | None:
        """The run's latest checkpoint, or None.

        Raises:
            CheckpointStoreCorrupt: the stored row is unreadable or disagrees with itself.
        """
        check_id(run_id, "run_id")
        from sqlalchemy import select

        r = self._runs
        query = select(r.c.run_id, r.c.step, r.c.status, r.c.record).where(
            r.c.run_id == _key(run_id)
        )
        engine = await self._engine()
        async with _Logged("latest"), engine.connect() as conn:
            row = (await conn.execute(query)).first()
        return None if row is None else stored(row[0], row[1], row[3], row[2])

    async def list_unfinished(
        self, *, name: str | None = None, limit: int = 100
    ) -> builtins.list[StoredCheckpoint]:
        """Runs whose latest checkpoint is `running`, oldest first, optionally of one name.

        Raises:
            ConfigError: a negative `limit`.
        """
        check_limit(limit)
        from sqlalchemy import select

        r = self._runs
        query = select(r.c.run_id, r.c.step, r.c.status, r.c.record).where(r.c.status == "running")
        if name is not None:
            query = query.where(r.c.name == name)
        query = query.order_by(r.c.saved_at, r.c.run_id).limit(limit)
        engine = await self._engine()
        async with _Logged("list"), engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [stored(row[0], row[1], row[3], row[2]) for row in rows]

    async def delete(self, run_id: str) -> bool:
        """Forget the run and its call log; False when nothing was stored."""
        check_id(run_id, "run_id")
        from sqlalchemy import delete

        r, c = self._runs, self._calls
        engine = await self._engine()
        async with _Logged("delete"), engine.begin() as conn:
            await conn.execute(delete(c).where(c.c.run_id == _key(run_id)))
            gone = await conn.execute(delete(r).where(r.c.run_id == _key(run_id)))
        return bool(gone.rowcount)

    async def record_call(self, run_id: str, segment: str, call: CallRecord) -> None:
        """Keep a call record in the run's current segment (latest record per key wins).

        Raises:
            CheckpointConflict: `segment` is not the run's current one.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        check_id(call.key, "call key")
        text = encode_call(call)
        from sqlalchemy import select

        r = self._runs
        current = select(r.c.segment).where(r.c.run_id == _key(run_id)).with_for_update()
        engine = await self._engine()
        async with _Logged("record_call"), engine.begin() as conn:
            row = (await conn.execute(current)).first()
            if row is None or bytes(row[0]) != _key(segment):
                raise stale_segment(run_id)
            values = {
                "run_id": _key(run_id),
                "segment": _key(segment),
                "call_key": _key(call.key),
                "record": text,
            }
            await conn.execute(_upsert_call(self._calls, engine.dialect.name, values))

    async def calls(self, run_id: str, segment: str) -> builtins.list[CallRecord]:
        """The call records of one segment, in the order first written.

        Raises:
            CheckpointStoreCorrupt: a stored record is unreadable or has another key.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        from sqlalchemy import select

        c = self._calls
        query = (
            select(c.c.call_key, c.c.record)
            .where(c.c.run_id == _key(run_id), c.c.segment == _key(segment))
            .order_by(c.c.seq)
        )
        engine = await self._engine()
        async with _Logged("calls"), engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        records = []
        for raw_key, raw in rows:
            record = decode_call(raw, repr(run_id[:64]))
            if bytes(raw_key) != _key(record.key):
                logger.error("checkpoint call row of run %r disagrees with its key", run_id[:64])
                raise CheckpointStoreCorrupt(f"a call row of run {run_id[:64]!r} has another key")
            records.append(record)
        return records

    async def aclose(self) -> None:
        """Dispose of the running loop's engine."""
        await self._engines.aclose()

    # --- database side -------------------------------------------------------------------------

    async def _engine(self) -> AsyncEngine:
        engine = self._engines.get()
        if not self._ready:
            async with self._setup_locks.get():
                if not self._ready:
                    await self._ensure_schema(engine)
                    self._ready = True
        return engine

    async def _ensure_schema(self, engine: AsyncEngine) -> None:
        name = engine.dialect.name
        if name not in DIALECTS:
            raise ConfigError(
                f"SQLAlchemyCheckpointStore supports {sorted(DIALECTS)}, not {name!r}"
            )
        async with _Logged("schema"):
            if self.create_schema:
                async with engine.begin() as conn:
                    await conn.run_sync(self.metadata.create_all, checkfirst=True)
            async with engine.connect() as conn:
                found = await conn.run_sync(_columns, (self._runs.name, self._calls.name))
        for table in (self._runs, self._calls):
            missing = set(table.c.keys()) - found.get(table.name, set())
            if missing:
                raise ConfigError(
                    f"table {table.name!r} lacks checkpoint store columns {sorted(missing)}"
                )

    async def _step(self, conn: AsyncConnection, run_id: str) -> int | None:
        from sqlalchemy import select

        row = (
            await conn.execute(select(self._runs.c.step).where(self._runs.c.run_id == _key(run_id)))
        ).first()
        return None if row is None else int(row[0])


class _Logged:
    """Logs a backend failure at ERROR (type only) and lets it propagate unchanged."""

    def __init__(self, op: str) -> None:
        self.op = op

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, kind: type[BaseException] | None, exc: Any, tb: Any) -> None:
        if exc is not None and isinstance(exc, Exception) and not isinstance(exc, AdapterError):
            log_failure(
                logger, logging.ERROR, exc, "sqlalchemy checkpoint store %s failed", self.op
            )


def _tables(name: str) -> tuple[MetaData, Table, Table]:
    """The runs table (one row per run) and the calls table (one row per call and segment)."""
    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise MissingDependency(f"SQLAlchemyCheckpointStore needs sqlalchemy: {INSTALL}") from exc
    metadata = sa.MetaData()
    blob = sa.LargeBinary
    runs = sa.Table(
        name,
        metadata,
        sa.Column("run_id", blob, primary_key=True),
        sa.Column("step", sa.BigInteger, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("segment", blob, nullable=False),
        sa.Column("saved_at", sa.Float, nullable=False),
        sa.Column("record", sa.Text, nullable=False),
    )
    sa.Index(f"{name}_status", runs.c.status, runs.c.saved_at)
    calls = sa.Table(
        f"{name}_calls",
        metadata,
        sa.Column("seq", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True),
        sa.Column("run_id", blob, nullable=False),
        sa.Column("segment", blob, nullable=False),
        sa.Column("call_key", blob, nullable=False),
        sa.Column("record", sa.Text, nullable=False),
        sa.UniqueConstraint("run_id", "segment", "call_key", name=f"{name}_calls_key"),
    )
    return metadata, runs, calls


def _insert(table: Table, dialect: str, values: dict[str, Any]) -> Executable:
    """`INSERT ... ON CONFLICT DO NOTHING RETURNING run_id`."""
    from sqlalchemy.dialects import postgresql, sqlite

    module: Any = postgresql if dialect == "postgresql" else sqlite
    statement: Executable = (
        module.insert(table).values(values).on_conflict_do_nothing().returning(table.c.run_id)
    )
    return statement


def _upsert_call(table: Table, dialect: str, values: dict[str, Any]) -> Executable:
    """`INSERT ... ON CONFLICT (run_id, segment, call_key) DO UPDATE SET record`."""
    from sqlalchemy.dialects import postgresql, sqlite

    module: Any = postgresql if dialect == "postgresql" else sqlite
    statement = module.insert(table).values(values)
    upsert: Executable = statement.on_conflict_do_update(
        index_elements=["run_id", "segment", "call_key"],
        set_={"record": statement.excluded.record},
    )
    return upsert


def _columns(conn: Any, names: tuple[str, ...]) -> dict[str, set[str]]:
    from sqlalchemy import inspect

    inspector = inspect(conn)
    return {
        name: {str(col["name"]) for col in inspector.get_columns(name)}
        for name in names
        if inspector.has_table(name)
    }


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


def _key(text: str) -> bytes:
    return text.encode("utf-8", "surrogatepass")
