"""SQLAlchemyBudgetStore: budget counters in Postgres (or SQLite) through SQLAlchemy 2 async."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ConfigError, IntegrationError, MissingDependency, log_failure
from agent_adapter.hooks.builtin.budgets.store import (
    BudgetUsage,
    Reserved,
    admits,
    check_ids,
    check_reserve,
    check_usd,
)
from agent_adapter.runtime.loop_local import LoopLocal

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

logger = logging.getLogger(__name__)

DIALECTS = frozenset({"postgresql", "sqlite"})
INSTALL = 'pip install "agent-adapter[sqlalchemy]"'
PURGE_EVERY_S = 60.0  # ended windows and expired holds are swept at most this often


class SQLAlchemyBudgetStore:
    """A `BudgetStore` on a SQLAlchemy async engine (`agent-adapter[sqlalchemy]`).

    `engine` is a URL (`"postgresql+asyncpg://..."`) or a callable returning an
    `AsyncEngine`; one engine is made per event loop. `reserve` locks the counter row with
    an `UPDATE` (READ COMMITTED on Postgres; SQLite's write lock) before it checks and
    holds, so workers never hold past the limit together. `metadata` holds both tables.

    Raises:
        ConfigError: a bad `engine` or `table` prefix; on first use, a dialect other than
            Postgres or SQLite.
        MissingDependency: SQLAlchemy is not installed.
    """

    def __init__(
        self,
        engine: str | Callable[[], AsyncEngine],
        *,
        table: str = "aa_budget",
        create_schema: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        factory = _engine_factory(engine)
        if not isinstance(table, str) or not table.isidentifier():
            raise ConfigError(f"table must be a plain SQL identifier, got {table!r}")
        self.table_name = table
        self.create_schema = create_schema
        self.metadata, self._counters, self._holds = _tables(table)
        self._engines: LoopLocal[AsyncEngine] = LoopLocal(factory, close=_dispose)
        self._setup_locks: LoopLocal[asyncio.Lock] = LoopLocal(asyncio.Lock)
        self._ready = False
        self._clock = clock
        self._purged_at = -math.inf

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
        """Hold `amount_usd` when it fits `limit_usd` (None: always), under a row lock."""
        check_reserve(key, reservation_id, amount_usd, hold_s)
        now = self._clock()
        engine = await self._engine()
        async with _Logged("reserve"), _begin(engine) as conn:
            usage = await self._locked(conn, engine.dialect.name, key, now)
            if not admits(usage, amount_usd, limit_usd):
                return Reserved(False, usage)
            h = self._holds
            await conn.execute(
                h.insert().values(
                    key=key, reservation_id=reservation_id, amount=amount_usd, until=now + hold_s
                )
            )
            if expire_at is not None:
                c = self._counters
                await conn.execute(c.update().where(c.c.key == key).values(expire_at=expire_at))
        await self._sweep(engine, now)
        return Reserved(True, usage)

    async def settle(
        self, key: str, reservation_id: str, cost_usd: float, *, expire_at: float | None
    ) -> None:
        """Drop the hold and add the real cost (even when the hold already expired)."""
        check_ids(key, reservation_id)
        check_usd(cost_usd, "cost_usd")
        now = self._clock()
        engine = await self._engine()
        c, h = self._counters, self._holds
        async with _Logged("settle"), _begin(engine) as conn:
            await self._locked(conn, engine.dialect.name, key, now)
            await conn.execute(
                h.delete().where(h.c.key == key, h.c.reservation_id == reservation_id)
            )
            change: dict[str, Any] = {"spent": c.c.spent + float(cost_usd)}
            if expire_at is not None:
                change["expire_at"] = expire_at
            await conn.execute(c.update().where(c.c.key == key).values(change))

    async def release(self, key: str, reservation_id: str) -> None:
        """Drop the hold; nothing was spent."""
        check_ids(key, reservation_id)
        engine = await self._engine()
        h = self._holds
        async with _Logged("release"), _begin(engine) as conn:
            await conn.execute(
                h.delete().where(h.c.key == key, h.c.reservation_id == reservation_id)
            )

    async def usage(self, key: str) -> BudgetUsage:
        """The counter's spend and live holds; zero for an unknown or ended key."""
        now = self._clock()
        engine = await self._engine()
        async with _Logged("usage"), _begin(engine) as conn:
            return await self._read(conn, key, now)

    async def aclose(self) -> None:
        """Dispose of the running loop's engine."""
        await self._engines.aclose()

    # --- internals -------------------------------------------------------------------------

    async def _locked(
        self, conn: AsyncConnection, dialect: str, key: str, now: float
    ) -> BudgetUsage:
        """Create the counter if needed, lock it, reset it if its window ended, read it."""
        from sqlalchemy import select

        c, h = self._counters, self._holds
        await conn.execute(_insert_new(c, dialect, {"key": key, "spent": 0.0, "expire_at": None}))
        # A real write takes the row lock: a later reserver waits, then reads every hold
        # committed before it (READ COMMITTED; under REPEATABLE READ it fails closed).
        await conn.execute(c.update().where(c.c.key == key).values(spent=c.c.spent))
        row = (await conn.execute(select(c.c.expire_at).where(c.c.key == key))).first()
        if row is not None and row[0] is not None and row[0] <= now:
            await conn.execute(c.update().where(c.c.key == key).values(spent=0.0, expire_at=None))
            await conn.execute(h.delete().where(h.c.key == key))
        await conn.execute(h.delete().where(h.c.key == key, h.c.until <= now))
        return await self._read(conn, key, now)

    async def _read(self, conn: AsyncConnection, key: str, now: float) -> BudgetUsage:
        from sqlalchemy import func, select

        c, h = self._counters, self._holds
        row = (await conn.execute(select(c.c.spent, c.c.expire_at).where(c.c.key == key))).first()
        if row is None or (row[1] is not None and row[1] <= now):
            return BudgetUsage()
        held = await conn.scalar(
            select(func.coalesce(func.sum(h.c.amount), 0.0)).where(h.c.key == key, h.c.until > now)
        )
        return BudgetUsage(_usd(row[0]), _usd(held))

    async def _sweep(self, engine: AsyncEngine, now: float) -> None:
        """Delete ended counters and expired holds, outside any counter's lock."""
        if now - self._purged_at < PURGE_EVERY_S:
            return
        self._purged_at = now
        c, h = self._counters, self._holds
        try:
            async with _begin(engine) as conn:
                await conn.execute(h.delete().where(h.c.until <= now))
                await conn.execute(c.delete().where(c.c.expire_at <= now))
        except Exception as exc:
            # Housekeeping only: a failed sweep leaves rows the next one removes.
            log_failure(logger, logging.WARNING, exc, "budget store sweep failed")

    async def _engine(self) -> AsyncEngine:
        engine = self._engines.get()
        if self._ready:
            return engine
        async with self._setup_locks.get():
            if not self._ready:
                dialect = engine.dialect.name
                if dialect not in DIALECTS:
                    raise ConfigError(
                        f"SQLAlchemyBudgetStore supports {sorted(DIALECTS)}, not {dialect!r}"
                    )
                if self.create_schema:
                    async with _Logged("schema"), engine.begin() as conn:
                        for statement in self._ddl(dialect):
                            await conn.execute(statement)
                self._ready = True
        return engine

    def _ddl(self, dialect: str) -> list[Any]:
        """`CREATE ... IF NOT EXISTS` for both tables and their indexes, safe to race."""
        from sqlalchemy import text
        from sqlalchemy.schema import CreateIndex, CreateTable

        ddl: list[Any] = []
        if dialect == "postgresql":  # concurrent IF NOT EXISTS can still collide in pg_class
            lock = text("SELECT pg_advisory_xact_lock(hashtext(:name))")
            ddl.append(lock.bindparams(name=f"agent_adapter.budget:{self.table_name}"))
        for table in (self._counters, self._holds):
            ddl.append(CreateTable(table, if_not_exists=True))
            indexes = sorted(table.indexes, key=lambda index: str(index.name))
            ddl.extend(CreateIndex(index, if_not_exists=True) for index in indexes)
        return ddl


@asynccontextmanager
async def _begin(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """One transaction; READ COMMITTED on Postgres, whatever the engine's default."""
    async with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            conn = await conn.execution_options(isolation_level="READ COMMITTED")
        async with conn.begin():
            yield conn


class _Logged:
    """Log a backend failure by type before it propagates."""

    def __init__(self, op: str) -> None:
        self.op = op

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, kind: Any, exc: BaseException | None, tb: Any) -> None:
        if isinstance(exc, Exception) and not isinstance(exc, ConfigError | IntegrationError):
            log_failure(logger, logging.ERROR, exc, "sqlalchemy budget store %s failed", self.op)


def _tables(prefix: str) -> tuple[MetaData, Table, Table]:
    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise MissingDependency(f"SQLAlchemyBudgetStore needs sqlalchemy: {INSTALL}") from exc
    metadata = sa.MetaData()
    counters = sa.Table(
        f"{prefix}_counters",
        metadata,
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("spent", sa.Float, nullable=False),
        sa.Column("expire_at", sa.Float, nullable=True, index=True),
    )
    holds = sa.Table(
        f"{prefix}_holds",
        metadata,
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("reservation_id", sa.String(128), primary_key=True),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("until", sa.Float, nullable=False, index=True),
    )
    return metadata, counters, holds


def _insert_new(table: Table, dialect: str, values: dict[str, Any]) -> Any:
    """`INSERT ... ON CONFLICT DO NOTHING` for `dialect`."""
    from sqlalchemy.dialects import postgresql, sqlite

    module = postgresql if dialect == "postgresql" else sqlite
    return module.insert(table).values(values).on_conflict_do_nothing()


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


def _usd(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        logger.error("budget store row holds a malformed amount")
        raise IntegrationError(
            "the budget store holds a malformed amount", integration="sqlalchemy"
        )
    return float(value)
