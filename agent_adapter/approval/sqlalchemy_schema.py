"""The SQLAlchemyApprovalStore table: its definition, idempotent DDL and the shape check."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ConfigError, MissingDependency

if TYPE_CHECKING:
    from sqlalchemy import Executable, MetaData, Table
    from sqlalchemy.ext.asyncio import AsyncEngine

INSTALL = 'pip install "agent-adapter[sqlalchemy]"'
OPEN_INDEX = "unique index on (run_id, call_id) WHERE consumed = 0"


def table_schema(name: str) -> tuple[MetaData, Table]:
    """The table: binary ids (lone surrogates survive), one unconsumed row per call."""
    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise MissingDependency(f"SQLAlchemyApprovalStore needs sqlalchemy: {INSTALL}") from exc
    metadata = sa.MetaData()
    blob = sa.LargeBinary
    table = sa.Table(
        name,
        metadata,
        sa.Column("seq", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True),
        sa.Column("approval_id", blob, unique=True, nullable=False),
        sa.Column("run_id", blob, nullable=False),
        sa.Column("call_id", blob, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("consumed", sa.SmallInteger, nullable=False),
        sa.Column("args_digest", sa.String(64), nullable=False),
        sa.Column("created_at", sa.Float, nullable=False),
        sa.Column("record", sa.Text, nullable=False),
    )
    unconsumed = sa.text("consumed = 0")
    columns = (table.c.run_id, table.c.call_id)
    sa.Index(
        f"{name}_open", *columns, unique=True, postgresql_where=unconsumed, sqlite_where=unconsumed
    )
    sa.Index(f"{name}_run", table.c.run_id, table.c.seq)
    sa.Index(f"{name}_status", table.c.status, table.c.seq)
    sa.Index(f"{name}_created", table.c.created_at)
    return metadata, table


def schema_ddl(table: Table, dialect: str) -> list[Executable]:
    """`CREATE ... IF NOT EXISTS` for the table and each index; Postgres locks the name first."""
    from sqlalchemy import text
    from sqlalchemy.schema import CreateIndex, CreateTable

    ddl: list[Any] = []
    if dialect == "postgresql":  # concurrent IF NOT EXISTS can still collide in pg_class
        lock = text("SELECT pg_advisory_xact_lock(hashtext(:name))")
        ddl.append(lock.bindparams(name=f"agent_adapter.approval:{table.name}"))
    ddl.append(CreateTable(table, if_not_exists=True))
    ddl.extend(CreateIndex(index, if_not_exists=True) for index in sorted(table.indexes, key=_name))
    return ddl


async def create_tables(engine: AsyncEngine, table: Table) -> None:
    """Run the DDL in one transaction; safe to repeat and to race from other loops or hosts."""
    async with engine.begin() as conn:
        for statement in schema_ddl(table, engine.dialect.name):
            await conn.execute(statement)


async def check_table(engine: AsyncEngine, table: Table) -> None:
    """Raise ConfigError unless the table has the store's columns and its open-call index."""
    async with engine.connect() as conn:
        columns, indexes = await conn.run_sync(_shape, table.name)
    missing = set(table.c.keys()) - columns
    if missing:
        raise ConfigError(f"table {table.name!r} lacks approval store columns {sorted(missing)}")
    if not any(_is_open_index(index, engine.dialect.name) for index in indexes):
        raise ConfigError(f"table {table.name!r} lacks the {OPEN_INDEX}")


def _name(index: Any) -> str:
    return str(index.name)


def _shape(conn: Any, table: str) -> tuple[set[str], list[Any]]:
    """The table's column names and reflected indexes (empty when it does not exist)."""
    from sqlalchemy import inspect

    inspector = inspect(conn)
    if not inspector.has_table(table):
        return set(), []
    columns = {str(column["name"]) for column in inspector.get_columns(table)}
    return columns, list(inspector.get_indexes(table))


def _is_open_index(index: Any, dialect: str) -> bool:
    """True for a unique index on exactly (run_id, call_id) limited to unconsumed rows."""
    columns = list(index.get("column_names") or ())
    if not index.get("unique") or sorted(columns) != ["call_id", "run_id"]:
        return False
    where = (index.get("dialect_options") or {}).get(f"{dialect}_where")
    text = "".join(("" if where is None else str(where)).split())
    return text.replace("(", "").replace(")", "").lower() == "consumed=0"
