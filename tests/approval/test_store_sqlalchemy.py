"""SQLAlchemyApprovalStore: configuration, the Postgres SQL it emits, schema checks, pruning."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_adapter.approval import sqlalchemy_schema, sqlalchemy_store
from agent_adapter.approval.persist import ApprovalStoreCorrupt
from agent_adapter.approval.sqlalchemy_store import SQLAlchemyApprovalStore
from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

sa = pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

YES = ApprovalDecision(approved=True, decided_by="ana")


def req(call_id: str = "call_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="run_1", call_id=call_id, tool="pay", arguments={"amount": 5}
    )


def url(tmp_path: Path, name: str = "a.db") -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}"


@pytest.fixture
async def store(tmp_path: Path) -> Any:
    made = SQLAlchemyApprovalStore(url(tmp_path))
    yield made
    await made.aclose()


def postgres(statement: Any) -> str:
    from sqlalchemy.dialects import postgresql

    sql = " ".join(str(statement.compile(dialect=postgresql.dialect())).split())
    return re.sub(r"::\w+", "", sql)  # SQLAlchemy 2.1 casts bound values


@pytest.mark.parametrize("bad", ["", 5, None])
def test_engine_must_be_a_url_or_a_callable(bad: Any) -> None:
    with pytest.raises(ConfigError, match="engine"):
        SQLAlchemyApprovalStore(bad)


@pytest.mark.parametrize("bad", ["", "drop table;", "a-b", 3])
def test_table_must_be_an_identifier(bad: Any) -> None:
    with pytest.raises(ConfigError, match="table"):
        SQLAlchemyApprovalStore("sqlite+aiosqlite:///x.db", table=bad)


def test_missing_sqlalchemy_is_a_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[sqlalchemy\]"):
        SQLAlchemyApprovalStore("sqlite+aiosqlite:///x.db")


def test_postgres_create_is_insert_on_conflict_do_nothing_returning() -> None:
    table = SQLAlchemyApprovalStore("postgresql+asyncpg://h/db")._t
    sql = postgres(sqlalchemy_store.insert_new(table, "postgresql", {"status": "pending"}))
    assert sql.startswith("INSERT INTO aa_approvals")
    assert sql.endswith("ON CONFLICT DO NOTHING RETURNING aa_approvals.seq")


def test_postgres_decide_and_consume_are_guarded_updates_returning() -> None:
    table = SQLAlchemyApprovalStore("postgresql+asyncpg://h/db")._t
    decide = postgres(sqlalchemy_store.decide_update(table, b"a", {"status": "approved"}))
    assert "WHERE aa_approvals.approval_id = %(approval_id_1)s" in decide
    assert "AND aa_approvals.status = %(status_1)s RETURNING aa_approvals.seq" in decide
    consume = postgres(sqlalchemy_store.consume_update(table, b"a", {"consumed": 1}))
    assert "AND aa_approvals.consumed = %(consumed_1)s" in consume
    assert "AND aa_approvals.status != %(status_1)s RETURNING aa_approvals.seq" in consume


def test_postgres_schema_has_binary_ids_and_a_partial_unique_index() -> None:
    from sqlalchemy.schema import CreateIndex, CreateTable

    store = SQLAlchemyApprovalStore("postgresql+asyncpg://h/db", table="approvals")
    assert "approval_id BYTEA NOT NULL" in postgres(CreateTable(store._t))
    [index] = [i for i in store._t.indexes if i.name == "approvals_open"]
    assert postgres(CreateIndex(index)) == (
        "CREATE UNIQUE INDEX approvals_open ON approvals (run_id, call_id) WHERE consumed = 0"
    )
    assert "approvals" in store.metadata.tables  # Alembic: target_metadata


async def test_other_dialects_are_refused_on_first_use() -> None:
    engine = SimpleNamespace(dialect=SimpleNamespace(name="mysql"))
    store = SQLAlchemyApprovalStore(lambda: engine)  # type: ignore[arg-type, return-value]
    with pytest.raises(ConfigError, match="mysql"):
        await store.get("x")


async def test_a_foreign_table_of_the_same_name_is_refused(tmp_path: Path) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url(tmp_path))
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE aa_approvals (id INTEGER PRIMARY KEY)")
    await engine.dispose()
    store = SQLAlchemyApprovalStore(url(tmp_path))
    with pytest.raises(ConfigError, match="lacks approval store columns"):
        await store.find("r", "c")
    await store.aclose()


async def test_without_create_schema_a_missing_table_is_a_config_error(tmp_path: Path) -> None:
    store = SQLAlchemyApprovalStore(url(tmp_path), create_schema=False)
    with pytest.raises(ConfigError, match="lacks"):
        await store.list()
    await store.aclose()


async def test_a_lost_table_creation_race_is_rechecked(tmp_path: Path, monkeypatch) -> None:
    first = SQLAlchemyApprovalStore(url(tmp_path))
    record = await first.create(req())
    second = SQLAlchemyApprovalStore(url(tmp_path))

    def raced(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("table exists")

    monkeypatch.setattr(sqlalchemy_schema, "create_tables", raced)
    assert await second.get(record.approval_id) == record
    for store in (first, second):
        await store.aclose()


async def test_prune_deletes_old_records_and_warns_for_unused_ones(store, caplog) -> None:
    used = await store.create(req("a"))
    await store.decide(used.approval_id, YES)
    await store.consume(used.approval_id)
    await store.create(req("b"))
    assert await store.prune(older_than_s=3600) == 0
    with caplog.at_level(logging.WARNING):
        assert await store.prune(older_than_s=0) == 2
    assert "pruned 1 approval(s) that were never used" in caplog.text
    assert await store.list() == []


@pytest.mark.parametrize("bad", [-1, True, "1", float("nan")])
async def test_prune_rejects_a_bad_age(store, bad: Any) -> None:
    with pytest.raises(ConfigError, match="older_than_s"):
        await store.prune(older_than_s=bad)


async def test_a_tampered_row_fails_closed(store) -> None:
    record = await store.create(req())
    engine = await store._engine()
    async with engine.begin() as conn:
        await conn.exec_driver_sql("UPDATE aa_approvals SET status = 'approved'")
    with pytest.raises(ApprovalStoreCorrupt, match="disagrees"):
        await store.get(record.approval_id)


async def test_backend_errors_are_logged_and_propagate(tmp_path: Path, caplog) -> None:
    store = SQLAlchemyApprovalStore(url(tmp_path / "missing" / "folder"))
    with caplog.at_level(logging.ERROR), pytest.raises(sa.exc.OperationalError):
        await store.get("x")
    assert "sqlalchemy approval store schema failed (" in caplog.text
    await store.aclose()


async def test_decide_that_keeps_losing_fails_closed(store, monkeypatch) -> None:
    record = await store.create(req())

    async def lost(op: str, statement: Any) -> bool:
        return False

    monkeypatch.setattr(store, "_write", lost)
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.decide(record.approval_id, YES)


async def test_create_that_keeps_conflicting_fails_closed(store, monkeypatch) -> None:
    def conflicting(table: Any, dialect: str, values: Any) -> Any:
        return sa.select(table.c.seq).where(sa.false())

    monkeypatch.setattr(sqlalchemy_store, "insert_new", conflicting)
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.create(req())


async def test_consume_of_a_record_consumed_meanwhile_is_false(store, monkeypatch) -> None:
    record = await store.create(req())
    decided = await store.decide(record.approval_id, YES)
    assert await store.consume(record.approval_id) is True

    async def stale(approval_id: str) -> Any:
        return decided  # read before the other consumer's write

    monkeypatch.setattr(store, "get", stale)
    assert await store.consume(record.approval_id) is False


async def precreate(tmp_path: Path, index_sql: str | None) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.schema import CreateTable

    _, table = sqlalchemy_schema.table_schema("aa_approvals")
    engine = create_async_engine(url(tmp_path))
    async with engine.begin() as conn:
        await conn.execute(CreateTable(table))
        if index_sql is not None:
            await conn.exec_driver_sql(index_sql)
    await engine.dispose()


WRONG_INDEXES = [
    None,
    "CREATE UNIQUE INDEX x ON aa_approvals (run_id) WHERE consumed = 0",
    "CREATE UNIQUE INDEX x ON aa_approvals (run_id, call_id) WHERE consumed = 1",
]
WRONG_OPEN_INDEXES = [  # the store's own index name, holding the wrong definition
    "CREATE INDEX aa_approvals_open ON aa_approvals (run_id, call_id)",
    "CREATE UNIQUE INDEX aa_approvals_open ON aa_approvals (run_id, call_id)",
]
NO_OPEN_INDEX = r"unique index on \(run_id, call_id\) WHERE consumed = 0"


@pytest.mark.parametrize("index_sql", WRONG_INDEXES + WRONG_OPEN_INDEXES)
async def test_without_create_schema_a_table_lacking_the_index_is_refused(
    tmp_path: Path, index_sql: str | None
) -> None:
    await precreate(tmp_path, index_sql)
    store = SQLAlchemyApprovalStore(url(tmp_path), create_schema=False)
    with pytest.raises(ConfigError, match=NO_OPEN_INDEX):
        await store.create(req())
    await store.aclose()


@pytest.mark.parametrize("index_sql", WRONG_OPEN_INDEXES)
async def test_a_wrong_index_under_the_store_index_name_is_refused(
    tmp_path: Path, index_sql: str
) -> None:
    await precreate(tmp_path, index_sql)
    store = SQLAlchemyApprovalStore(url(tmp_path))
    with pytest.raises(ConfigError, match=NO_OPEN_INDEX):
        await store.create(req())
    await store.aclose()


@pytest.mark.parametrize("index_sql", WRONG_INDEXES)
async def test_create_schema_adds_a_missing_index_to_an_existing_table(
    tmp_path: Path, index_sql: str | None
) -> None:
    await precreate(tmp_path, index_sql)
    store = SQLAlchemyApprovalStore(url(tmp_path))
    record = await store.create(req())
    assert await store.create(req()) == record
    await store.aclose()


async def test_a_table_with_its_own_partial_unique_index_is_accepted(tmp_path: Path) -> None:
    sql = "CREATE UNIQUE INDEX my_open ON aa_approvals (call_id, run_id) WHERE (consumed = 0)"
    await precreate(tmp_path, sql)
    store = SQLAlchemyApprovalStore(url(tmp_path), create_schema=False)
    record = await store.create(req())
    assert await store.create(req()) == record
    await store.aclose()


async def test_a_failed_table_creation_is_logged_at_warning(tmp_path: Path, monkeypatch, caplog):
    first = SQLAlchemyApprovalStore(url(tmp_path))
    await first.create(req())
    second = SQLAlchemyApprovalStore(url(tmp_path))

    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("permission denied for schema public")

    monkeypatch.setattr(sqlalchemy_schema, "create_tables", denied)
    with caplog.at_level(logging.WARNING, logger=sqlalchemy_store.__name__):
        await second.list()
    [entry] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "approval table creation failed" in entry.getMessage()
    assert "PermissionError" in entry.getMessage() and "denied" not in entry.getMessage()
    for store in (first, second):
        await store.aclose()
