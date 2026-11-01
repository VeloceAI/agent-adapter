"""SQLAlchemyApprovalStore first use: one idempotent, serialised schema setup per store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.approval import sqlalchemy_schema
from agent_adapter.approval.sqlalchemy_store import SQLAlchemyApprovalStore
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.runtime.sync import run_sync
from tests.approval.support import ids_from_threads

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")


def req(call_id: str = "call_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="run_1", call_id=call_id, tool="pay", arguments={"amount": 5}
    )


def url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}"


def sql(statement: Any, dialect: str) -> str:
    from sqlalchemy.dialects import postgresql, sqlite

    found = {"postgresql": postgresql, "sqlite": sqlite}[dialect].dialect()
    return " ".join(str(statement.compile(dialect=found)).split())


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_schema_ddl_is_create_if_not_exists_for_the_table_and_every_index(dialect: str) -> None:
    store = SQLAlchemyApprovalStore("sqlite+aiosqlite:///x.db")
    texts = [sql(s, dialect) for s in sqlalchemy_schema.schema_ddl(store._t, dialect)]
    creates = [t for t in texts if t.startswith("CREATE")]
    assert creates[0].startswith("CREATE TABLE IF NOT EXISTS aa_approvals")
    assert len(creates) == 1 + len(store._t.indexes)
    assert all(
        t.startswith(("CREATE INDEX IF NOT EXISTS", "CREATE UNIQUE INDEX IF NOT EXISTS"))
        for t in creates[1:]
    )
    assert any("aa_approvals_open" in t and "WHERE consumed = 0" in t for t in creates)


def test_postgres_ddl_takes_a_transaction_advisory_lock_first() -> None:
    store = SQLAlchemyApprovalStore("postgresql+asyncpg://h/db", table="approvals")
    first = sql(sqlalchemy_schema.schema_ddl(store._t, "postgresql")[0], "postgresql")
    assert first.startswith("SELECT pg_advisory_xact_lock(hashtext(")
    sqlite_first = sql(sqlalchemy_schema.schema_ddl(store._t, "sqlite")[0], "sqlite")
    assert sqlite_first.startswith("CREATE TABLE IF NOT EXISTS")


async def test_concurrent_first_uses_in_one_loop_set_up_the_schema_once(tmp_path, monkeypatch):
    store = SQLAlchemyApprovalStore(url(tmp_path))
    real = store._ensure_schema
    calls = 0

    async def counted(engine: Any) -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)  # let the other first uses pile up behind it
        await real(engine)

    monkeypatch.setattr(store, "_ensure_schema", counted)
    records = await asyncio.gather(*(store.create(req()) for _ in range(40)))
    assert calls == 1 and len({r.approval_id for r in records}) == 1
    await store.aclose()


def test_two_stores_on_two_loops_set_up_one_database_at_once(tmp_path: Path) -> None:
    stores = [SQLAlchemyApprovalStore(url(tmp_path)) for _ in range(2)]
    turn = iter(stores)

    async def burst() -> list[str]:
        store = next(turn)
        try:
            records = await asyncio.gather(*(store.create(req()) for _ in range(20)))
            return [r.approval_id for r in records]
        finally:
            await store.aclose()

    ids = ids_from_threads(burst)
    assert len(ids) == 40 and len(set(ids)) == 1
    check = SQLAlchemyApprovalStore(url(tmp_path), create_schema=False)
    assert len(run_sync(check.list())) == 1
    run_sync(check.aclose())
