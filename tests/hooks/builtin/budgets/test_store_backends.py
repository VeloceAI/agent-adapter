"""Backend-specific BudgetStore behaviour: configuration, schema, malformed data, eviction."""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from contextlib import closing
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError, IntegrationError, MissingDependency
from agent_adapter.hooks.builtin.budgets import (
    MemoryBudgetStore,
    RedisBudgetStore,
    SQLAlchemyBudgetStore,
    SQLiteBudgetStore,
    sqlite_store,
)
from agent_adapter.hooks.builtin.budgets import sqlalchemy_store as sa_store
from agent_adapter.runtime.loop_local import LoopLocal
from tests.hooks.builtin.budgets.stores import Clock, fake_redis

# --- memory ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_memory_max_keys_must_be_a_positive_int(bad):
    with pytest.raises(ConfigError, match="max_keys"):
        MemoryBudgetStore(max_keys=bad)


async def test_memory_drops_ended_windows_then_refuses_new_keys(caplog):
    clock = Clock()
    store = MemoryBudgetStore(max_keys=2, clock=clock)
    await store.settle("ended", "r", 1.0, expire_at=clock.now + 1)
    await store.settle("a", "r", 1.0, expire_at=None)
    clock.now += 5
    await store.settle("b", "r", 1.0, expire_at=None)  # "ended" goes, quietly
    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="max_keys"):
        await store.reserve("c", "r", 0.1, limit_usd=1.0, hold_s=5, expire_at=None)
    assert (await store.usage("a")).spent_usd == 1.0  # never forgotten to make room
    await store.settle("a", "r2", 1.0, expire_at=None)  # existing keys still work
    assert "budget store full" in caplog.text


# --- sqlite ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [":memory:", "", "file:x.db?mode=memory"])
def test_sqlite_needs_a_file_path(bad):
    with pytest.raises(ConfigError, match="file path"):
        SQLiteBudgetStore(bad)


@pytest.mark.parametrize(("kwargs", "match"), [({"timeout": 0}, "timeout"), ({}, "path")])
def test_sqlite_bad_settings(kwargs, match, tmp_path):
    path: Any = 5 if match == "path" else tmp_path / "b.db"
    with pytest.raises(ConfigError, match=match):
        SQLiteBudgetStore(path, **kwargs)


async def test_sqlite_refuses_another_schema_version(tmp_path):
    path = tmp_path / "b.db"
    await SQLiteBudgetStore(path).usage("k")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"UPDATE {sqlite_store.META} SET value = 99 WHERE key = 'schema'")
    with pytest.raises(ConfigError, match="schema 99"):
        await SQLiteBudgetStore(path).usage("k")


async def test_sqlite_fails_closed_on_a_malformed_row(tmp_path, caplog):
    path = tmp_path / "b.db"
    store = SQLiteBudgetStore(path)
    await store.settle("k", "r", 0.1, expire_at=None)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"UPDATE {sqlite_store.COUNTERS} SET spent = 'lots'")
    with caplog.at_level(logging.ERROR), pytest.raises(IntegrationError):
        await store.reserve("k", "r2", 0.1, limit_usd=1.0, hold_s=5, expire_at=None)


async def test_sqlite_logs_a_backend_failure_by_type(tmp_path, caplog):
    store = SQLiteBudgetStore(tmp_path / "missing" / "b.db")
    with caplog.at_level(logging.ERROR), pytest.raises((sqlite3.Error, OSError)):
        await store.usage("k")
    assert "sqlite budget store usage failed" in caplog.text


# --- redis ----------------------------------------------------------------------------------


def test_redis_settings_are_checked():
    with pytest.raises(ConfigError, match="client_factory"):
        RedisBudgetStore(object())  # type: ignore[arg-type]
    for bad in ("", "aa", "{aa}:"):
        with pytest.raises(ConfigError, match="prefix"):
            RedisBudgetStore(lambda: None, prefix=bad)


def test_redis_keys_of_one_counter_share_a_hash_tag():
    keys = RedisBudgetStore(lambda: None).keys_of("team|shop")
    assert keys == ("aa:budget:{team|shop}:c", "aa:budget:{team|shop}:h", "aa:budget:{team|shop}:x")


async def test_redis_reads_clients_that_decode_responses():
    client = fake_redis(decode_responses=True)
    store = RedisBudgetStore(lambda: client)
    assert (await store.reserve("k", "r", 0.5, limit_usd=1.0, hold_s=5, expire_at=None)).accepted
    assert (await store.usage("k")).reserved_usd == 0.5
    await LoopLocal.aclose_all()


async def test_redis_windowed_counters_expire_with_their_window():
    client = fake_redis()
    store = RedisBudgetStore(lambda: client, prefix="w:")
    end = time.time() + 3600
    await store.reserve("k", "r", 0.5, limit_usd=1.0, hold_s=5, expire_at=end)
    ttl = await client.pttl("w:{k}:c")
    assert 3_500_000 < ttl <= 3_601_000
    await LoopLocal.aclose_all()


class Replying:
    def __init__(self, reply: Any) -> None:
        self.reply = reply

    async def eval(self, *args: Any) -> Any:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    async def evalsha(self, *args: Any) -> Any:
        raise RuntimeError("NOSCRIPT No matching script")


@pytest.mark.parametrize(
    "reply", [None, [b"1", b"0"], [b"2", b"0", b"0"], [b"1", b"x", b"0"], [b"1", b"-1", b"0"]]
)
async def test_redis_fails_closed_on_a_malformed_reply(reply, caplog):
    store = RedisBudgetStore(lambda: Replying(reply))
    with caplog.at_level(logging.ERROR), pytest.raises(IntegrationError):
        await store.usage("k")


async def test_redis_logs_a_backend_failure_by_type(caplog):
    store = RedisBudgetStore(lambda: Replying(ConnectionError("secret host")))
    with caplog.at_level(logging.ERROR), pytest.raises(ConnectionError):
        await store.release("k", "r")
    assert "redis budget store release failed" in caplog.text and "secret" not in caplog.text


async def test_redis_usage_needs_a_key():
    with pytest.raises(ConfigError, match="key"):
        await RedisBudgetStore(lambda: None).usage("")


# --- sqlalchemy -----------------------------------------------------------------------------


def test_sqlalchemy_settings_are_checked():
    pytest.importorskip("sqlalchemy")
    with pytest.raises(ConfigError, match="engine"):
        SQLAlchemyBudgetStore(5)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="table"):
        SQLAlchemyBudgetStore("sqlite+aiosqlite:///x.db", table="no-dash")


def test_sqlalchemy_missing_is_a_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[sqlalchemy\]"):
        SQLAlchemyBudgetStore("postgresql+asyncpg://db/x")


def test_sqlalchemy_postgres_ddl_takes_an_advisory_lock():
    pytest.importorskip("sqlalchemy")
    store = SQLAlchemyBudgetStore("postgresql+asyncpg://db/x", table="shop_budget")
    ddl = [str(statement) for statement in store._ddl("postgresql")]
    assert "pg_advisory_xact_lock" in ddl[0]
    assert sum("IF NOT EXISTS" in s for s in ddl[1:]) == len(ddl) - 1
    assert {t.name for t in store.metadata.sorted_tables} == {
        "shop_budget_counters",
        "shop_budget_holds",
    }


def test_sqlalchemy_postgres_insert_skips_existing_counters():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.dialects import postgresql

    store = SQLAlchemyBudgetStore("postgresql+asyncpg://db/x")
    statement = sa_store._insert_new(store._counters, "postgresql", {"key": "k", "spent": 0.0})
    assert "ON CONFLICT DO NOTHING" in str(statement.compile(dialect=postgresql.dialect()))


async def test_sqlalchemy_refuses_other_dialects():
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import create_async_engine

    def engine() -> Any:
        made = create_async_engine("sqlite+aiosqlite://")
        made.dialect.name = "mysql"
        return made

    store = SQLAlchemyBudgetStore(engine)
    with pytest.raises(ConfigError, match="mysql"):
        await store.usage("k")
    await LoopLocal.aclose_all()


async def test_sqlalchemy_fails_closed_on_a_malformed_row(tmp_path, caplog):
    pytest.importorskip("aiosqlite")
    path = tmp_path / "b.db"
    store = SQLAlchemyBudgetStore(f"sqlite+aiosqlite:///{path.as_posix()}")
    await store.settle("k", "r", 0.1, expire_at=None)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE aa_budget_counters SET spent = 'lots'")
    with caplog.at_level(logging.ERROR), pytest.raises(IntegrationError):
        await store.usage("k")
    await LoopLocal.aclose_all()


async def test_sqlalchemy_aclose_disposes_the_engine_and_a_new_one_reopens(tmp_path):
    pytest.importorskip("aiosqlite")
    path = tmp_path / "b.db"
    store = SQLAlchemyBudgetStore(f"sqlite+aiosqlite:///{path.as_posix()}")
    await store.settle("k", "r", 0.1, expire_at=None)
    await store.aclose()
    await store.aclose()  # a second close is a no-op
    assert (await store.usage("k")).spent_usd == pytest.approx(0.1)
    await store.aclose()


async def test_sqlalchemy_sweep_failure_is_only_a_warning(tmp_path, caplog, monkeypatch):
    pytest.importorskip("aiosqlite")
    store = SQLAlchemyBudgetStore(f"sqlite+aiosqlite:///{(tmp_path / 'b.db').as_posix()}")
    engine = await store._engine()

    class Broken:
        def begin(self) -> Any:
            raise OSError("gone")

    with caplog.at_level(logging.WARNING):
        await store._sweep(Broken(), time.time())  # type: ignore[arg-type]
        await store._sweep(Broken(), time.time())  # within PURGE_EVERY_S: skipped
    assert caplog.text.count("budget store sweep failed") == 1
    await engine.dispose()
    await LoopLocal.aclose_all()


async def test_sqlalchemy_create_schema_false_uses_existing_tables(tmp_path):
    pytest.importorskip("aiosqlite")
    url = f"sqlite+aiosqlite:///{(tmp_path / 'b.db').as_posix()}"
    writer = SQLAlchemyBudgetStore(url)  # kept: aclose_all closes only live stores' engines
    await writer.settle("k", "r", 0.3, expire_at=None)
    reader = SQLAlchemyBudgetStore(url, create_schema=False)
    assert (await reader.usage("k")).spent_usd == pytest.approx(0.3)
    await LoopLocal.aclose_all()


async def test_sqlalchemy_logs_a_backend_failure_by_type(tmp_path, caplog):
    pytest.importorskip("aiosqlite")
    url = f"sqlite+aiosqlite:///{(tmp_path / 'missing' / 'b.db').as_posix()}"
    store = SQLAlchemyBudgetStore(url)
    with caplog.at_level(logging.ERROR), pytest.raises(Exception):  # noqa: B017
        await store.usage("k")
    assert "sqlalchemy budget store schema failed" in caplog.text
    await LoopLocal.aclose_all()


async def test_redis_does_not_retry_with_eval_on_other_errors():
    class Down:
        evals = 0

        async def evalsha(self, *args: Any) -> Any:
            raise ConnectionError("down")

        async def eval(self, *args: Any) -> Any:
            Down.evals += 1

    with pytest.raises(ConnectionError):
        await RedisBudgetStore(Down).usage("k")
    assert Down.evals == 0
