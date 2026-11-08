"""Backend specifics: configuration, schema checks, corrupt rows (each store fails closed)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import CheckpointStoreCorrupt, ConfigError, ToolUsageError
from agent_adapter.exec.checkpoints import (
    CallRecord,
    Checkpoint,
    RedisCheckpointStore,
    SQLAlchemyCheckpointStore,
    SQLiteCheckpointStore,
    _port,
)

CP = Checkpoint(kind="flow", name="f", status="running", segment="s", data="{}")


# --- SQLite ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [":memory:", "", "file:x.db", 5])
def test_sqlite_needs_a_file_path(bad: Any) -> None:
    with pytest.raises(ConfigError):
        SQLiteCheckpointStore(bad)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), True, "5"])
def test_sqlite_needs_a_positive_timeout(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(ConfigError):
        SQLiteCheckpointStore(tmp_path / "c.db", timeout=bad)


async def test_sqlite_refuses_a_foreign_table_and_another_schema(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE aa_checkpoints (x)")
    with pytest.raises(ConfigError, match="without checkpoint store metadata"):
        await SQLiteCheckpointStore(path).latest("r")
    other = tmp_path / "o.db"
    await SQLiteCheckpointStore(other).latest("r")
    with closing(sqlite3.connect(other)) as conn, conn:
        conn.execute("UPDATE aa_checkpoint_meta SET value = 9 WHERE key = 'schema'")
    with pytest.raises(ConfigError, match="schema 9"):
        await SQLiteCheckpointStore(other).latest("r")


async def test_sqlite_reopens_an_existing_file(tmp_path: Path) -> None:
    await SQLiteCheckpointStore(tmp_path / "c.db").save("r", 0, CP)
    assert (await SQLiteCheckpointStore(tmp_path / "c.db").latest("r")).step == 0  # type: ignore[union-attr]


async def test_sqlite_corrupt_rows_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    store = SQLiteCheckpointStore(path)
    await store.save("r", 0, CP)
    await store.record_call("r", "s", CallRecord(key="k", tool="t", state="started"))
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE aa_checkpoints SET status = 'done'")
    with pytest.raises(CheckpointStoreCorrupt, match="disagrees"):
        await store.latest("r")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE aa_checkpoints SET record = 'not json', status = 'running'")
    with pytest.raises(CheckpointStoreCorrupt, match="unreadable"):
        await store.latest("r")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE aa_checkpoint_calls SET call_key = x'00'")
    with pytest.raises(CheckpointStoreCorrupt, match="disagrees"):
        await store.calls("r", "s")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE aa_checkpoints SET step = 'x'")
    with pytest.raises(CheckpointStoreCorrupt):
        await store.save("r", 1, CP)


async def test_sqlite_backend_errors_propagate(tmp_path: Path, caplog: Any) -> None:
    store = SQLiteCheckpointStore(tmp_path / "missing-dir" / "c.db")
    with pytest.raises((sqlite3.Error, OSError)):
        await store.latest("r")
    assert "sqlite checkpoint store latest failed" in caplog.text


# --- Redis ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{"prefix": "nocolon"}, {"prefix": 5}, {"ttl_s": 0}, {"ttl_s": float("nan")}, {"ttl_s": True}],
)
def test_redis_options_are_checked(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        RedisCheckpointStore(lambda: None, **options)


def test_redis_needs_a_client_factory() -> None:
    with pytest.raises(ConfigError):
        RedisCheckpointStore("redis://x")  # type: ignore[arg-type]


def fake_redis() -> tuple[Any, Any]:
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    server = fakeredis.FakeServer()
    return server, lambda: fakeredis.FakeAsyncRedis(server=server)


async def test_redis_corrupt_hashes_fail_closed() -> None:
    _, factory = fake_redis()
    store = RedisCheckpointStore(factory, ttl_s=None)
    await store.save("r", 0, CP)
    await store.record_call("r", "s", CallRecord(key="k", tool="t", state="started"))
    client = factory()
    run_key = b"aa:checkpoint:run:" + b"r".hex().encode()
    await client.hset(run_key, "status", "done")
    with pytest.raises(CheckpointStoreCorrupt, match="disagrees"):
        await store.latest("r")
    await client.hset(run_key, "step", "x")
    with pytest.raises(CheckpointStoreCorrupt, match="unreadable"):
        await store.save("r", 1, CP)
    calls_key = b"aa:checkpoint:calls:" + b"r".hex().encode()
    field = b"s".hex().encode() + b":" + b"k".hex().encode()
    await client.hset(calls_key, field, '{"key": "other", "tool": "t", "state": "done"}')
    with pytest.raises(CheckpointStoreCorrupt, match="another key"):
        await store.calls("r", "s")
    await store.aclose()


async def test_redis_skips_runs_that_expired_from_the_index() -> None:
    _, factory = fake_redis()
    store = RedisCheckpointStore(factory)
    await store.save("gone", 0, CP)
    await store.save("kept", 0, CP)
    await factory().delete(b"aa:checkpoint:run:" + b"gone".hex().encode())
    assert [s.run_id for s in await store.list_unfinished()] == ["kept"]
    assert await factory().zrange(b"aa:checkpoint:unfinished", 0, -1) == [b"kept"]  # forgotten


async def test_redis_keys_expire_but_a_paused_run_does_not() -> None:
    _, factory = fake_redis()
    store = RedisCheckpointStore(factory, ttl_s=60)
    client = factory()
    await store.save("r", 0, CP)
    await store.record_call("r", "s", CallRecord(key="k", tool="t", state="started"))
    run_key = b"aa:checkpoint:run:" + b"r".hex().encode()
    calls_key = b"aa:checkpoint:calls:" + b"r".hex().encode()
    assert 0 < await client.pttl(run_key) <= 60_000
    assert 0 < await client.pttl(calls_key) <= 60_000
    await store.save("r", 1, CP.model_copy(update={"status": "paused", "segment": "s2"}))
    assert await client.pttl(run_key) == -1  # waits for a person: never expires
    assert await store.calls("r", "s") == []  # the old segment's calls are gone


class _Broken:
    async def eval(self, *args: Any) -> Any:
        raise ConnectionError("down")

    async def hmget(self, *args: Any) -> Any:
        raise ConnectionError("down")

    async def zrange(self, *args: Any) -> Any:
        raise ConnectionError("down")


@pytest.mark.parametrize("op", ["save", "latest", "list", "calls"])
async def test_redis_backend_errors_propagate(op: str, caplog: Any) -> None:
    store = RedisCheckpointStore(_Broken)
    calls = {
        "save": lambda: store.save("r", 0, CP),
        "latest": lambda: store.latest("r"),
        "list": lambda: store.list_unfinished(),
        "calls": lambda: store.calls("r", "s"),
    }
    with pytest.raises(ConnectionError):
        await calls[op]()  # type: ignore[no-untyped-call]
    assert "redis checkpoint store" in caplog.text


# --- SQLAlchemy -----------------------------------------------------------------------------


def sa_url(tmp_path: Path) -> str:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    return f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"


@pytest.mark.parametrize("bad", ["", 5, None])
def test_sqlalchemy_needs_a_url_or_factory(bad: Any) -> None:
    pytest.importorskip("sqlalchemy")
    with pytest.raises(ConfigError):
        SQLAlchemyCheckpointStore(bad)


def test_sqlalchemy_table_name_is_an_identifier(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        SQLAlchemyCheckpointStore(sa_url(tmp_path), table="bad name")


async def test_sqlalchemy_checks_the_tables_it_finds(tmp_path: Path) -> None:
    url = sa_url(tmp_path)
    plain = SQLiteCheckpointStore(tmp_path / "s.db")  # another schema, same table name
    await plain.save("r", 0, CP)
    store = SQLAlchemyCheckpointStore(url, create_schema=False)
    with pytest.raises(ConfigError, match="lacks checkpoint store columns"):
        await store.latest("r")
    await store.aclose()


async def test_sqlalchemy_refuses_other_dialects(tmp_path: Path) -> None:
    url = sa_url(tmp_path)

    class Dialect:
        name = "mysql"

    class Engine:
        dialect = Dialect()

        async def dispose(self) -> None:
            return None

    store = SQLAlchemyCheckpointStore(lambda: Engine())  # type: ignore[arg-type,return-value]
    with pytest.raises(ConfigError, match="mysql"):
        await store.latest("r")
    await store.aclose()
    del url


async def test_sqlalchemy_corrupt_rows_fail_closed(tmp_path: Path) -> None:
    from sqlalchemy import text

    store = SQLAlchemyCheckpointStore(sa_url(tmp_path))
    await store.save("r", 0, CP)
    await store.record_call("r", "s", CallRecord(key="k", tool="t", state="started"))
    engine = await store._engine()
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE aa_checkpoints SET status = 'done'"))
    with pytest.raises(CheckpointStoreCorrupt):
        await store.latest("r")
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE aa_checkpoints_calls SET call_key = x'00'"))
    with pytest.raises(CheckpointStoreCorrupt, match="another key"):
        await store.calls("r", "s")
    await store.aclose()


def test_sqlalchemy_missing_is_a_typed_error(monkeypatch: Any) -> None:
    import builtins

    real = builtins.__import__

    def no_sqlalchemy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sqlalchemy":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sqlalchemy)
    from agent_adapter.core.errors import MissingDependency

    with pytest.raises(MissingDependency, match="agent-adapter\\[sqlalchemy\\]"):
        SQLAlchemyCheckpointStore("sqlite+aiosqlite:///x.db")


# --- records --------------------------------------------------------------------------------


def test_stored_rows_check_their_columns() -> None:
    text = CP.model_dump_json()
    assert _port.stored(b"r", b"3", text, b"running").step == 3
    for run, step, status in (
        (5, 0, None),
        (b"r", -1, None),
        (b"r", "x", None),
        (b"\xff", 0, None),
    ):
        with pytest.raises(CheckpointStoreCorrupt):
            _port.stored(run, step, text, status)
    with pytest.raises(CheckpointStoreCorrupt):
        _port.stored("r", 0, 42)


def test_a_call_record_that_is_not_json_is_refused() -> None:
    record = CallRecord(key="k", tool="t", state="done", result={"v": object()})
    with pytest.raises(ToolUsageError):
        _port.encode_call(record)
