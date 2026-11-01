"""SQLiteApprovalStore: configuration, schema, durability and failing closed on bad rows."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

from agent_adapter.approval.persist import ApprovalStoreCorrupt
from agent_adapter.approval.sqlite_store import META, SCHEMA_VERSION, TABLE, SQLiteApprovalStore
from agent_adapter.core.errors import AdapterError, ApprovalNotFound, ConfigError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

YES = ApprovalDecision(approved=True, decided_by="ana")


def req(call_id: str = "call_1", **extra: object) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="",
        run_id="run_1",
        call_id=call_id,
        tool="pay",
        arguments={"amount": 5},
        **extra,
    )


def sql(path: Path, statement: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(path)) as conn, conn:  # closed even when the statement fails
        return conn.execute(statement, params).fetchall()


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "approvals.db"


@pytest.mark.parametrize("bad", ["", ":memory:", "file:x.db?mode=rwc", 7])
def test_path_must_name_a_file(bad):
    with pytest.raises(ConfigError, match="path"):
        SQLiteApprovalStore(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_max_records_must_be_a_positive_int(path, bad):
    with pytest.raises(ConfigError, match="max_records"):
        SQLiteApprovalStore(path, max_records=bad)


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan"), True, "5"])
def test_timeout_must_be_a_finite_positive_number(path, bad):
    with pytest.raises(ConfigError, match="timeout"):
        SQLiteApprovalStore(path, timeout=bad)


async def test_first_use_creates_a_versioned_wal_database(path, caplog):
    store = SQLiteApprovalStore(path, timeout=2)
    assert not path.exists()  # no I/O until first use
    with caplog.at_level(logging.INFO):
        await store.list()
    assert "approval database created (schema 1)" in caplog.text
    assert sql(path, "PRAGMA journal_mode") == [("wal",)]
    assert sql(path, f"SELECT value FROM {META} WHERE key = 'schema'") == [(SCHEMA_VERSION,)]
    indexes = {name for (name,) in sql(path, "SELECT name FROM sqlite_master WHERE type='index'")}
    assert f"{TABLE}_open" in indexes and store.timeout == 2.0


async def test_records_survive_a_restart(path):
    first = SQLiteApprovalStore(path)
    record = await first.create(req())
    await first.decide(record.approval_id, YES)
    restarted = SQLiteApprovalStore(path)
    assert await restarted.get(record.approval_id) == await first.get(record.approval_id)
    assert await restarted.consume(record.approval_id) is True
    assert await first.consume(record.approval_id) is False


async def test_the_open_record_is_unique_per_call_in_the_schema_itself(path):
    store = SQLiteApprovalStore(path)
    record = await store.create(req())
    with pytest.raises(sqlite3.IntegrityError):
        sql(
            path,
            f"INSERT INTO {TABLE} (approval_id, run_id, call_id, status, consumed, args_digest,"
            " tier, tier_seq, record) SELECT x'00', run_id, call_id, status, 0, args_digest,"
            f" tier, tier_seq, record FROM {TABLE}",
        )
    assert [r.approval_id for r in await store.list()] == [record.approval_id]


async def test_another_schema_version_is_refused(path):
    await SQLiteApprovalStore(path).list()
    sql(path, f"UPDATE {META} SET value = 2 WHERE key = 'schema'")
    with pytest.raises(ConfigError, match="schema 2 is not 1"):
        await SQLiteApprovalStore(path).list()


async def test_a_foreign_table_with_the_same_name_is_refused(path):
    sql(path, f"CREATE TABLE {TABLE} (id INTEGER)")
    with pytest.raises(ConfigError, match="without approval store metadata"):
        await SQLiteApprovalStore(path).create(req())
    assert sql(path, f"SELECT count(*) FROM {TABLE}") == [(0,)]


class _NoWal:
    """A connection whose file stays in its old journal mode (e.g. a network share)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, statement: str, *params: object) -> sqlite3.Cursor:
        if statement.startswith("PRAGMA journal_mode"):
            return self._conn.execute("SELECT 'delete'")
        return self._conn.execute(statement, *params)

    def close(self) -> None:
        self._conn.close()


async def test_a_database_that_cannot_use_wal_is_refused(path, monkeypatch):
    store = SQLiteApprovalStore(path)
    real = store._connect
    monkeypatch.setattr(store, "_connect", lambda: _NoWal(real()))
    with pytest.raises(ConfigError, match="WAL journal mode; the file reports 'delete'"):
        await store.list()


async def test_backend_errors_propagate_and_are_logged_without_detail(tmp_path, caplog):
    store = SQLiteApprovalStore(tmp_path / "missing" / "dir" / "approvals.db")
    with caplog.at_level(logging.ERROR), pytest.raises(FileNotFoundError):
        await store.create(req())
    assert "sqlite approval store create failed (FileNotFoundError)" in caplog.text


async def test_a_new_database_file_is_created_owner_only(path, monkeypatch):
    modes: list[tuple[str, int]] = []
    real = os.open

    def spy(name: str, flags: int, mode: int = 0o777) -> int:
        modes.append((os.fspath(name), mode))
        return real(name, flags, mode)

    monkeypatch.setattr(os, "open", spy)
    record = await SQLiteApprovalStore(path).create(req())
    assert await SQLiteApprovalStore(path).list() == [record]  # an existing file is kept as is
    assert modes == [(str(path), 0o600)] * 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes; Windows files take the folder's ACL")
async def test_the_database_and_its_wal_files_are_mode_0600(path):
    store = SQLiteApprovalStore(path)
    await store.create(req())
    for name in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        assert stat.S_IMODE(name.stat().st_mode) & 0o077 == 0, name


async def test_values_json_cannot_carry_are_refused_before_any_write(path):
    store = SQLiteApprovalStore(path)
    with pytest.raises(ToolUsageError, match="metadata is not JSON"):
        await store.create(req(metadata={"handle": object()}))
    record = await store.create(req())
    with pytest.raises(ToolUsageError, match="not JSON"):
        await store.decide(
            record.approval_id, ApprovalDecision(approved=True, edited_args={"a": (1,)})
        )
    assert (await store.get(record.approval_id)).status == "pending"
    assert await store.list() == [record]


# --- corrupt or foreign rows fail closed ----------------------------------------------------


def _tamper(path: Path, column: str, value: object) -> None:
    sql(path, f"UPDATE {TABLE} SET {column} = ?", (value,))


async def _decided(path: Path) -> tuple[SQLiteApprovalStore, str]:
    store = SQLiteApprovalStore(path)
    record = await store.create(req())
    await store.decide(record.approval_id, YES)
    return store, record.approval_id


def _swap_arguments(text: object) -> str:
    data = json.loads(str(text))
    data["request"]["arguments"] = {"amount": 5000}  # the digest still names amount 5
    return json.dumps(data)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("record", "{not json"),
        ("record", "[1, 2]"),
        ("record", '{"request": {"approval_id": "x"}}'),
        ("record", b"\x00\x01"),
        ("status", "approved\x00"),
        ("consumed", 7),
        ("run_id", "run_1"),  # a TEXT id from a foreign writer, not the store's BLOB
    ],
)
async def test_unreadable_rows_raise_a_typed_error_and_are_never_consumed(
    path, column, value, caplog
):
    store, approval_id = await _decided(path)
    _tamper(path, column, value)
    with caplog.at_level(logging.ERROR):
        for call in (store.get(approval_id), store.consume(approval_id), store.list()):
            with pytest.raises(ApprovalStoreCorrupt) as info:
                await call
            assert isinstance(info.value, AdapterError) and info.value.kind == "internal-error"
    assert "failing closed" in caplog.text or "is unreadable" in caplog.text
    assert sql(path, f"SELECT consumed FROM {TABLE}") != [(1,)]


async def test_a_record_whose_arguments_no_longer_match_its_digest_fails_closed(path):
    store, approval_id = await _decided(path)
    (text,) = sql(path, f"SELECT record FROM {TABLE}")[0]
    _tamper(path, "record", _swap_arguments(text))
    with pytest.raises(ApprovalStoreCorrupt, match="is unreadable"):
        await store.consume(approval_id)
    with pytest.raises(ApprovalStoreCorrupt):
        await store.find("run_1", "call_1")
    with pytest.raises(ApprovalStoreCorrupt):
        await store.create(req())  # the call's open row is unreadable: nothing new is stored
    assert sql(path, f"SELECT count(*), max(consumed) FROM {TABLE}") == [(1, 0)]


async def test_a_row_moved_to_another_id_fails_closed(path):
    store = SQLiteApprovalStore(path)
    one = await store.create(req("a"))
    await store.create(req("b"))
    (text,) = sql(path, f"SELECT record FROM {TABLE} WHERE call_id = ?", (b"a",))[0]
    sql(path, f"UPDATE {TABLE} SET record = ? WHERE call_id = ?", (text, b"b"))
    with pytest.raises(ApprovalStoreCorrupt, match="disagrees with its record"):
        await store.list()
    assert await store.get(one.approval_id) == one


async def test_a_broken_eviction_counter_fails_closed(path):
    store = SQLiteApprovalStore(path)
    await store.create(req("a"))
    sql(path, f"DELETE FROM {META} WHERE key = 'tick'")
    with pytest.raises(ApprovalStoreCorrupt, match="tick"):
        await store.create(req("b"))
    assert len(await store.list()) == 1  # the failed create rolled back


async def test_a_foreign_text_id_is_never_found_and_fails_a_listing_closed(path):
    store, approval_id = await _decided(path)
    _tamper(path, "approval_id", approval_id)  # TEXT, where the store writes BLOB
    with pytest.raises(ApprovalNotFound):
        await store.get(approval_id)
    with pytest.raises(ApprovalStoreCorrupt, match="<NoneType>"):
        await store.list()


class _Failing(_NoWal):
    """A connection whose setup statement fails; it must still be closed."""

    closed = False

    def execute(self, statement: str, *params: object) -> sqlite3.Cursor:
        raise sqlite3.OperationalError("disk I/O error")

    def close(self) -> None:
        type(self).closed = True
        super().close()


async def test_a_connection_whose_setup_fails_is_closed(path, monkeypatch):
    real = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _Failing(real(*a, **k)))
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        await SQLiteApprovalStore(path).list()
    assert _Failing.closed
