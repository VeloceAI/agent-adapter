"""Erasure leaves no text in the SQLite file: rows, free pages, FTS segments and the WAL."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent_adapter.memory import SQLiteMemory
from agent_adapter.memory import _sqlite_sql as sql
from tests.memory.helpers import ALICE, ALICE_ALL, ALICE_S2, BOB, days_ago, rec

MARK = "zebracrossing"


def _copies(path, word: str = MARK) -> int:
    """How many times `word` appears in the database file and its WAL."""
    found = 0
    for suffix in ("", "-wal"):
        part = path.with_name(path.name + suffix)
        if part.exists():
            found += part.read_bytes().count(word.encode())
    return found


@pytest.fixture(params=["fts5-secure", "fts5-optimize", "like"])
def memory(request, tmp_path, monkeypatch):
    if request.param == "fts5-optimize":
        monkeypatch.setattr(sql, "secure_fts", lambda conn: False)  # SQLite before 3.42
    if request.param == "like":
        monkeypatch.setattr(sql, "fts5_available", lambda conn: False)
    return SQLiteMemory(tmp_path / "erase.db")


async def _fill(memory):
    await memory.add([rec(ALICE, f"{MARK} visit {i} to the gallery") for i in range(25)])
    await memory.add([rec(ALICE_S2, f"{MARK} note {i}", metadata={"m": MARK}) for i in range(25)])
    await memory.add([rec(BOB, "bob keeps his trains")])


async def test_erasing_a_user_removes_every_byte_of_their_text(memory):
    await _fill(memory)
    path = Path(memory.path)
    idle = sqlite3.connect(path)  # another process keeps the file (and its WAL) open
    try:
        assert _copies(path) > 0
        assert await memory.delete(scope=ALICE_ALL) == 50
        assert _copies(path) == 0
    finally:
        idle.close()
    assert _copies(path, "trains") > 0  # bob's data is untouched
    assert [h.record.text for h in await memory.search("trains", scope=BOB)]


async def test_deleting_by_id_and_purging_expired_leave_no_bytes(memory, tmp_path):
    path = Path(memory.path)
    (one,) = await memory.add([rec(ALICE, f"{MARK} single")])
    assert await memory.delete([one.id]) == 1
    assert _copies(path) == 0
    await memory.add([rec(ALICE, f"{MARK} stale", expires_at=days_ago(1))])
    assert await memory.purge() == 1
    assert _copies(path) == 0


async def test_retention_purge_on_write_leaves_no_bytes(tmp_path):
    path = tmp_path / "ret.db"
    await SQLiteMemory(path).add([rec(ALICE, f"{MARK} old", created_at=days_ago(40))])
    await SQLiteMemory(path, retention_days=30).add([rec(ALICE, "fresh")])
    assert _copies(path) == 0


async def test_an_existing_file_gets_secure_delete_on_open(tmp_path):
    path = tmp_path / "old.db"
    await SQLiteMemory(path).add([rec(ALICE, "x")])
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DELETE FROM aa_memory_fts_config WHERE k = 'secure-delete'")
    reopened = SQLiteMemory(path)
    await reopened.export(ALICE)
    assert reopened._fts_secure is True


async def test_a_blocked_checkpoint_is_logged(tmp_path, caplog):
    path = tmp_path / "busy.db"
    memory = SQLiteMemory(path)
    await memory.add([rec(ALICE, f"{MARK} a")])
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM aa_memory").fetchone()
    try:
        await memory.delete(scope=ALICE_ALL)
    finally:
        reader.close()
    assert "checkpoint was blocked" in caplog.text
