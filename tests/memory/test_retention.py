"""Retention and privacy: expiry on read, purge on write, delete per user, export."""

import sqlite3
from contextlib import closing
from datetime import timedelta

from agent_adapter.memory import SQLiteMemory
from tests.memory.helpers import ALICE, ALICE_ALL, ALICE_S2, BOB, days_ago, rec


def _rows(path) -> int:
    with closing(sqlite3.connect(path)) as conn, conn:
        (count,) = conn.execute("SELECT count(*) FROM aa_memory").fetchone()
        (fts,) = conn.execute("SELECT count(*) FROM aa_memory_fts").fetchone()
    assert count == fts
    return count


async def test_expired_records_are_never_read_and_purged_on_the_next_write(tmp_path):
    path = tmp_path / "m.db"
    memory = SQLiteMemory(path)
    (gone,) = await memory.add([rec(ALICE, "old museum note", expires_at=days_ago(1))])
    await memory.add([rec(ALICE, "new museum note")])
    assert [h.record.text for h in await memory.search("museum", scope=ALICE)] == [
        "new museum note"
    ]
    assert await memory.get(gone.id) is None
    assert [r.text for r in await memory.export(ALICE)] == ["new museum note"]
    assert _rows(path) == 1  # export purged it


async def test_retention_days_sets_expires_at_and_hides_older_records(tmp_path):
    path = tmp_path / "r.db"
    await SQLiteMemory(path).add([rec(ALICE, "from long ago", created_at=days_ago(40))])
    memory = SQLiteMemory(path, retention_days=30)
    (fresh,) = await memory.add([rec(ALICE, "fresh")])
    assert fresh.expires_at is not None
    assert round((fresh.expires_at - fresh.created_at).total_seconds()) == 30 * 86400
    assert [h.record.text for h in await memory.search("", scope=ALICE)] == ["fresh"]
    assert _rows(path) == 1  # the add purged the 40-day-old record


async def test_an_explicit_expires_at_wins_over_retention(tmp_path):
    memory = SQLiteMemory(tmp_path / "e.db", retention_days=30)
    (kept,) = await memory.add([rec(ALICE, "short", expires_at=days_ago(-1))])
    assert kept.expires_at - kept.created_at < timedelta(days=2)
    assert (await memory.get(kept.id)).text == "short"


async def test_purge_reports_how_many_it_removed(tmp_path):
    path = tmp_path / "p.db"
    memory = SQLiteMemory(path, retention_days=10)
    await SQLiteMemory(path).add(
        [rec(ALICE, "a", created_at=days_ago(20)), rec(BOB, "b", created_at=days_ago(20))]
    )
    await SQLiteMemory(path).add([rec(ALICE, "c")])
    assert await memory.purge() == 2
    assert await memory.purge() == 0
    assert memory.purge_sync() == 0


async def test_delete_for_a_user_removes_every_session_and_nothing_else(tmp_path):
    path = tmp_path / "d.db"
    memory = SQLiteMemory(path)
    await memory.add([rec(ALICE, "one"), rec(ALICE_S2, "two"), rec(BOB, "bob")])
    assert await memory.delete(scope=ALICE_ALL) == 2
    assert await memory.export(ALICE_ALL) == []
    assert [r.text for r in await memory.export(BOB)] == ["bob"]
    assert _rows(path) == 1


async def test_export_is_complete_for_a_privacy_request(tmp_path):
    memory = SQLiteMemory(tmp_path / "x.db")
    texts = [f"fact {i}" for i in range(30)]
    await memory.add(
        [rec(ALICE if i % 2 else ALICE_S2, t, metadata={"i": i}) for i, t in enumerate(texts)]
    )
    exported = await memory.export(ALICE_ALL)
    assert sorted(r.text for r in exported) == sorted(texts)
    assert {r.metadata["i"] for r in exported} == set(range(30))
