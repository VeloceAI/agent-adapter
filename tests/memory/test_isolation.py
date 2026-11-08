"""Scope isolation, adversarially: no search, get, delete or export crosses a user."""

import pytest

from agent_adapter.memory import CallableMemory, MemoryScope, SQLiteMemory
from agent_adapter.memory import _sqlite_sql as sql
from tests.memory.helpers import ALICE, ALICE_ALL, BOB, BOB_ALL, embedder, rec

TRICKY_USERS = ["", "alice ", "ALICE", "alice%", "al_ce", "alice' OR '1'='1", "*", "alice\x00"]


@pytest.fixture(params=["fts5", "like", "vectors"])
def memory(request, tmp_path, monkeypatch):
    if request.param == "like":
        monkeypatch.setattr(sql, "fts5_available", lambda conn: False)
    if request.param == "vectors":
        return SQLiteMemory(tmp_path / "iso.db", embedder=embedder())
    return SQLiteMemory(tmp_path / "iso.db")


async def test_lookalike_users_never_see_alices_records(memory):
    await memory.add([rec(ALICE, "alice has a cat and likes the museum")])
    for user in TRICKY_USERS:
        scope = MemoryScope(user_id=user, session_id="s1") if user else MemoryScope(session_id="s1")
        for query in ["", "cat", "museum", "alice", "%", "*"]:
            assert await memory.search(query, scope=scope, k=100) == [], (user, query)
        assert await memory.export(scope) == []
        assert await memory.delete(scope=scope) == 0
    assert len(await memory.export(ALICE)) == 1


async def test_another_users_id_cannot_be_read_or_deleted_through_a_scope(memory):
    (secret,) = await memory.add([rec(ALICE, "alice's pet is a cat")])
    assert await memory.get(secret.id, scope=BOB_ALL) is None
    assert await memory.delete([secret.id], scope=BOB_ALL) == 0
    assert (await memory.get(secret.id, scope=ALICE_ALL)).text == "alice's pet is a cat"


async def test_same_session_id_for_two_users_stays_apart(memory):
    await memory.add([rec(ALICE, "pizza for alice"), rec(BOB, "pizza for bob")])
    assert [h.record.text for h in await memory.search("pizza", scope=ALICE, k=10)] == [
        "pizza for alice"
    ]
    assert [h.record.text for h in await memory.search("pizza", scope=BOB, k=10)] == [
        "pizza for bob"
    ]


async def test_filters_cannot_widen_a_scope(memory):
    await memory.add([rec(BOB, "bob note", metadata={"user_id": "alice"})])
    assert await memory.search("", scope=ALICE, filters={"user_id": "alice"}) == []


async def test_a_leaky_backend_is_caught_by_the_shared_checks(caplog):
    leaked = [rec(BOB, "bob's secret", id="m_bob"), rec(ALICE, "alice's", id="m_alice")]
    memory = CallableMemory(
        add=lambda records: records,
        search=lambda query, scope, k, filters: leaked,
        get=lambda id, scope: leaked[0],
        export=lambda scope: leaked,
    )
    hits = await memory.search("secret", scope=ALICE_ALL)
    assert [h.record.id for h in hits] == ["m_alice"]
    assert await memory.get("m_bob", scope=ALICE_ALL) is None
    assert await memory.get("m_alice") is None  # the backend answered with another id
    assert [r.id for r in await memory.export(ALICE_ALL)] == ["m_alice"]
    assert "outside the scope" in caplog.text and "secret" not in caplog.text
