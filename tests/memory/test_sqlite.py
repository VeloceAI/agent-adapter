"""SQLiteMemory: add, search (FTS5, LIKE, vectors), get, delete, export, schema and failures."""

import asyncio
import sqlite3
import threading
from contextlib import closing

import pytest

from agent_adapter.embeddings import CallableEmbedder
from agent_adapter.errors import ConfigError, IntegrationError, ModelError, ToolUsageError
from agent_adapter.memory import SQLiteMemory
from agent_adapter.memory import _sqlite_sql as sql
from tests.memory.helpers import ALICE, ALICE_ALL, ALICE_S2, BOB, embedder, rec


@pytest.fixture(params=[True, False], ids=["fts5", "like"])
def memory(request, tmp_path, monkeypatch):
    if not request.param:
        monkeypatch.setattr(sql, "fts5_available", lambda conn: False)
    return SQLiteMemory(tmp_path / "memory.db")


async def test_add_fills_ids_and_times_and_search_finds_by_keyword(memory):
    stored = await memory.add([rec(ALICE, "Likes the modern art museum"), rec(ALICE, "Vegan")])
    assert all(r.id.startswith("mem_") and r.created_at for r in stored)
    hits = await memory.search("museum tickets", scope=ALICE)
    assert [h.record.text for h in hits] == ["Likes the modern art museum"]
    assert hits[0].score > 0


async def test_search_ranks_more_matching_words_first(memory):
    await memory.add([rec(ALICE, "train to the coast"), rec(ALICE, "train and museum trip")])
    hits = await memory.search("train museum trip", scope=ALICE, k=2)
    assert hits[0].record.text == "train and museum trip"


async def test_an_empty_query_lists_the_newest_first(memory):
    await memory.add([rec(ALICE, "first")])
    await memory.add([rec(ALICE, "second")])
    hits = await memory.search("", scope=ALICE, k=1)
    assert [h.record.text for h in hits] == ["second"]
    assert await memory.search("?!", scope=ALICE) == []  # no words and no embedder


async def test_query_text_is_never_sql_or_fts_syntax(memory):
    await memory.add([rec(ALICE, "100% sure about snake_case")])
    for query in ['" OR 1=1 --', "NEAR(a b)", "*", "snake_case", "100%", "'; DROP TABLE x"]:
        await memory.search(query, scope=ALICE)
    assert [h.record.text for h in await memory.search("snake_case", scope=ALICE)]
    assert len(await memory.export(ALICE_ALL)) == 1


async def test_like_fallback_escapes_wildcards(tmp_path, monkeypatch):
    monkeypatch.setattr(sql, "fts5_available", lambda conn: False)
    memory = SQLiteMemory(tmp_path / "like.db")
    await memory.add([rec(ALICE, "plain words"), rec(ALICE, "under_score")])
    assert [h.record.text for h in await memory.search("under_score", scope=ALICE)] == [
        "under_score"
    ]


async def test_filters_match_metadata_exactly(memory):
    await memory.add(
        [
            rec(ALICE, "museum visit", metadata={"kind": "visit", "n": 2, "paid": True}),
            rec(ALICE, "museum shop", metadata={"kind": "shop", "n": 2.5, "paid": False}),
        ]
    )
    assert [
        h.record.text for h in await memory.search("museum", scope=ALICE, filters={"kind": "visit"})
    ] == ["museum visit"]
    assert [
        h.record.text for h in await memory.search("museum", scope=ALICE, filters={"n": 2.5})
    ] == ["museum shop"]
    assert [
        h.record.text for h in await memory.search("museum", scope=ALICE, filters={"paid": True})
    ] == ["museum visit"]
    assert await memory.search("museum", scope=ALICE, filters={"n": True}) == []
    assert await memory.search("museum", scope=ALICE, filters={"kind": 1}) == []
    assert len(await memory.search("", scope=ALICE, filters={"paid": False})) == 1


async def test_get_reads_one_record_and_respects_scope(memory):
    (stored,) = await memory.add([rec(ALICE, "note")])
    assert (await memory.get(stored.id)).text == "note"
    assert (await memory.get(stored.id, scope=ALICE_ALL)).text == "note"
    assert await memory.get(stored.id, scope=BOB) is None
    assert await memory.get(stored.id, scope=ALICE_S2) is None
    assert await memory.get("mem_missing") is None


async def test_delete_by_ids_scope_or_both(memory):
    a1, a2 = await memory.add([rec(ALICE, "one"), rec(ALICE_S2, "two")])
    (b1,) = await memory.add([rec(BOB, "bob's")])
    assert await memory.delete([b1.id], scope=ALICE_ALL) == 0  # not alice's
    assert await memory.delete([a1.id, "mem_missing"]) == 1
    assert await memory.delete([]) == 0
    assert await memory.delete(scope=ALICE_ALL) == 1
    assert await memory.export(ALICE_ALL) == []
    assert [r.id for r in await memory.export(BOB)] == [b1.id]
    assert a2.id not in {r.id for r in await memory.export(BOB)}
    with pytest.raises(ToolUsageError):
        await memory.delete()


async def test_export_returns_everything_in_scope_oldest_first(memory):
    await memory.add([rec(ALICE, "a"), rec(ALICE_S2, "b"), rec(BOB, "c")])
    assert [r.text for r in await memory.export(ALICE_ALL)] == ["a", "b"]
    assert [r.text for r in await memory.export(ALICE_S2)] == ["b"]


async def test_a_stored_id_is_never_overwritten(memory):
    await memory.add([rec(ALICE, "mine", id="mem_fixed")])
    with pytest.raises(ToolUsageError):
        await memory.add([rec(BOB, "steal", id="mem_fixed")])
    assert (await memory.get("mem_fixed")).scope == ALICE
    assert await memory.add([]) == []


async def test_vectors_find_related_words_and_fuse_with_keywords(tmp_path):
    memory = SQLiteMemory(tmp_path / "vec.db", embedder=embedder())
    await memory.add([rec(ALICE, "has a cat"), rec(ALICE, "pizza for lunch")])
    hits = await memory.search("kitten", scope=ALICE, k=1)
    assert [h.record.text for h in hits] == ["has a cat"]
    hits = await memory.search("pizza", scope=ALICE, k=2)
    assert hits[0].record.text == "pizza for lunch"


async def test_a_failing_embedder_fails_add_but_search_falls_back(tmp_path, caplog):
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("down")
        return [[1.0, 0.0] for _ in texts]

    memory = SQLiteMemory(tmp_path / "flaky.db", embedder=CallableEmbedder(flaky))
    await memory.add([rec(ALICE, "museum ticket")])
    hits = await memory.search("museum", scope=ALICE)
    assert [h.record.text for h in hits] == ["museum ticket"]
    assert "keyword-only" in caplog.text and "down" not in caplog.text
    with pytest.raises(ModelError):
        await memory.add([rec(ALICE, "more")])
    assert len(await memory.export(ALICE)) == 1  # nothing half-written


async def test_an_embedder_returning_the_wrong_count_is_refused(tmp_path):
    class Short:
        provider, model = "x", "y"

        async def embed(self, texts, *, ctx=None):
            from agent_adapter.embeddings import EmbeddingResult

            return EmbeddingResult(vectors=[[1.0]])

    memory = SQLiteMemory(tmp_path / "short.db", embedder=Short())
    with pytest.raises(ModelError):
        await memory.add([rec(ALICE, "a"), rec(ALICE, "b")])


async def test_vectors_of_another_dimension_are_skipped(tmp_path):
    path = tmp_path / "dims.db"
    await SQLiteMemory(path, embedder=CallableEmbedder(lambda t: [[1.0, 0.0]] * len(t))).add(
        [rec(ALICE, "old vector")]
    )
    memory = SQLiteMemory(path, embedder=embedder())
    assert await memory.search("unrelated", scope=ALICE) == []


@pytest.mark.parametrize("path", [":memory:", "", "file:x.db", 3])
def test_bad_paths_are_refused(path):
    with pytest.raises(ConfigError):
        SQLiteMemory(path)


@pytest.mark.parametrize(
    "option", [{"timeout": 0}, {"timeout": True}, {"embedder": object()}, {"retention_days": -1}]
)
def test_bad_options_are_refused(tmp_path, option):
    with pytest.raises(ConfigError):
        SQLiteMemory(tmp_path / "x.db", **option)


async def test_a_foreign_table_or_schema_is_refused(tmp_path):
    path = tmp_path / "foreign.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE aa_memory (x)")
    with pytest.raises(ConfigError):
        await SQLiteMemory(path).export(ALICE)
    other = tmp_path / "v9.db"
    await SQLiteMemory(other).export(ALICE)
    with closing(sqlite3.connect(other)) as conn, conn:
        conn.execute("UPDATE aa_memory_meta SET value = 9 WHERE key = 'schema'")
    with pytest.raises(ConfigError):
        await SQLiteMemory(other).export(ALICE)


async def test_an_fts_file_needs_fts_on_reopen(tmp_path, monkeypatch):
    path = tmp_path / "fts.db"
    await SQLiteMemory(path).add([rec(ALICE, "kept")])
    monkeypatch.setattr(sql, "fts5_available", lambda conn: False)
    with pytest.raises(ConfigError):
        await SQLiteMemory(path).export(ALICE)


async def test_a_like_file_stays_like_when_fts_appears(tmp_path, monkeypatch):
    path = tmp_path / "like-first.db"
    monkeypatch.setattr(sql, "fts5_available", lambda conn: False)
    await SQLiteMemory(path).add([rec(ALICE, "kept words")])
    monkeypatch.undo()
    assert [h.record.text for h in await SQLiteMemory(path).search("kept", scope=ALICE)]


async def test_a_corrupt_row_raises_integration_error_without_its_text(tmp_path, caplog):
    path = tmp_path / "bad.db"
    memory = SQLiteMemory(path)
    (stored,) = await memory.add([rec(ALICE, "secret words")])
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE aa_memory SET record = '{broken'")
    with pytest.raises(IntegrationError):
        await memory.get(stored.id)
    assert "secret" not in caplog.text


def test_fts5_probe_reports_true_on_this_build():
    with closing(sqlite3.connect(":memory:")) as conn, conn:
        assert sql.fts5_available(conn) is True


async def test_parallel_writers_from_threads_and_tasks_all_land(tmp_path):
    path = tmp_path / "busy.db"
    memory = SQLiteMemory(path)
    await asyncio.gather(*(memory.add([rec(ALICE, f"task {i}")]) for i in range(20)))

    def worker(n: int) -> None:
        other = SQLiteMemory(path)  # another instance, as another process would
        for i in range(5):
            other.add_sync([rec(BOB, f"thread {n} {i}")])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(await memory.export(ALICE)) == 20
    assert len(await memory.export(BOB)) == 20


async def test_keyword_search_matches_word_forms(memory):
    await memory.add([rec(ALICE, "Prefers aisle seats")])
    assert [h.record.text for h in await memory.search("seat", scope=ALICE)] == [
        "Prefers aisle seats"
    ]


async def test_a_non_adapter_embedder_error_fails_add_but_not_search(tmp_path, caplog):
    class Broken:
        provider, model = "x", "y"

        def __init__(self):
            self.calls = 0

        async def embed(self, texts, *, ctx=None):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("secret text in message")
            from agent_adapter.embeddings import EmbeddingResult

            return EmbeddingResult(vectors=[[1.0, 0.0]] * len(texts))

    memory = SQLiteMemory(tmp_path / "broken.db", embedder=Broken())
    await memory.add([rec(ALICE, "museum day")])
    assert [h.record.text for h in await memory.search("museum", scope=ALICE)] == ["museum day"]
    with pytest.raises(ModelError):
        await memory.add([rec(ALICE, "more")])
    assert "secret text" not in caplog.text


async def test_purge_uses_the_indexes(tmp_path):
    path = tmp_path / "plan.db"
    await SQLiteMemory(path).add([rec(ALICE, "x")])
    with closing(sqlite3.connect(path)) as conn, conn:
        plans = [
            " ".join(str(c) for c in row)
            for query in ("expires_at <= ?", "created_at <= ?")
            for row in conn.execute(
                f"EXPLAIN QUERY PLAN SELECT seq FROM aa_memory WHERE {query}", (0,)
            )
        ]
    assert all("USING" in plan and "INDEX" in plan for plan in plans), plans
