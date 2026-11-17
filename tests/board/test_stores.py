"""Every BoardStore behaves the same; SQLite boards are shared by processes."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.board import (
    Board,
    BoardStore,
    Entry,
    MemoryBoardStore,
    RedisBoardStore,
    SQLiteBoardStore,
)
from agent_adapter.core.errors import ConfigError, IntegrationError
from agent_adapter.runtime import run_sync
from tests.work import run_child

CAPS = {"max_entries": 3, "max_bytes": 10_000}


def entry(text: str, by: str = "a") -> Entry:
    return Entry(id=f"bd_{text}", at=1.0, by=by, kind="finding", text=text)


def fake_redis() -> Callable[[], Any]:
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    server = fakeredis.FakeServer()
    return lambda: fakeredis.FakeAsyncRedis(server=server)


@pytest.fixture(params=["memory", "sqlite", "redis"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> BoardStore:
    if request.param == "memory":
        return MemoryBoardStore()
    if request.param == "sqlite":
        return SQLiteBoardStore(tmp_path / "board.db")
    return RedisBoardStore(fake_redis())


async def test_append_numbers_entries_and_read_since(store: BoardStore):
    assert isinstance(store, BoardStore)
    first = await store.append("s", entry("one"), **CAPS)
    second = await store.append("s", entry("two"), **CAPS)
    assert (first.seq, second.seq) == (1, 2)  # type: ignore[union-attr]
    assert [e.text for e in await store.read("s")] == ["one", "two"]
    assert [e.seq for e in await store.read("s", since=1)] == [2]
    assert await store.read("other") == []


async def test_caps_refuse_an_entry(store: BoardStore):
    for text in ("a", "b", "c"):
        assert await store.append("s", entry(text), **CAPS) is not None
    assert await store.append("s", entry("d"), **CAPS) is None
    tiny = {"max_entries": 10, "max_bytes": entry("x").size() + 5}
    assert await store.append("t", entry("x"), **tiny) is not None
    assert await store.append("t", entry("y"), **tiny) is None


async def test_first_claim_wins_and_drop_forgets(store: BoardStore):
    assert await store.claim("s", "job", "a") == ("a", True)
    assert await store.claim("s", "job", "b") == ("a", False)
    assert await store.claim("s", "job", "a") == ("a", False)
    await store.append("s", entry("one"), **CAPS)
    await store.drop("s")
    assert await store.read("s") == []
    assert await store.claim("s", "job", "b") == ("b", True)


async def test_board_runs_on_every_store(store: BoardStore):
    b = Board(store, scope="run:x")
    await b.post("finding", "found", by="a")
    claim = await b.claim("k", by="b")
    assert claim.won and [e.kind for e in await b.read()] == ["finding", "claim"]


def test_memory_store_drops_the_least_recent_board():
    store = MemoryBoardStore(max_scopes=2)
    for scope in ("a", "b", "c"):
        run_sync(store.append(scope, entry(scope), **CAPS))
    assert run_sync(store.read("a")) == []
    assert len(run_sync(store.read("c"))) == 1
    with pytest.raises(ConfigError):
        MemoryBoardStore(max_scopes=0)


@pytest.mark.parametrize("options", [{"prefix": "x"}, {"prefix": "a{b}:"}, {"ttl_s": 0}])
def test_redis_options_are_checked(options: dict[str, Any]):
    with pytest.raises(ConfigError):
        RedisBoardStore(lambda: None, **options)
    with pytest.raises(ConfigError):
        RedisBoardStore("redis://x")  # type: ignore[arg-type]


@pytest.mark.parametrize("path", [":memory:", "", "file:x"])
def test_sqlite_needs_a_file(path: str):
    with pytest.raises(ConfigError):
        SQLiteBoardStore(path)
    with pytest.raises(ConfigError):
        SQLiteBoardStore("x.db", timeout=0)


async def test_unreadable_rows_fail_loudly(tmp_path: Path):
    factory = fake_redis()
    redis_store = RedisBoardStore(factory)
    await factory().rpush(redis_store.keys_of("s")[0], "not json")
    with pytest.raises(IntegrationError):
        await redis_store.read("s")
    sqlite_store = SQLiteBoardStore(tmp_path / "b.db")
    await sqlite_store.append("s", entry("one"), **CAPS)
    conn = sqlite3.connect(tmp_path / "b.db")
    try:
        conn.execute("UPDATE aa_board_entries SET body = 'nope'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(IntegrationError):
        await sqlite_store.read("s")


CHILD = """import asyncio, json
from agent_adapter.board import Board, SQLiteBoardStore

async def main():
    board = Board(SQLiteBoardStore({path!r}, timeout=60), scope="run:shared")
    won = 0
    for i in range(20):
        claim = await board.claim("job:" + str(i), by={me!r})
        won += claim.won
        await board.post("finding", {me!r} + " " + str(i), by={me!r})
    return won

print(json.dumps({{"won": asyncio.run(main())}}))
"""


def test_two_processes_share_one_sqlite_board(tmp_path: Path):
    path = str(tmp_path / "board.db")
    won: dict[str, int] = {}
    errors: list[BaseException] = []

    def one(me: str) -> None:
        try:
            won[me] = run_child(CHILD.format(path=path, me=me))["won"]
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(me,)) for me in ("p1", "p2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    assert won["p1"] + won["p2"] == 20  # every job claimed exactly once
    entries = run_sync(Board(SQLiteBoardStore(path), scope="run:shared").read())
    assert len([e for e in entries if e.kind == "finding"]) == 40
    assert [e.seq for e in entries] == list(range(1, len(entries) + 1))
    assert len({e.id for e in entries}) == len(entries)
