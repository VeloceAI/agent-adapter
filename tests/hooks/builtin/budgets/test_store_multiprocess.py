"""Budget stores shared by several processes: racing reservations never overspend."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.hooks.builtin.budgets import SQLAlchemyBudgetStore, SQLiteBudgetStore
from agent_adapter.runtime import run_sync
from tests.work import run_child

TRIES = 40  # reservations each child attempts; the limit admits 30 in total

CHILD = """import asyncio, json
from agent_adapter.hooks.builtin.budgets import {cls}

async def main():
    store = {cls}({target!r}, **{options!r})
    won = 0
    for i in range({tries}):
        reserved = await store.reserve(
            "team|acme", "{me}-" + str(i), 0.1, limit_usd=3.0, hold_s=600, expire_at=None
        )
        won += reserved.accepted
    return won

print(json.dumps({{"won": asyncio.run(main())}}))
"""


def _race(cls: str, target: str, options: dict[str, Any]) -> list[int]:
    results: list[int] = [0, 0, 0]
    errors: list[BaseException] = []

    def one(i: int) -> None:
        script = CHILD.format(cls=cls, target=target, options=options, tries=TRIES, me=f"p{i}")
        try:
            results[i] = run_child(script)["won"]
        except BaseException as exc:  # re-raised in the test's thread
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def test_sqlite_processes_share_one_limit(tmp_path: Path):
    path = str(tmp_path / "budget.db")
    won = _race("SQLiteBudgetStore", path, {"timeout": 60})
    assert sum(won) == 30
    usage = run_sync(SQLiteBudgetStore(path).usage("team|acme"))
    assert usage.reserved_usd == pytest.approx(3.0)


def test_sqlalchemy_processes_share_one_limit(tmp_path: Path):
    pytest.importorskip("aiosqlite")
    url = f"sqlite+aiosqlite:///{(tmp_path / 'budget.db').as_posix()}?timeout=60"
    won = _race("SQLAlchemyBudgetStore", url, {})
    assert sum(won) == 30
    reader = SQLAlchemyBudgetStore(url)  # kept alive: run_sync disposes live stores' engines
    usage = run_sync(reader.usage("team|acme"))
    assert usage.reserved_usd == pytest.approx(3.0)
