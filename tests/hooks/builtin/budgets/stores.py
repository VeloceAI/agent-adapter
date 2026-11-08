"""Every BudgetStore backend behind one factory, for the conformance tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.hooks.builtin.budgets import (
    BudgetStore,
    MemoryBudgetStore,
    RedisBudgetStore,
    SQLAlchemyBudgetStore,
    SQLiteBudgetStore,
)

BACKENDS = ("memory", "sqlite", "redis", "sqlalchemy")
CLOCKED = ("memory", "sqlite", "sqlalchemy")  # redis holds expire on the Redis clock


class Clock:
    """A settable clock (epoch seconds)."""

    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_store(kind: str, tmp_path: Path, clock: Callable[[], float]) -> BudgetStore:
    """A fresh store of `kind`; redis runs its Lua under fakeredis (skipped when absent)."""
    if kind == "memory":
        return MemoryBudgetStore(clock=clock)
    if kind == "sqlite":
        return SQLiteBudgetStore(tmp_path / "budget.db", clock=clock)
    if kind == "redis":
        fakeredis = pytest.importorskip("fakeredis")
        pytest.importorskip("lupa")
        server = fakeredis.FakeServer()
        return RedisBudgetStore(lambda: fakeredis.FakeAsyncRedis(server=server), prefix="t:")
    pytest.importorskip("aiosqlite")
    url = f"sqlite+aiosqlite:///{(tmp_path / 'budget_sa.db').as_posix()}"
    return SQLAlchemyBudgetStore(url, clock=clock)


def fake_redis(**options: Any) -> Any:
    """A fakeredis async client (skipped when fakeredis or lupa is absent)."""
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    return fakeredis.FakeAsyncRedis(**options)
