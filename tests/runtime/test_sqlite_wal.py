"""enable_wal retries a refused journal-mode switch within the busy timeout, and only that."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.runtime._sqlite_wal import enable_wal


class FakeConn:
    def __init__(self, timeout_ms: int, failures: list[str]) -> None:
        self.timeout_ms = timeout_ms
        self.failures = failures
        self.switches = 0

    def execute(self, sql: str) -> Any:
        if sql == "PRAGMA busy_timeout":
            return Row(self.timeout_ms)
        self.switches += 1
        if self.failures:
            raise sqlite3.OperationalError(self.failures.pop(0))
        return Row("WAL")


class Row:
    def __init__(self, value: object) -> None:
        self.value = value

    def fetchone(self) -> tuple[object]:
        return (self.value,)


def test_a_locked_switch_is_retried_until_it_succeeds() -> None:
    conn = FakeConn(5_000, ["database is locked", "database is locked"])
    assert enable_wal(conn) == "wal"  # type: ignore[arg-type]
    assert conn.switches == 3


def test_it_gives_up_once_the_busy_timeout_has_passed() -> None:
    conn = FakeConn(50, ["database is locked"] * 1000)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        enable_wal(conn)  # type: ignore[arg-type]
    assert 1 < conn.switches < 1000


def test_other_errors_are_not_retried() -> None:
    conn = FakeConn(5_000, ["disk I/O error"])
    with pytest.raises(sqlite3.OperationalError, match="disk"):
        enable_wal(conn)  # type: ignore[arg-type]
    assert conn.switches == 1


def test_a_real_file_reports_wal(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "w.db", timeout=1)) as conn:
        assert enable_wal(conn) == "wal"
