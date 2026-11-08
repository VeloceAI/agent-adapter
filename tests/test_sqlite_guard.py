"""The suite-wide guard that fails a test which drops an open sqlite3 connection."""

import sqlite3
from contextlib import closing

from tests import sqlite_guard


def _leak_in_a_cycle() -> None:
    conn = sqlite3.connect(":memory:")
    conn.self_ref = conn  # type: ignore[attr-defined]  # only the cycle collector frees it


def test_guard_reports_an_unclosed_connection_with_its_open_site():
    _leak_in_a_cycle()
    leaks = sqlite_guard.take_leaks()
    assert len(leaks) == 1
    assert "test_guard_reports_an_unclosed_connection" in leaks[0]
    assert "test_sqlite_guard.py" in leaks[0] and "_leak_in_a_cycle" in leaks[0]


def test_guard_ignores_closed_connections():
    with closing(sqlite3.connect(":memory:")) as conn, conn:
        conn.execute("SELECT 1")
    conn = sqlite3.connect(":memory:")
    conn.close()
    del conn
    assert sqlite_guard.take_leaks() == []


def test_guard_tracks_every_connect_path():
    assert sqlite3.connect is sqlite3.dbapi2.connect
    with closing(sqlite3.dbapi2.connect(":memory:")) as conn:
        assert isinstance(conn, sqlite_guard.TrackedConnection)


def test_guard_keeps_an_explicit_factory():
    class Plain(sqlite3.Connection):
        pass

    with closing(sqlite3.connect(":memory:", factory=Plain)) as conn:
        assert type(conn) is Plain
