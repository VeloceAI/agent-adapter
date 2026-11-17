"""Switch a SQLite file to WAL, retrying within the connection's busy timeout."""

from __future__ import annotations

import sqlite3
import time

RETRY_S = 0.02


def enable_wal(conn: sqlite3.Connection) -> str:
    """Ask for WAL and return the journal mode the file reports, lower-cased.

    Changing the journal mode needs the file's exclusive lock, and SQLite can refuse it at once
    ("database is locked") instead of waiting out the busy timeout when several processes open
    the same new file together. So the switch is retried until that timeout has passed.
    """
    (timeout_ms,) = conn.execute("PRAGMA busy_timeout").fetchone()
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            return str(mode).lower()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(RETRY_S)
