"""Golden path: who asked for each parked call, so a reviewer never approves their own."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS requests "
    "(approval_id TEXT PRIMARY KEY, user TEXT NOT NULL, run_id TEXT NOT NULL)"
)


class Requesters:
    """Approval id -> (requester, run id), in one SQLite file every worker on the host shares."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def add(self, approval_id: str, user: str, run_id: str) -> None:
        """Remember who asked; the first record of an id wins."""
        await asyncio.to_thread(self._add, approval_id, user, run_id)

    async def get(self, approval_id: str) -> tuple[str, str] | None:
        """`(requester, run_id)` of a parked call, or None for an unknown id."""
        return await asyncio.to_thread(self._get, approval_id)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute(SCHEMA)
        return conn

    def _add(self, approval_id: str, user: str, run_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO requests VALUES (?, ?, ?)", (approval_id, user, run_id)
            )

    def _get(self, approval_id: str) -> tuple[str, str] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT user, run_id FROM requests WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return None if row is None else (str(row[0]), str(row[1]))
