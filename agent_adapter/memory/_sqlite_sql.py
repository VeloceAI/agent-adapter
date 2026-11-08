"""SQLiteMemory's schema, migration, SQL fragments and vector maths (stdlib only)."""

from __future__ import annotations

import array
import logging
import math
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from agent_adapter.core.errors import ConfigError
from agent_adapter.memory.records import FilterValue, MemoryScope

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TABLE = "aa_memory"
FTS = "aa_memory_fts"
META = "aa_memory_meta"
MAX_TOKENS = 32
_TOKEN = re.compile(r"\w+")
_TABLE_SQL = (
    f"CREATE TABLE {TABLE} ("
    " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
    " id TEXT NOT NULL UNIQUE,"
    " user_id TEXT NOT NULL,"
    " session_id TEXT NOT NULL,"
    " agent_id TEXT NOT NULL,"
    " created_at REAL NOT NULL,"
    " expires_at REAL,"
    " text TEXT NOT NULL,"
    " metadata TEXT NOT NULL,"
    " record TEXT NOT NULL,"
    " vector BLOB,"
    " dim INTEGER)",
    f"CREATE INDEX {TABLE}_scope ON {TABLE} (user_id, session_id, agent_id, created_at)",
    f"CREATE INDEX {TABLE}_expiry ON {TABLE} (expires_at) WHERE expires_at IS NOT NULL",
    f"CREATE INDEX {TABLE}_created ON {TABLE} (created_at)",
)
_FTS_SQL = f"CREATE VIRTUAL TABLE {FTS} USING fts5(text, tokenize='porter unicode61')"
_OWN_NAMES = (TABLE, FTS, *(f"{TABLE}_{s}" for s in ("scope", "expiry", "created")))


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[None]:
    """One write transaction that takes the database lock before it reads."""
    # TODO(dedupe): same as approval.sqlite_store._immediate; move both to runtime.
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def create_private(path: str) -> None:
    """Create the file owner-only (0600) unless it exists; Windows applies the folder's ACL."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    os.close(fd)


def fts5_available(conn: sqlite3.Connection) -> bool:
    """True when this SQLite build has the FTS5 extension."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.aa_fts_probe USING fts5(x)")
    except sqlite3.OperationalError:
        return False
    conn.execute("DROP TABLE temp.aa_fts_probe")
    return True


def secure_fts(conn: sqlite3.Connection) -> bool:
    """Turn on FTS5 `secure-delete` (SQLite 3.42+), so deleted tokens leave the index."""
    try:
        conn.execute(f"INSERT INTO {FTS} ({FTS}, rank) VALUES ('secure-delete', 1)")
    except sqlite3.OperationalError:
        return False
    return True


def scrub(conn: sqlite3.Connection, fts: bool, fts_secure: bool) -> None:
    """After deletes: merge the FTS index when needed, then empty the WAL into the file."""
    if fts and not fts_secure:
        conn.execute(f"INSERT INTO {FTS} ({FTS}) VALUES ('optimize')")
    busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if busy:
        logger.warning("memory WAL checkpoint was blocked by a reader; deleted text stays in it")


def migrate(conn: sqlite3.Connection) -> tuple[bool, bool]:
    """Switch to WAL and create (or check) the schema; whether FTS5 and its secure-delete are on.

    Raises:
        ConfigError: no WAL support, another schema version, or foreign tables with our names.
    """
    (mode,) = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if str(mode).lower() != "wal":
        raise ConfigError(f"SQLiteMemory needs WAL journal mode; the file reports {mode!r}")
    fts = fts5_available(conn)
    with immediate(conn):
        conn.execute(f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value)")
        meta = dict(conn.execute(f"SELECT key, value FROM {META}").fetchall())
        if "schema" in meta:
            if meta["schema"] != SCHEMA_VERSION:
                raise ConfigError(
                    f"memory database schema {meta['schema']!r} is not {SCHEMA_VERSION}; "
                    "use the agent_adapter version that wrote it, or a new file"
                )
            if meta.get("fts") == 1 and not fts:
                raise ConfigError("this memory database uses FTS5, which this SQLite lacks")
            uses_fts = meta.get("fts") == 1
            return uses_fts, uses_fts and secure_fts(conn)
        marks = ", ".join("?" for _ in _OWN_NAMES)
        clash = conn.execute(
            f"SELECT name FROM sqlite_master WHERE name IN ({marks})", _OWN_NAMES
        ).fetchone()
        if clash is not None:
            raise ConfigError(f"the database already has {clash[0]!r} without memory metadata")
        for statement in _TABLE_SQL:
            conn.execute(statement)
        secure = False
        if fts:
            conn.execute(_FTS_SQL)
            secure = secure_fts(conn)
        conn.executemany(
            f"INSERT INTO {META} (key, value) VALUES (?, ?)",
            [("schema", SCHEMA_VERSION), ("fts", int(fts))],
        )
    if not fts:
        logger.warning("SQLite has no FTS5: memory keyword search falls back to LIKE")
    logger.info("memory database created (schema %d)", SCHEMA_VERSION)
    return fts, secure


def tokens(query: str) -> list[str]:
    """The distinct lower-case words of `query`, at most `MAX_TOKENS`."""
    return list(dict.fromkeys(_TOKEN.findall(query.lower())))[:MAX_TOKENS]


def match_expression(words: Sequence[str]) -> str:
    """An FTS5 query matching any of `words` (each is \\w+ only, so quoting is safe)."""
    return " OR ".join(f'"{word}"' for word in words)


def like_pattern(word: str) -> str:
    """`%word%` with LIKE's wildcards escaped (`_` is a word character)."""
    return "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def scope_where(scope: MemoryScope) -> tuple[list[str], list[object]]:
    """SQL conditions selecting what `scope` covers (see `MemoryScope.covers`)."""
    where: list[str] = ["m.user_id = ?"]
    params: list[object] = [scope.user_id]
    if scope.session_id:
        where.append("m.session_id = ?")
        params.append(scope.session_id)
    if scope.agent_id:
        where.append("m.agent_id = ?")
        params.append(scope.agent_id)
    return where, params


def expired_seqs(conn: sqlite3.Connection, now: float, cutoff: float | None) -> list[int]:
    """Rows past `expires_at` or retention, found through the expiry and created indexes."""
    found = conn.execute(f"SELECT seq FROM {TABLE} WHERE expires_at <= ?", (now,))
    seqs = {row[0] for row in found}
    if cutoff is not None:
        found = conn.execute(f"SELECT seq FROM {TABLE} WHERE created_at <= ?", (cutoff,))
        seqs.update(row[0] for row in found)
    return sorted(seqs)


def live_where(now: float, cutoff: float | None) -> tuple[list[str], list[object]]:
    """SQL conditions selecting records that are neither expired nor past retention."""
    where: list[str] = ["(m.expires_at IS NULL OR m.expires_at > ?)"]
    params: list[object] = [now]
    if cutoff is not None:
        where.append("m.created_at > ?")
        params.append(cutoff)
    return where, params


def filter_where(filters: dict[str, FilterValue]) -> tuple[list[str], list[object]]:
    """SQL conditions for metadata equality; keys are checked, so the JSON path is safe."""
    where: list[str] = []
    params: list[object] = []
    for key, value in filters.items():
        path = f'$."{key}"'
        if isinstance(value, bool):
            where.append("json_type(m.metadata, ?) = ?")
            params += [path, "true" if value else "false"]
            continue
        kinds = "('text')" if isinstance(value, str) else "('integer', 'real')"
        where.append(f"json_type(m.metadata, ?) IN {kinds} AND json_extract(m.metadata, ?) = ?")
        params += [path, path, value]
    return where, params


def pack(vector: Sequence[float]) -> bytes:
    """A vector as float32 bytes."""
    return array.array("f", vector).tobytes()


def unpack(blob: bytes) -> array.array[float]:
    """float32 bytes as a vector."""
    values = array.array("f")
    values.frombytes(blob)
    return values


def cosine(query: Sequence[float], query_norm: float, vector: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for a zero vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if not norm or not query_norm:
        return 0.0
    return sum(a * b for a, b in zip(query, vector, strict=True)) / (query_norm * norm)
