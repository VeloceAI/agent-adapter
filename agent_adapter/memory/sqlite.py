"""SQLiteMemory: memory records in one SQLite file, keyword search (FTS5) plus optional vectors."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from datetime import timedelta
from typing import Any, TypeVar

from agent_adapter.core.errors import (
    AdapterError,
    ConfigError,
    ModelError,
    ModelOutputInvalid,
    ToolUsageError,
    log_failure,
)
from agent_adapter.embeddings.port import EmbeddingPort
from agent_adapter.memory import _sqlite_sql as sql
from agent_adapter.memory._base import GuardedMemory
from agent_adapter.memory.records import (
    FilterValue,
    MemoryHit,
    MemoryRecord,
    MemoryScope,
    utc_now,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
MAX_SCAN = 20_000  # newest candidate vectors compared per search
RRF_K = 60  # reciprocal-rank fusion constant
_RESERVED = (":memory:", "")
_COLUMNS = "m.seq, m.record"


class SQLiteMemory(GuardedMemory):
    """Memory in one SQLite file: keyword search, optional vectors, deletes that leave no text.

    Raises:
        ConfigError: a bad `path`, `timeout`, `retention_days` or `embedder` (here); on
            first use, a file without WAL, another schema or foreign tables with our names.
    """

    backend = "sqlite"

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        embedder: EmbeddingPort | None = None,
        retention_days: float | None = None,
        timeout: float = 5.0,
        capture_content: bool = False,
    ) -> None:
        super().__init__(retention_days=retention_days, capture_content=capture_content)
        if not isinstance(path, str | os.PathLike):
            raise ConfigError(f"path must be a str or os.PathLike, got {type(path).__name__}")
        self.path = os.fspath(path)
        if self.path in _RESERVED or self.path.startswith("file:"):
            raise ConfigError("SQLiteMemory needs a file path (not :memory:, '' or a URI)")
        if embedder is not None and not isinstance(embedder, EmbeddingPort):
            raise ConfigError(f"embedder must be an EmbeddingPort, got {type(embedder).__name__}")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigError(f"timeout must be a finite number > 0, got {timeout!r}")
        self.embedder = embedder
        self.timeout = float(timeout)
        self._fts = False
        self._fts_secure = False
        self._ready = False
        self._ready_lock = threading.Lock()

    async def purge(self) -> int:
        """Delete expired records and those past `retention_days` now; how many."""
        return await self._guard("purge", self._run(self._purge_now))

    # --- backend hooks ----------------------------------------------------------------------

    async def _add(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        vectors = await self._embed([record.text for record in records], strict=True)
        rows = [_row(record, vector) for record, vector in zip(records, vectors, strict=True)]
        await self._run(self._insert, rows)
        logger.debug("sqlite memory added %d records", len(records))
        return records

    async def _search(
        self, query: str, scope: MemoryScope, k: int, filters: dict[str, FilterValue]
    ) -> tuple[list[MemoryHit], str]:
        words = sql.tokens(query)
        if not words and not query.strip():
            return await self._run(self._recent, scope, k, filters), "recent"
        ranked: dict[str, list[MemoryHit]] = {}
        if words:
            ranked["keyword"] = await self._run(self._keyword, words, scope, k * 4, filters)
        vector = (await self._embed([query], strict=False))[0] if query.strip() else None
        if vector is not None:
            ranked["vector"] = await self._run(self._nearest, vector, scope, k * 4, filters)
        found = {mode: hits for mode, hits in ranked.items() if hits}
        if len(found) > 1:
            return _fuse(list(found.values()))[:k], "hybrid"
        mode = next(iter(found or ranked), "keyword")
        return found.get(mode, [])[:k], mode

    async def _get(self, id: str, scope: MemoryScope | None) -> MemoryRecord | None:
        return await self._run(self._get_row, id, scope)

    async def _delete(self, ids: list[str] | None, scope: MemoryScope | None) -> int:
        return await self._run(self._delete_rows, ids, scope)

    async def _export(self, scope: MemoryScope) -> list[MemoryRecord]:
        return await self._run(self._export_rows, scope)

    # --- embedding --------------------------------------------------------------------------

    async def _embed(self, texts: list[str], *, strict: bool) -> list[tuple[float, ...] | None]:
        """One vector per text, or Nones without an embedder (or, unless `strict`, on failure).

        Raises:
            ModelError: the embedder failed and `strict` (an add).
        """
        if self.embedder is None:
            return [None] * len(texts)
        try:
            result = await self.embedder.embed(texts)  # bills the caller's current context
            if len(result.vectors) != len(texts):
                raise ModelOutputInvalid(
                    f"embedder returned {len(result.vectors)} vectors for {len(texts)} texts",
                    provider=self.embedder.provider,
                    model=self.embedder.model,
                )
        except Exception as exc:
            if strict and isinstance(exc, AdapterError):
                raise
            if strict:
                raise ModelError(
                    "memory embedder failed",
                    provider=self.embedder.provider,
                    model=self.embedder.model,
                ) from exc
            log_failure(logger, logging.WARNING, exc, "memory search runs keyword-only")
            return [None]
        return list(result.vectors)

    # --- worker-thread side -----------------------------------------------------------------

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.to_thread(self._call, fn, *args)

    def _call(self, fn: Callable[..., T], *args: Any) -> T:
        self._ensure_schema()
        with closing(self._connect()) as conn:
            return fn(conn, *args)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA secure_delete=ON")  # deleted rows are zeroed, not just freed
        except BaseException:
            conn.close()
            raise
        return conn

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._ready_lock:
            if not self._ready:
                sql.create_private(self.path)
                with closing(self._connect()) as conn:
                    self._fts, self._fts_secure = sql.migrate(conn)
                self._ready = True

    def _live(self) -> tuple[list[str], list[object]]:
        now = utc_now()
        cutoff = None
        if self.retention_days is not None:
            cutoff = (now - timedelta(days=self.retention_days)).timestamp()
        return sql.live_where(now.timestamp(), cutoff)

    def _where(
        self, scope: MemoryScope | None, filters: dict[str, FilterValue] | None = None
    ) -> tuple[str, list[object]]:
        where, params = self._live()
        if scope is not None:
            more, extra = sql.scope_where(scope)
            where, params = where + more, params + extra
        if filters:
            more, extra = sql.filter_where(filters)
            where, params = where + more, params + extra
        return " AND ".join(where), params

    def _insert(self, conn: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
        with sql.immediate(conn):
            purged = self._purge(conn)
            ids = [row[0] for row in rows]
            marks = ", ".join("?" for _ in ids)
            taken = conn.execute(f"SELECT 1 FROM {sql.TABLE} WHERE id IN ({marks})", ids)
            if taken.fetchone() is not None:
                raise ToolUsageError("a memory record with one of these ids is already stored")
            for row in rows:
                cursor = conn.execute(
                    f"INSERT INTO {sql.TABLE} (id, user_id, session_id, agent_id, created_at,"
                    " expires_at, text, metadata, record, vector, dim)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
                if self._fts:
                    conn.execute(
                        f"INSERT INTO {sql.FTS} (rowid, text) VALUES (?, ?)",
                        (cursor.lastrowid, row[6]),
                    )
        self._scrub(conn, purged)

    def _purge(self, conn: sqlite3.Connection) -> int:
        # Caller holds BEGIN IMMEDIATE.
        now = utc_now()
        cutoff = None
        if self.retention_days is not None:
            cutoff = (now - timedelta(days=self.retention_days)).timestamp()
        seqs = sql.expired_seqs(conn, now.timestamp(), cutoff)
        self._drop(conn, seqs)
        if seqs:
            logger.info("sqlite memory purged %d expired records", len(seqs))
        return len(seqs)

    def _purge_now(self, conn: sqlite3.Connection) -> int:
        with sql.immediate(conn):
            count = self._purge(conn)
        self._scrub(conn, count)
        return count

    def _scrub(self, conn: sqlite3.Connection, removed: int) -> None:
        if removed:
            sql.scrub(conn, self._fts, self._fts_secure)

    def _drop(self, conn: sqlite3.Connection, seqs: list[int]) -> None:
        pairs = [(seq,) for seq in seqs]
        conn.executemany(f"DELETE FROM {sql.TABLE} WHERE seq = ?", pairs)
        if self._fts:
            conn.executemany(f"DELETE FROM {sql.FTS} WHERE rowid = ?", pairs)

    def _recent(
        self, conn: sqlite3.Connection, scope: MemoryScope, k: int, filters: dict[str, Any]
    ) -> list[MemoryHit]:
        clause, params = self._where(scope, filters)
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM {sql.TABLE} m WHERE {clause}"
            " ORDER BY m.created_at DESC, m.seq DESC LIMIT ?",
            (*params, k),
        ).fetchall()
        return [MemoryHit(record=_record(row[1])) for row in rows]

    def _keyword(
        self,
        conn: sqlite3.Connection,
        words: list[str],
        scope: MemoryScope,
        limit: int,
        filters: dict[str, Any],
    ) -> list[MemoryHit]:
        clause, params = self._where(scope, filters)
        if self._fts:
            rows = conn.execute(
                f"SELECT {_COLUMNS}, bm25({sql.FTS}) FROM {sql.FTS}"
                f" JOIN {sql.TABLE} m ON m.seq = {sql.FTS}.rowid"
                f" WHERE {sql.FTS} MATCH ? AND {clause} ORDER BY bm25({sql.FTS}) LIMIT ?",
                (sql.match_expression(words), *params, limit),
            ).fetchall()
            return [MemoryHit(record=_record(row[1]), score=-float(row[2])) for row in rows]
        likes = " OR ".join("lower(m.text) LIKE ? ESCAPE '\\'" for _ in words)
        rows = conn.execute(
            f"SELECT {_COLUMNS}, lower(m.text) FROM {sql.TABLE} m WHERE ({likes}) AND {clause}"
            " ORDER BY m.created_at DESC LIMIT ?",
            (*[sql.like_pattern(word) for word in words], *params, MAX_SCAN),
        ).fetchall()
        scored = [(sum(word in row[2] for word in words), row) for row in rows]
        scored.sort(key=lambda item: -item[0])  # stable: newest first among equals
        return [MemoryHit(record=_record(row[1]), score=float(n)) for n, row in scored[:limit]]

    def _nearest(
        self,
        conn: sqlite3.Connection,
        vector: tuple[float, ...],
        scope: MemoryScope,
        limit: int,
        filters: dict[str, Any],
    ) -> list[MemoryHit]:
        clause, params = self._where(scope, filters)
        rows = conn.execute(
            f"SELECT {_COLUMNS}, m.vector FROM {sql.TABLE} m"
            f" WHERE m.vector IS NOT NULL AND m.dim = ? AND {clause}"
            " ORDER BY m.created_at DESC LIMIT ?",
            (len(vector), *params, MAX_SCAN),
        ).fetchall()
        if len(rows) == MAX_SCAN:
            logger.warning("memory vector search compared only the newest %d records", MAX_SCAN)
        norm = math.sqrt(sum(x * x for x in vector))
        scored = [(sql.cosine(vector, norm, sql.unpack(row[2])), row[1]) for row in rows]
        scored.sort(key=lambda item: -item[0])
        return [MemoryHit(record=_record(raw), score=score) for score, raw in scored[:limit]]

    def _get_row(
        self, conn: sqlite3.Connection, id: str, scope: MemoryScope | None
    ) -> MemoryRecord | None:
        clause, params = self._where(scope)
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM {sql.TABLE} m WHERE m.id = ? AND {clause}", (id, *params)
        ).fetchone()
        return None if row is None else _record(row[1])

    def _delete_rows(
        self, conn: sqlite3.Connection, ids: list[str] | None, scope: MemoryScope | None
    ) -> int:
        where: list[str] = []
        params: list[object] = []
        if ids is not None:
            where.append(f"m.id IN ({', '.join('?' for _ in ids)})")
            params += ids
        if scope is not None:
            more, extra = sql.scope_where(scope)
            where, params = where + more, params + extra
        with sql.immediate(conn):
            purged = self._purge(conn)
            seqs = [
                row[0]
                for row in conn.execute(
                    f"SELECT m.seq FROM {sql.TABLE} m WHERE {' AND '.join(where)}", params
                )
            ]
            self._drop(conn, seqs)
        self._scrub(conn, purged + len(seqs))
        return len(seqs)

    def _export_rows(self, conn: sqlite3.Connection, scope: MemoryScope) -> list[MemoryRecord]:
        with sql.immediate(conn):
            purged = self._purge(conn)
        self._scrub(conn, purged)
        clause, params = self._where(scope)
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM {sql.TABLE} m WHERE {clause} ORDER BY m.created_at, m.seq",
            params,
        ).fetchall()
        return [_record(row[1]) for row in rows]


def _row(record: MemoryRecord, vector: tuple[float, ...] | None) -> tuple[Any, ...]:
    created = record.created_at or utc_now()
    expires = record.expires_at.timestamp() if record.expires_at is not None else None
    return (
        record.id,
        record.scope.user_id,
        record.scope.session_id,
        record.scope.agent_id,
        created.timestamp(),
        expires,
        record.text,
        json.dumps(record.metadata, allow_nan=False),
        record.model_dump_json(),
        None if vector is None else sql.pack(vector),
        None if vector is None else len(vector),
    )


def _record(raw: str) -> MemoryRecord:
    return MemoryRecord.model_validate_json(raw)


def _fuse(ranked: list[list[MemoryHit]]) -> list[MemoryHit]:
    """Reciprocal-rank fusion of several rankings of the same records."""
    scores: dict[str, float] = {}
    records: dict[str, MemoryRecord] = {}
    for hits in ranked:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.record.id] = scores.get(hit.record.id, 0.0) + 1.0 / (RRF_K + rank)
            records[hit.record.id] = hit.record
    order = sorted(scores, key=lambda id: -scores[id])
    return [MemoryHit(record=records[id], score=round(scores[id], 6)) for id in order]
