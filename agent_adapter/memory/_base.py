"""The front every memory backend shares: checks, scope and expiry rules, events, sync twins."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from typing import Any, TypeVar

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError, IntegrationError, ToolUsageError, log_failure
from agent_adapter.core.types import Event
from agent_adapter.memory.records import (
    Filters,
    FilterValue,
    MemoryHit,
    MemoryRecord,
    MemoryScope,
    check_filters,
    check_ids,
    check_query,
    check_records,
    check_retention,
    check_scope,
    matches,
    scope_kind,
    utc_now,
)
from agent_adapter.runtime._fork import ForkSafeLock
from agent_adapter.runtime.current import current_context, use_context
from agent_adapter.runtime.ids import new_id
from agent_adapter.runtime.loop_runner import LoopRunner
from agent_adapter.tracing.events import EventType, emit, make_event, require_recorded

logger = logging.getLogger(__name__)

T = TypeVar("T")

# TODO(w3a): share one process loop with exec's sync twins once it moves to runtime.
_runner: LoopRunner | None = None
MAX_CLOCK_SKEW = timedelta(minutes=5)  # a created_at later than now + this is refused
_runner_lock = ForkSafeLock()


def _blocking(work: Callable[[], Awaitable[T]], timeout: float | None) -> T:
    global _runner
    with _runner_lock:
        if _runner is None or _runner.closed:
            _runner = LoopRunner(name="agent_adapter-memory")
        runner = _runner
    return runner.run(work, timeout=timeout)


class GuardedMemory(ABC):
    """The base of every backend: checks inputs, drops out-of-scope or expired output."""

    backend = "memory"
    keeps_expiry = True  # False: the backend drops expires_at, so a record may not set one
    user_only = False  # True: scopes must be MemoryScope(user_id=...) and records carry no metadata

    def __init__(self, *, retention_days: float | None = None, capture_content: bool = False):
        self.retention_days = check_retention(retention_days)
        self.capture_content = bool(capture_content)

    # --- the port ---------------------------------------------------------------------------

    async def add(
        self, records: Sequence[MemoryRecord], *, ctx: ToolContext | None = None
    ) -> list[MemoryRecord]:
        """Store `records` (ids, `created_at` and retention `expires_at` filled in); the stored.

        Raises:
            ToolUsageError: bad records, or an id that is already stored.
            IntegrationError: the backend failed (the cause is chained).
        """
        given = check_records(records)
        latest = utc_now() + MAX_CLOCK_SKEW
        if any(record.created_at and record.created_at > latest for record in given):
            raise ToolUsageError("created_at is in the future; it would outlive retention")
        if not self.keeps_expiry and any(record.expires_at for record in given):
            raise ToolUsageError(
                f"{self.backend} memory cannot keep expires_at; use retention_days instead"
            )
        items = [self._filled(record) for record in given]
        if not items:
            return []
        await self._required(ctx)
        started = time.perf_counter()
        stored = await self._guard("add", self._add(items), ctx)
        content = {"arguments": {"texts": [r.text for r in stored]}}
        self._event(
            EventType.MEMORY_ADDED,
            ctx,
            started,
            len(stored),
            content,
            scope_kind=scope_kind(r.scope for r in stored),
        )
        return stored

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        k: int = 5,
        filters: Filters | None = None,
        ctx: ToolContext | None = None,
    ) -> list[MemoryHit]:
        """Up to `k` live records in `scope` that match `query` and `filters`, best first.

        An empty query returns the newest records.

        Raises:
            ToolUsageError: bad query, scope, k or filters.
            IntegrationError: the backend failed.
        """
        query, k = check_query(query, k)
        scope = check_scope(scope)
        clean = check_filters(filters)
        started = time.perf_counter()
        hits, mode = await self._guard("search", self._search(query, scope, k, clean), ctx)
        kept = self._own(scope, [hit.record for hit in hits], "search")
        ids = {record.id for record in kept if matches(record, clean)}
        result = [hit for hit in hits if hit.record.id in ids][:k]
        content = {
            "arguments": {"query": query},
            "result": [hit.record.text for hit in result],
        }
        kind = scope_kind([scope])
        self._event(
            EventType.MEMORY_SEARCHED,
            ctx,
            started,
            len(result),
            content,
            k=k,
            mode=mode,
            scope_kind=kind,
        )
        return result

    async def get(
        self, id: str, *, scope: MemoryScope | None = None, ctx: ToolContext | None = None
    ) -> MemoryRecord | None:
        """The live record `id`, or None; with `scope`, None unless the record is in it.

        Raises:
            ToolUsageError: bad id or scope.
            IntegrationError: the backend failed.
        """
        (record_id,) = check_ids([id])
        if scope is not None:
            check_scope(scope)
        record = await self._guard("get", self._get(record_id, scope), ctx)
        if record is None or record.id != record_id:
            return None
        if scope is not None and not self._own(scope, [record], "get"):
            return None
        return record if record.live(utc_now(), self.retention_days) else None

    async def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        scope: MemoryScope | None = None,
        ctx: ToolContext | None = None,
    ) -> int:
        """Delete `ids`, every record in `scope`, or `ids` only within `scope`; the count.

        Raises:
            ToolUsageError: neither `ids` nor `scope`, or a bad one.
            IntegrationError: the backend failed.
        """
        if ids is None and scope is None:
            raise ToolUsageError("delete needs ids=, scope= or both")
        items = None if ids is None else check_ids(ids)
        if scope is not None:
            check_scope(scope)
        if items == []:
            return 0
        await self._required(ctx)
        started = time.perf_counter()
        count = await self._guard("delete", self._delete(items, scope), ctx)
        self._event(
            EventType.MEMORY_DELETED, ctx, started, count, {}, scope_kind=scope_kind([scope])
        )
        logger.info("%s memory deleted %d records", self.backend, count)
        return count

    async def export(
        self, scope: MemoryScope, *, ctx: ToolContext | None = None
    ) -> list[MemoryRecord]:
        """Every live record in `scope`, oldest first (for a privacy request).

        Raises:
            ToolUsageError: bad scope.
            IntegrationError: the backend failed.
        """
        scope = check_scope(scope)
        await self._required(ctx)
        started = time.perf_counter()
        records = self._own(scope, await self._guard("export", self._export(scope), ctx), "export")
        self._event(
            EventType.MEMORY_EXPORTED,
            ctx,
            started,
            len(records),
            {},
            scope_kind=scope_kind([scope]),
        )
        logger.info("%s memory exported %d records", self.backend, len(records))
        return records

    async def purge(self) -> int:
        """Delete expired records now; how many. Backends without storage of their own: 0."""
        return 0

    # --- sync twins -------------------------------------------------------------------------

    def add_sync(
        self, records: Sequence[MemoryRecord], *, timeout: float | None = None, **kw: Any
    ) -> list[MemoryRecord]:
        """`add` from sync code."""
        return _blocking(lambda: self.add(records, **kw), timeout)

    def search_sync(
        self, query: str, *, timeout: float | None = None, **kw: Any
    ) -> list[MemoryHit]:
        """`search` from sync code."""
        return _blocking(lambda: self.search(query, **kw), timeout)

    def get_sync(self, id: str, *, timeout: float | None = None, **kw: Any) -> MemoryRecord | None:
        """`get` from sync code."""
        return _blocking(lambda: self.get(id, **kw), timeout)

    def delete_sync(
        self, ids: Sequence[str] | None = None, *, timeout: float | None = None, **kw: Any
    ) -> int:
        """`delete` from sync code."""
        return _blocking(lambda: self.delete(ids, **kw), timeout)

    def export_sync(
        self, scope: MemoryScope, *, timeout: float | None = None, **kw: Any
    ) -> list[MemoryRecord]:
        """`export` from sync code."""
        return _blocking(lambda: self.export(scope, **kw), timeout)

    def purge_sync(self, *, timeout: float | None = None) -> int:
        """`purge` from sync code."""
        return _blocking(self.purge, timeout)

    # --- backend hooks ----------------------------------------------------------------------

    @abstractmethod
    async def _add(self, records: list[MemoryRecord]) -> list[MemoryRecord]: ...

    @abstractmethod
    async def _search(
        self, query: str, scope: MemoryScope, k: int, filters: dict[str, FilterValue]
    ) -> tuple[list[MemoryHit], str]: ...

    @abstractmethod
    async def _get(self, id: str, scope: MemoryScope | None) -> MemoryRecord | None: ...

    @abstractmethod
    async def _delete(self, ids: list[str] | None, scope: MemoryScope | None) -> int: ...

    @abstractmethod
    async def _export(self, scope: MemoryScope) -> list[MemoryRecord]: ...

    # --- helpers ----------------------------------------------------------------------------

    def _filled(self, record: MemoryRecord) -> MemoryRecord:
        created = record.created_at or utc_now()
        expires = record.expires_at
        if expires is None and self.retention_days is not None and self.keeps_expiry:
            expires = created + timedelta(days=self.retention_days)
        update = {"id": record.id or new_id("mem_"), "created_at": created, "expires_at": expires}
        return record.model_copy(update=update)

    def _own(self, scope: MemoryScope, records: list[MemoryRecord], op: str) -> list[MemoryRecord]:
        """The live records `scope` covers; any other is dropped, loudly if out of scope."""
        inside = [record for record in records if scope.covers(record.scope)]
        if len(inside) != len(records):
            logger.warning(
                "%s memory %s returned %d records outside the scope; dropped them",
                self.backend,
                op,
                len(records) - len(inside),
            )
        now = utc_now()
        return [record for record in inside if record.live(now, self.retention_days)]

    async def _guard(self, op: str, work: Awaitable[T], ctx: ToolContext | None = None) -> T:
        try:
            if ctx is None:
                return await work
            with use_context(ctx):  # the embedder bills and traces the caller's run
                return await work
        except AdapterError:
            raise
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "%s memory %s failed", self.backend, op)
            raise IntegrationError(
                f"{self.backend} memory {op} failed", integration=self.backend
            ) from exc

    async def _required(self, ctx: ToolContext | None) -> None:
        """Stop a change or an export when a fail-closed audit log lost a record."""
        ctx = ctx if ctx is not None else current_context()
        await require_recorded(ctx.sink if ctx is not None else None)

    def _event(
        self,
        type: str,
        ctx: ToolContext | None,
        started: float,
        count: int,
        content: dict[str, Any],
        **data: Any,
    ) -> None:
        ctx = ctx if ctx is not None else current_context()
        if ctx is None or ctx.sink is None:
            return
        latency = round((time.perf_counter() - started) * 1000, 3)
        extra = content if self.capture_content else {}
        event: Event = make_event(
            type, ctx, backend=self.backend, count=count, latency_ms=latency, **data, **extra
        )
        emit(ctx.sink, event)
