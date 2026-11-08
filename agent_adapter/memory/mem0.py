"""Mem0Memory: a Mem0 2.x client (`Memory` or `AsyncMemory`) as a `MemoryPort`."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.errors import IntegrationError, MissingDependency
from agent_adapter.memory import _clients as c
from agent_adapter.memory._base import GuardedMemory
from agent_adapter.memory.records import FilterValue, MemoryHit, MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)

_META = "aa"  # our fields inside Mem0 metadata: source, created_at, expires_at
_ENTITY = ("user_id", "agent_id", "run_id")
_STORED_EVENTS = {"ADD", "UPDATE"}  # infer=True may also report DELETE or NONE


class Mem0Memory(GuardedMemory):
    """Mem0 2.x behind the memory port: user, session (`run_id`) and agent scopes.

    Raises:
        MissingDependency: no client and `mem0ai` is not installed (on first use).
    """

    backend = "mem0"

    def __init__(
        self,
        client: Any = None,
        *,
        infer: bool = False,
        retention_days: float | None = None,
        capture_content: bool = False,
    ) -> None:
        super().__init__(retention_days=retention_days, capture_content=capture_content)
        self._client = client
        self.infer = bool(infer)

    @property
    def client(self) -> Any:
        """The Mem0 client, built on first use when none was given."""
        if self._client is None:
            try:
                from mem0 import Memory
            except ImportError as exc:
                raise MissingDependency(
                    'Mem0Memory needs mem0ai: pip install "agent-adapter[mem0]"'
                ) from exc
            self._client = Memory()
        return self._client

    async def _add(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        stored: list[MemoryRecord] = []
        for record in records:
            meta = {**record.metadata, _META: _ours(record)}
            found = await c.call(
                self.client.add,
                [{"role": "user", "content": record.text}],
                metadata=meta,
                infer=self.infer,
                **_ids(record.scope),
            )
            for item in c.results(found):
                memory_id = c.field(item, "id")
                event = c.field(item, "event") or "ADD"
                if isinstance(memory_id, str) and memory_id and event in _STORED_EVENTS:
                    text = c.field(item, "memory") or record.text
                    stored.append(record.model_copy(update={"id": memory_id, "text": text}))
        return stored

    async def _search(
        self, query: str, scope: MemoryScope, k: int, filters: dict[str, FilterValue]
    ) -> tuple[list[MemoryHit], str]:
        if not query.strip():
            recent = list(reversed(await self._export(scope)))  # newest first, ties too
            return [MemoryHit(record=r) for r in recent[:k]], "recent"
        # Our scope ids are set last, so a metadata filter can never replace them.
        wanted = {key: value for key, value in filters.items() if key not in _ENTITY}
        found = await c.call(self.client.search, query, top_k=k, filters={**wanted, **_ids(scope)})
        hits = []
        for item in c.results(found):
            record = _record(item)
            if record is not None:
                hits.append(MemoryHit(record=record, score=float(c.field(item, "score") or 0.0)))
        return hits, "backend"

    async def _get(self, id: str, scope: MemoryScope | None) -> MemoryRecord | None:
        return _record(await c.call(self.client.get, id))

    async def _delete(self, ids: list[str] | None, scope: MemoryScope | None) -> int:
        if ids is not None:
            return await self._delete_ids(ids, scope)
        if scope is None:
            return 0
        owned = await self._export(scope)
        if scope.user_id:
            # Mem0's delete_all matches every given id, as MemoryScope.covers does.
            await c.call(self.client.delete_all, **_ids(scope))
        else:
            for record in owned:  # without a user, Mem0 would match other users' records
                await c.call(self.client.delete, record.id)
        if await self._export(scope):
            raise IntegrationError("mem0 kept records after delete", integration=self.backend)
        return len(owned)

    async def _delete_ids(self, ids: list[str], scope: MemoryScope | None) -> int:
        targets = []
        for memory_id in ids:
            record = await self._get(memory_id, scope)
            if record is not None and (scope is None or scope.covers(record.scope)):
                targets.append(memory_id)
        for memory_id in targets:
            await c.call(self.client.delete, memory_id)
        return len(targets)

    async def _export(self, scope: MemoryScope) -> list[MemoryRecord]:
        found = await c.call(self.client.get_all, filters=_ids(scope), top_k=c.EXPORT_LIMIT)
        items = c.results(found)
        if len(items) >= c.EXPORT_LIMIT:
            logger.warning("mem0 export hit its limit of %d records", c.EXPORT_LIMIT)
        records = [record for record in map(_record, items) if record is not None]
        return [r for r in sorted(records, key=_created) if scope.covers(r.scope)]


def _ids(scope: MemoryScope) -> dict[str, str]:
    """Mem0's entity ids for `scope` (only the set ones)."""
    pairs = {"user_id": scope.user_id, "run_id": scope.session_id, "agent_id": scope.agent_id}
    return {key: value for key, value in pairs.items() if value}


def _created(record: MemoryRecord) -> float:
    return record.created_at.timestamp() if record.created_at else 0.0


def _ours(record: MemoryRecord) -> dict[str, str]:
    return {
        "source": record.source,
        "created_at": record.created_at.isoformat() if record.created_at else "",
        "expires_at": record.expires_at.isoformat() if record.expires_at else "",
    }


def _record(item: Any) -> MemoryRecord | None:
    """A Mem0 memory as a record, or None when it has no id, text or scope."""
    memory_id, text = c.field(item, "id"), c.field(item, "memory")
    if not (isinstance(memory_id, str) and memory_id and isinstance(text, str) and text):
        return None
    meta = dict(c.field(item, "metadata") or {})
    ours = meta.pop(_META, None)
    ours = ours if isinstance(ours, dict) else {}
    try:
        scope = MemoryScope(
            user_id=c.field(item, "user_id") or "",
            session_id=c.field(item, "run_id") or "",
            agent_id=c.field(item, "agent_id") or "",
        )
    except ValidationError:
        return None  # an unscoped memory belongs to no caller
    return MemoryRecord(
        id=memory_id,
        scope=scope,
        text=text,
        metadata=meta,
        created_at=c.timestamp(ours.get("created_at")) or c.timestamp(c.field(item, "created_at")),
        expires_at=c.timestamp(ours.get("expires_at")),
        source=str(ours.get("source") or ""),
    )
