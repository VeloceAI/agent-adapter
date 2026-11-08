"""The memory port: what every memory backend offers, async first."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agent_adapter.core.context import ToolContext
from agent_adapter.memory.records import Filters, MemoryHit, MemoryRecord, MemoryScope


@runtime_checkable
class MemoryPort(Protocol):
    """Stores and finds texts per scope; every backend here also has `*_sync` twins."""

    backend: str
    user_only: bool  # True: only MemoryScope(user_id=...) and no metadata (GuardedMemory: False)

    async def add(
        self, records: Sequence[MemoryRecord], *, ctx: ToolContext | None = None
    ) -> list[MemoryRecord]: ...

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        k: int = 5,
        filters: Filters | None = None,
        ctx: ToolContext | None = None,
    ) -> list[MemoryHit]: ...

    async def get(
        self, id: str, *, scope: MemoryScope | None = None, ctx: ToolContext | None = None
    ) -> MemoryRecord | None: ...

    async def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        scope: MemoryScope | None = None,
        ctx: ToolContext | None = None,
    ) -> int: ...

    async def export(
        self, scope: MemoryScope, *, ctx: ToolContext | None = None
    ) -> list[MemoryRecord]: ...
