"""The retriever port: a query in, the best matching documents out."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_adapter.core.context import ToolContext
from agent_adapter.retrieval.documents import Document, Filters


@runtime_checkable
class RetrieverPort(Protocol):
    """Finds up to `k` documents for `query` that match `filters`, best first."""

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filters: Filters | None = None,
        ctx: ToolContext | None = None,
    ) -> list[Document]: ...
