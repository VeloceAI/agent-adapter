"""`LangChainRetriever`: a LangChain retriever (`ainvoke`/`invoke`) behind the `RetrieverPort`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.retrieval._base import GuardedRetriever
from agent_adapter.retrieval.callable import results
from agent_adapter.retrieval.documents import (
    Document,
    FilterValue,
    document_id,
    score_of,
    source_from,
)

# (k, filters) -> keyword arguments for `retriever.ainvoke(query, **kwargs)`.
SearchKwargs = Callable[[int, dict[str, FilterValue]], Mapping[str, Any]]


class LangChainRetriever(GuardedRetriever):
    """Calls `retriever.ainvoke(query, **search_kwargs(k, filters))` (or `invoke` in a thread).

    LangChain retrievers take no `k` or filters by default, so none are sent unless you give
    `search_kwargs` (a vector-store retriever accepts `{"k": k, "filter": filters}`). Results
    are cut to `k` and re-checked against `filters`: a mismatch is dropped, never returned.

    Raises:
        ToolUsageError: `retriever` has neither `ainvoke` nor `invoke`.
    """

    backend = "langchain"

    def __init__(
        self,
        retriever: Any,
        *,
        search_kwargs: SearchKwargs | None = None,
        name: str = "",
        timeout_s: float | None = None,
        capture_content: bool = False,
    ) -> None:
        if not (hasattr(retriever, "ainvoke") or hasattr(retriever, "invoke")):
            raise ToolUsageError("LangChainRetriever needs a retriever with ainvoke or invoke")
        super().__init__(name=name, timeout_s=timeout_s, capture_content=capture_content)
        self.retriever = retriever
        self.search_kwargs = search_kwargs

    async def _retrieve(self, query: str, k: int, filters: dict[str, FilterValue]) -> Sequence[Any]:
        kwargs = dict(self.search_kwargs(k, dict(filters))) if self.search_kwargs else {}
        if hasattr(self.retriever, "ainvoke"):
            raw = await self.retriever.ainvoke(query, **kwargs)
        else:
            raw = await asyncio.to_thread(self.retriever.invoke, query, **kwargs)
        return [langchain_document(item) for item in results(raw, self.name)]


def langchain_document(item: Any) -> Document:
    """A `Document` from a LangChain `Document` (`page_content`, `metadata`, optional `id`).

    Raises:
        ToolUsageError: the item has no str `page_content`.
    """
    text = getattr(item, "page_content", None)
    if not isinstance(text, str):
        raise ToolUsageError(f"a LangChain result has no page_content ({type(item).__name__})")
    metadata = dict(getattr(item, "metadata", None) or {})
    source = source_from(metadata)
    doc_id = getattr(item, "id", None) or metadata.get("id")
    return Document(
        id=str(doc_id) if doc_id else document_id(text, source.uri),
        text=text,
        score=score_of(metadata.get("score", metadata.get("relevance_score"))),
        source=source,
        metadata=metadata,
    )
