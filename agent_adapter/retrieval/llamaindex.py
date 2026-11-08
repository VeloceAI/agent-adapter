"""`LlamaIndexRetriever`: a LlamaIndex index or retriever behind the `RetrieverPort`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import MissingDependency, ToolUsageError
from agent_adapter.retrieval._base import GuardedRetriever
from agent_adapter.retrieval.callable import results
from agent_adapter.retrieval.documents import (
    Document,
    FilterValue,
    document_id,
    score_of,
    source_from,
)

INSTALL = 'pip install "agent-adapter[llamaindex]"'

# filters -> the backend's native filter object (for `index.as_retriever(filters=...)`).
NativeFilters = Callable[[dict[str, FilterValue]], Any]


class LlamaIndexRetriever(GuardedRetriever):
    """Searches an index (`as_retriever(similarity_top_k=k, filters=...)`) or a retriever.

    An index gets `k` and the filters natively (`MetadataFilters` by default, or your
    `native_filters(filters)`). A bare retriever keeps its own top-k and filters; the
    results are still re-checked against `filters`, so a mismatch is dropped, never returned.

    Raises:
        ToolUsageError: the object has neither `as_retriever`, `aretrieve` nor `retrieve`.
    """

    backend = "llamaindex"

    def __init__(
        self,
        index_or_retriever: Any,
        *,
        native_filters: NativeFilters | None = None,
        name: str = "",
        timeout_s: float | None = None,
        capture_content: bool = False,
    ) -> None:
        target = index_or_retriever
        if not any(hasattr(target, attr) for attr in ("as_retriever", "aretrieve", "retrieve")):
            raise ToolUsageError(
                "LlamaIndexRetriever needs an index (as_retriever) or a retriever (retrieve)"
            )
        super().__init__(name=name, timeout_s=timeout_s, capture_content=capture_content)
        self.target = target
        self.native_filters = native_filters

    async def _retrieve(self, query: str, k: int, filters: dict[str, FilterValue]) -> Sequence[Any]:
        retriever = self.target
        if hasattr(retriever, "as_retriever"):
            options: dict[str, Any] = {"similarity_top_k": k}
            if filters:
                convert = self.native_filters or metadata_filters
                options["filters"] = convert(filters)
            retriever = retriever.as_retriever(**options)
        if hasattr(retriever, "aretrieve"):
            raw = await retriever.aretrieve(query)
        else:
            raw = await asyncio.to_thread(retriever.retrieve, query)
        return [node_document(item) for item in results(raw, self.name)]


def metadata_filters(filters: Mapping[str, FilterValue]) -> Any:
    """LlamaIndex `MetadataFilters`: every key must equal its value.

    Raises:
        MissingDependency: `llama-index-core` is not installed.
    """
    try:
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
    except ImportError as exc:
        raise MissingDependency(f"LlamaIndex filters need llama-index-core: {INSTALL}") from exc
    return MetadataFilters(
        filters=[MetadataFilter(key=key, value=value) for key, value in filters.items()]
    )


def node_document(item: Any) -> Document:
    """A `Document` from a LlamaIndex `NodeWithScore` (or a bare node).

    Raises:
        ToolUsageError: the item has no text.
    """
    node = getattr(item, "node", item)
    getter = getattr(node, "get_content", None)
    text = getter() if callable(getter) else getattr(node, "text", None)
    if not isinstance(text, str):
        raise ToolUsageError(f"a LlamaIndex result has no text ({type(item).__name__})")
    metadata = dict(getattr(node, "metadata", None) or {})
    source = source_from(metadata)
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
    return Document(
        id=str(node_id) if node_id else document_id(text, source.uri),
        text=text,
        score=score_of(getattr(item, "score", None)),
        source=source,
        metadata=metadata,
    )
