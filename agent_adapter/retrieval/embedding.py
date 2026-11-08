"""`EmbeddingRetriever`: an in-memory cosine search over any `EmbeddingPort`, for small corpora."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Sequence
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.embeddings.port import EmbeddingPort
from agent_adapter.exec._sugar import blocking
from agent_adapter.retrieval._base import GuardedRetriever
from agent_adapter.retrieval.documents import Document, FilterValue, as_document, matches

MAX_DOCUMENTS = 50_000  # beyond this, use a vector database


class EmbeddingRetriever(GuardedRetriever):
    """In-memory cosine search; with `scope_key`, documents and searches need a non-empty scope.

    Args:
        scope_key: Metadata key every document must carry and every search must filter on.
    """

    backend = "embedding"

    def __init__(
        self,
        embedder: EmbeddingPort,
        documents: Iterable[Document | str] = (),
        *,
        scope_key: str | None = None,
        name: str = "",
        timeout_s: float | None = None,
        capture_content: bool = False,
    ) -> None:
        if not isinstance(embedder, EmbeddingPort):
            raise ToolUsageError("EmbeddingRetriever needs an EmbeddingPort (embed(texts, ctx=))")
        if scope_key is not None and (not isinstance(scope_key, str) or not scope_key):
            raise ConfigError("scope_key must be a non-empty str or None")
        super().__init__(name=name, timeout_s=timeout_s, capture_content=capture_content)
        self.embedder = embedder
        self.scope_key = scope_key
        self._pending: list[Document] = []
        self._documents: list[Document] = []
        self._vectors: list[tuple[float, ...]] = []
        self._ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._queue([as_document(item) for item in documents])

    def __len__(self) -> int:
        return len(self._documents) + len(self._pending)

    async def add(
        self, documents: Iterable[Document | str], *, ctx: ToolContext | None = None
    ) -> int:
        """Embed `documents` (and any still queued) now; how many were added.

        Raises:
            ToolUsageError: a bad document, a repeated id, a missing scope or too many documents.
            ModelError: the embedder failed.
        """
        self._queue([as_document(item) for item in documents])
        async with self._lock:
            return await self._embed_pending(ctx)

    def add_sync(
        self, documents: Iterable[Document | str], *, timeout: float | None = None, **kw: Any
    ) -> int:
        """`add` from sync code."""
        return blocking(lambda: self.add(documents, **kw), timeout)

    async def _retrieve(self, query: str, k: int, filters: dict[str, FilterValue]) -> Sequence[Any]:
        if self.scope_key is not None and not _scoped(filters.get(self.scope_key)):
            raise ToolUsageError(
                f"{self.name}: every search needs a non-empty str filters[{self.scope_key!r}]"
            )
        async with self._lock:
            await self._embed_pending(None)
            if not self._documents:
                return []
            (query_vector,) = (await self.embedder.embed([query])).vectors
            unit = _unit(query_vector)
            if len(unit) != len(self._vectors[0]):
                raise ConfigError(
                    f"{self.name}: the query vector has {len(unit)} dimensions, the documents "
                    f"{len(self._vectors[0])}; rebuild the retriever after changing embedder"
                )
            ranked = [
                (sum(a * b for a, b in zip(unit, vector, strict=True)), index)
                for index, (doc, vector) in enumerate(
                    zip(self._documents, self._vectors, strict=True)
                )
                if matches(doc, filters)
            ]
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            self._documents[index].model_copy(update={"score": round(score, 6)})
            for score, index in ranked[:k]
        ]

    def _queue(self, documents: list[Document]) -> None:
        """Check `documents` and queue them for embedding.

        Raises:
            ToolUsageError: a repeated id, a missing scope, or more than `MAX_DOCUMENTS`.
        """
        ids = [doc.id for doc in documents]
        queued = {doc.id for doc in self._pending}
        if len(set(ids)) != len(ids) or self._ids.intersection(ids) or queued.intersection(ids):
            raise ToolUsageError(f"{self.name}: document ids must be unique")
        if len(self) + len(documents) > MAX_DOCUMENTS:
            raise ToolUsageError(f"{self.name}: at most {MAX_DOCUMENTS} documents")
        if self.scope_key is not None:
            for doc in documents:
                if not _scoped(doc.metadata.get(self.scope_key)):
                    raise ToolUsageError(
                        f"{self.name}: document {doc.id!r} needs a non-empty str "
                        f"metadata[{self.scope_key!r}]"
                    )
        self._pending.extend(documents)

    async def _embed_pending(self, ctx: ToolContext | None) -> int:
        """Embed the queued documents (the caller holds the lock); how many were added."""
        pending, self._pending = self._pending, []
        if not pending:
            return 0
        try:
            result = await self.embedder.embed([doc.text for doc in pending], ctx=ctx)
        except BaseException:
            self._pending = pending + self._pending  # keep them queued for the next try
            raise
        vectors = [_unit(vector) for vector in result.vectors]
        problem = ""
        if len(vectors) != len(pending):
            problem = f"the embedder returned {len(vectors)} vectors for {len(pending)} texts"
        elif len({len(vector) for vector in [*vectors, *self._vectors[:1]]}) > 1:
            problem = "the embedder returned vectors of different sizes"
        if problem:
            self._pending = pending + self._pending
            raise ConfigError(f"{self.name}: {problem}")
        self._documents.extend(pending)
        self._vectors.extend(vectors)
        self._ids.update(doc.id for doc in pending)
        return len(pending)


def _scoped(value: object) -> bool:
    """True for a usable scope value: a str that is not blank."""
    return isinstance(value, str) and bool(value.strip())


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    """`vector` scaled to length 1 (a zero vector stays zero, so it scores 0)."""
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(vector) if norm == 0 else tuple(value / norm for value in vector)
