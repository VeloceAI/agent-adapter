"""Scorers: how similar or relevant texts are to a query, from 0 to 1."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, runtime_checkable

from agent_adapter.context.rerank import RerankerPort
from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import ConfigError, ModelOutputInvalid
from agent_adapter.embeddings.port import EmbeddingPort

ScoreFn = Callable[[str, list[str]], Sequence[float] | Awaitable[Sequence[float]]]
MEMO_SIZE = 4096  # vectors an EmbeddingScorer keeps, so repeated texts are embedded once


@runtime_checkable
class Scorer(Protocol):
    """Scores each text against `query`; one float in 0..1 per text, in order."""

    async def score(self, query: str, texts: list[str]) -> list[float]: ...


class EmbeddingScorer:
    """Cosine similarity of your embedder's vectors; negative similarity counts as 0.

    Raises:
        ConfigError: `embedder` is not an `EmbeddingPort`.
    """

    def __init__(self, embedder: EmbeddingPort) -> None:
        if not isinstance(embedder, EmbeddingPort):
            raise ConfigError("EmbeddingScorer needs an EmbeddingPort")
        self.embedder = embedder
        self._memo: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    async def score(self, query: str, texts: list[str]) -> list[float]:
        """The cosine similarity of each text to `query`, clamped to 0..1.

        Raises:
            ModelOutputInvalid: the embedder returned vectors of different sizes.
            ModelError: the embedder failed.
        """
        if not texts:
            return []
        vectors = await self._vectors([query, *texts])
        anchor = vectors[query]
        return [_cosine(anchor, vectors[text], self.embedder) for text in texts]

    async def _vectors(self, texts: list[str]) -> dict[str, tuple[float, ...]]:
        found = {text: self._memo[text] for text in texts if text in self._memo}
        missing = list(dict.fromkeys(text for text in texts if text not in found))
        if missing:
            result = await self.embedder.embed(missing)
            if len(result.vectors) != len(missing):
                raise _invalid(self.embedder, "the embedder returned the wrong number of vectors")
            for text, vector in zip(missing, result.vectors, strict=True):
                found[text] = tuple(vector)
                self._memo[text] = tuple(vector)
        for text in found:
            self._memo.move_to_end(text)
        while len(self._memo) > MEMO_SIZE:
            self._memo.popitem(last=False)
        return found


class RerankerScorer:
    """Your reranker's scores, which must be in 0..1 (apply a sigmoid to raw logits first).

    Raises:
        ConfigError: `reranker` is not a `RerankerPort`.
    """

    def __init__(self, reranker: RerankerPort) -> None:
        if not isinstance(reranker, RerankerPort):
            raise ConfigError("RerankerScorer needs a RerankerPort, e.g. CallableReranker(fn)")
        self.reranker = reranker

    async def score(self, query: str, texts: list[str]) -> list[float]:
        """The reranker's score of each text for `query`.

        Raises:
            ModelError: the reranker failed.
        """
        if not texts:
            return []
        return list(await self.reranker.rerank(query, texts))


class MyScorer:
    """Your function `fn(query, texts) -> list[float]` in 0..1; a sync one runs in a thread.

    Raises:
        ConfigError: `fn` is not callable.
    """

    def __init__(self, fn: ScoreFn) -> None:
        if not callable(fn):
            raise ConfigError("MyScorer needs a function(query, texts) -> list[float]")
        self.fn = fn
        self._is_async = is_async(fn)

    async def score(self, query: str, texts: list[str]) -> list[float]:
        """What `fn` returns for `query` and a copy of `texts`; its errors propagate."""
        if not texts:
            return []
        if self._is_async:
            result = self.fn(query, list(texts))
        else:
            result = await asyncio.to_thread(self.fn, query, list(texts))  # copies contextvars
        if inspect.isawaitable(result):
            result = await result
        return list(result)


def _cosine(a: tuple[float, ...], b: tuple[float, ...], embedder: EmbeddingPort) -> float:
    if len(a) != len(b):
        raise _invalid(embedder, f"vectors of {len(a)} and {len(b)} dimensions")
    norm = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b))
    if norm == 0:
        return 0.0
    return min(1.0, max(0.0, math.fsum(x * y for x, y in zip(a, b, strict=True)) / norm))


def _invalid(embedder: EmbeddingPort, problem: str) -> ModelOutputInvalid:
    provider = str(getattr(embedder, "provider", "embedder"))
    model = str(getattr(embedder, "model", ""))
    return ModelOutputInvalid(f"{provider} {model}: {problem}", provider=provider, model=model)
