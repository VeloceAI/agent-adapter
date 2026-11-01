"""Embedding test helpers: a deterministic fake embedder that records its batches."""

from __future__ import annotations

import threading
from typing import Any

from agent_adapter.core.types import Usage
from agent_adapter.embeddings import EmbeddingResult


def vector(text: str) -> tuple[float, float, float]:
    """A 3-d vector: length, vowels, spaces."""
    return (float(len(text)), float(sum(c in "aeiou" for c in text)), float(text.count(" ")))


class Recorder:
    """A sync and an async embedder fn; `batches` holds every list it received."""

    def __init__(self, *, tokens_per_text: int = 0, model: str = "") -> None:
        self.batches: list[list[str]] = []
        self.threads: list[int] = []
        self.tokens_per_text = tokens_per_text
        self.model = model

    def _answer(self, texts: list[str]) -> Any:
        self.batches.append(list(texts))
        self.threads.append(threading.get_ident())
        vectors = [vector(t) for t in texts]
        if not self.tokens_per_text and not self.model:
            return vectors
        usage = Usage(input_tokens=self.tokens_per_text * len(texts))
        return EmbeddingResult(vectors=vectors, model=self.model, usage=usage)

    def sync(self, texts: list[str]) -> Any:
        return self._answer(texts)

    async def run(self, texts: list[str]) -> Any:
        return self._answer(texts)
