"""Embedding cache: one vector per (provider, model, text) over any CachePort."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from agent_adapter.cache.codec import EnvelopeKind, decode, encode
from agent_adapter.cache.keys import KEY_VERSION, digest
from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import CachePort
from agent_adapter.embeddings.port import VECTOR

logger = logging.getLogger(__name__)

_KIND: EnvelopeKind = "embedding"
_PARALLEL = 32  # backend round trips in flight at once


def embed_key(namespace: str, text: str, *, provider: str, model: str) -> str:
    """Key for one text's vector; the text appears only as a digest."""
    # provider:model is for reading only; the digest covers both, as names may hold ":".
    return f"{namespace}:embed:v{KEY_VERSION}:{provider}:{model}:{digest([provider, model, text])}"


class EmbeddingCache:
    """Stores vectors under `embed_key`; backend failures and bad entries are misses."""

    def __init__(self, port: CachePort, *, namespace: str = "aa", ttl: float | None = None) -> None:
        self.port = port
        self.namespace = namespace
        self.ttl = ttl

    async def get_many(
        self, texts: Sequence[str], *, provider: str, model: str
    ) -> dict[str, tuple[float, ...]]:
        """The cached vector of each text that has one."""
        found: dict[str, tuple[float, ...]] = {}
        for start in range(0, len(texts), _PARALLEL):
            chunk = texts[start : start + _PARALLEL]
            vectors = await asyncio.gather(*(self._get(t, provider, model) for t in chunk))
            found.update((t, v) for t, v in zip(chunk, vectors, strict=True) if v is not None)
        logger.debug(
            "embedding cache: %d of %d hit for %s:%s", len(found), len(texts), provider, model
        )
        return found

    async def set_many(
        self, vectors: Mapping[str, tuple[float, ...]], *, provider: str, model: str
    ) -> int:
        """Store each vector; returns how many were stored."""
        items = list(vectors.items())
        stored = 0
        for start in range(0, len(items), _PARALLEL):
            chunk = items[start : start + _PARALLEL]
            done = await asyncio.gather(*(self._set(t, v, provider, model) for t, v in chunk))
            stored += sum(done)
        return stored

    async def _get(self, text: str, provider: str, model: str) -> tuple[float, ...] | None:
        key = embed_key(self.namespace, text, provider=provider, model=model)
        try:
            data = await self.port.get(key)
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "embedding cache get failed for %s:%s",
                provider,
                model,
            )
            return None
        if data is None:
            return None
        envelope = decode(data, _KIND)
        if envelope is None:
            return None
        try:
            return VECTOR.validate_python(envelope.payload)
        except ValidationError:
            logger.debug("embedding cache entry for %s:%s is not a vector; miss", provider, model)
            return None

    async def _set(self, text: str, vector: tuple[float, ...], provider: str, model: str) -> bool:
        key = embed_key(self.namespace, text, provider=provider, model=model)
        try:
            await self.port.set(key, encode(_KIND, list(vector)), self.ttl)
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "embedding cache set failed for %s:%s",
                provider,
                model,
            )
            return False
        return True
