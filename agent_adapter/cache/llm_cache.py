"""LLM response cache over any CachePort; backend failures become misses."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agent_adapter.cache.codec import decode, encode
from agent_adapter.cache.keys import llm_key
from agent_adapter.core.errors import log_failure
from agent_adapter.core.messages import LLMRequest, LLMResponse
from agent_adapter.core.ports import CachePort

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """Stores normalised `LLMResponse`s under `llm_key` (provider, model, request, fields)."""

    def __init__(self, port: CachePort, *, namespace: str = "aa", ttl: float | None = None) -> None:
        self.port = port
        self.namespace = namespace
        self.ttl = ttl

    async def get(
        self, request: LLMRequest, *, provider: str, model: str, fields: Mapping[str, Any]
    ) -> LLMResponse | None:
        """The cached response with `cached=True` and `latency_ms=0`, or None."""
        key = self._key(request, provider, model, fields)
        if not key:
            return None
        try:
            data = await self.port.get(key)
        except Exception as exc:
            log_failure(
                logger, logging.WARNING, exc, "llm cache get failed for %s:%s", provider, model
            )
            return None
        if data is None:
            logger.debug("llm cache miss for %s:%s", provider, model)
            return None
        envelope = decode(data, "llm")
        if envelope is None:
            return None
        try:
            cached = LLMResponse.model_validate(envelope.payload)
        except ValidationError:
            logger.debug("llm cache entry for %s:%s is not a response; miss", provider, model)
            return None
        return cached.model_copy(update={"cached": True, "latency_ms": 0.0})

    async def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
        *,
        provider: str,
        model: str,
        fields: Mapping[str, Any],
    ) -> bool:
        """Store `response`; False for errors, cache hits, opt-outs or backend failures."""
        if response.finish_reason == "error" or response.cached:
            logger.debug("llm response from %s:%s not stored (error or hit)", provider, model)
            return False
        key = self._key(request, provider, model, fields)
        if not key:
            return False
        try:
            data = encode("llm", response.model_dump(mode="json"))
        except (TypeError, ValueError) as exc:  # PydanticSerializationError is a ValueError
            logger.debug("llm response from %s:%s not cacheable (%s)", provider, model, exc)
            return False
        try:
            await self.port.set(key, data, self.ttl)
        except Exception as exc:
            log_failure(
                logger, logging.WARNING, exc, "llm cache set failed for %s:%s", provider, model
            )
            return False
        return True

    def _key(
        self, request: LLMRequest, provider: str, model: str, fields: Mapping[str, Any]
    ) -> str:
        if not request.response_cache:
            logger.debug("request opted out of the llm response cache")
            return ""
        try:
            return llm_key(self.namespace, request, provider=provider, model=model, fields=fields)
        except TypeError as exc:
            logger.debug("llm request to %s:%s not cacheable (%s)", provider, model, exc)
            return ""
