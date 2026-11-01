"""Gemini prompt caching: explicit `caches.create` for the static prefix, or implicit."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Any

from agent_adapter.cache.codec import decode, encode
from agent_adapter.cache.keys import canonical_json, digest, gcache_key
from agent_adapter.cache.memory import MemoryCache
from agent_adapter.cache.providers._base import caps_for
from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.messages import LLMRequest
from agent_adapter.core.ports import CachePort, CapabilityLookup
from agent_adapter.core.types import LedgerEntry
from agent_adapter.runtime.current import current_context
from agent_adapter.runtime.loop_local import LoopLocal

logger = logging.getLogger(__name__)

_NAMESPACE = "aa"
_FAILURE_WINDOW_S = 300.0  # a failed creation is not retried for min(ttl_s, this)
_MAX_FAILURE_MARKERS = 1024  # oldest markers are dropped beyond this
# Gemini rejects generate requests that set these together with `cached_content`,
# so they move into the cache.
_PREFIX_FIELDS = ("system_instruction", "tools", "tool_config")


class GeminiExplicitCache:
    """Creates one Gemini cached content per (model, prefix), then sends its name instead."""

    name = "gemini_explicit_cache"
    providers = frozenset({"gemini"})

    def __init__(
        self,
        *,
        ttl_s: float = 3600,
        store: CachePort | None = None,
        caps: CapabilityLookup | None = None,
    ) -> None:
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ConfigError(f"GeminiExplicitCache ttl_s must be > 0, got {ttl_s}")
        self.ttl_s = float(ttl_s)
        self.store: CachePort = store if store is not None else MemoryCache(max_entries=1000)
        self.caps = caps
        self._clock = time.monotonic
        # One in-flight creation per key and loop; waiters share its outcome.
        self._inflight: LoopLocal[dict[str, asyncio.Future[str]]] = LoopLocal(dict)
        # key -> monotonic time after which a failed creation may be retried (all loops).
        self._failed: dict[str, float] = {}
        self._failed_lock = threading.Lock()

    def key_fields(self) -> dict[str, Any]:
        """Constructor options."""
        return {"ttl_s": self.ttl_s}

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]:
        """Kwargs whose config uses the cached prefix; unchanged when not eligible."""
        caps = caps_for(self.caps, model)
        if not caps.explicit_cache:
            logger.debug("%s: %s has no explicit caching; unchanged", self.name, model)
            return kwargs
        config = _config_dict(kwargs.get("config"))
        prefix = {name: config[name] for name in _PREFIX_FIELDS if config.get(name)}
        if "system_instruction" not in prefix or config.get("cached_content"):
            logger.debug("%s: no system instruction, or cached_content already set", self.name)
            return kwargs
        try:
            text = canonical_json(prefix)
        except TypeError as exc:
            logger.debug("%s: prefix is not serialisable (%s); unchanged", self.name, exc)
            return kwargs
        estimate = len(text) // 4
        if estimate < caps.min_cache_tokens:
            logger.debug(
                "%s: prefix ~%d tokens < %d; unchanged", self.name, estimate, caps.min_cache_tokens
            )
            return kwargs
        only_system = len(prefix) == 1 and isinstance(prefix["system_instruction"], str)
        prefix_digest = digest(prefix["system_instruction"] if only_system else prefix)
        name = await self._cache_name(model, prefix_digest, prefix, estimate, client)
        if not name:
            return kwargs
        new_config = {k: v for k, v in config.items() if k not in _PREFIX_FIELDS}
        new_config["cached_content"] = name
        return {**kwargs, "config": new_config}

    async def _cache_name(
        self, model: str, prefix_digest: str, prefix: dict[str, Any], estimate: int, client: Any
    ) -> str:
        key = gcache_key(_NAMESPACE, model, prefix_digest)
        name = await self._stored(key)
        if name:
            return name
        if self._failing(key):
            logger.debug("%s: creation failed recently for %s; unchanged", self.name, model)
            return ""
        inflight = self._inflight.get()
        task = inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._create_once(model, key, prefix, estimate, client))
            inflight[key] = task
            task.add_done_callback(lambda done: _forget(inflight, key, done))
        # A cancelled waiter must not cancel the creation the other waiters share.
        return await asyncio.shield(task)

    async def _create_once(
        self, model: str, key: str, prefix: dict[str, Any], estimate: int, client: Any
    ) -> str:
        name = await self._stored(key)  # another loop may have created it meanwhile
        if name:
            return name
        name = await self._create(model, key, prefix, estimate, client)
        if name:
            await self._remember(key, name, model)
        else:
            self._mark_failed(key)
        return name

    def _mark_failed(self, key: str) -> None:
        now = self._clock()
        with self._failed_lock:
            # Markers expire in insertion order (one window length), so prune from the front.
            self._failed.pop(key, None)
            while self._failed:
                old = next(iter(self._failed))
                if self._failed[old] > now and len(self._failed) < _MAX_FAILURE_MARKERS:
                    break
                del self._failed[old]
            self._failed[key] = now + min(self.ttl_s, _FAILURE_WINDOW_S)

    def _failing(self, key: str) -> bool:
        with self._failed_lock:
            until = self._failed.get(key)
            if until is not None and self._clock() >= until:
                del self._failed[key]
                until = None
        return until is not None

    async def _stored(self, key: str) -> str:
        try:
            data = await self.store.get(key)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "%s: cache-name store get failed", self.name)
            return ""
        envelope = decode(data, "gcache") if data is not None else None
        payload = envelope.payload if envelope is not None else None
        name = payload.get("name") if isinstance(payload, dict) else None
        return name if isinstance(name, str) else ""

    async def _remember(self, key: str, name: str, model: str) -> None:
        # Forget the name before Gemini expires the cache, so it is never sent stale.
        margin = min(60.0, self.ttl_s * 0.1)
        payload = {"name": name, "model": model, "expires_at": time.time() + self.ttl_s}
        try:
            await self.store.set(key, encode("gcache", payload), self.ttl_s - margin)
        except Exception as exc:
            log_failure(logger, logging.WARNING, exc, "%s: cache-name store set failed", self.name)

    async def _create(
        self, model: str, key: str, prefix: dict[str, Any], estimate: int, client: Any
    ) -> str:
        started = time.time()
        try:
            created = await client.aio.caches.create(
                model=model,
                config={**prefix, "ttl": _duration(self.ttl_s), "display_name": key[:128]},
            )
        except Exception as exc:
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "%s: caches.create failed for %s; request sent uncached",
                self.name,
                model,
            )
            return ""
        name = getattr(created, "name", None)
        if not isinstance(name, str) or not name:
            logger.warning("%s: caches.create for %s returned no name", self.name, model)
            return ""
        tokens = getattr(getattr(created, "usage_metadata", None), "total_token_count", None)
        self._record_storage(model, name, tokens, estimate, started)
        logger.debug("%s: created %s for %s", self.name, name, model)
        return name

    def _record_storage(
        self, model: str, name: str, tokens: Any, estimate: int, started: float
    ) -> None:
        ctx = current_context()
        if ctx is None or ctx.ledger is None:
            logger.debug("%s: no ledger in the current context; storage not recorded", self.name)
            return
        reported = isinstance(tokens, int) and not isinstance(tokens, bool)
        count = tokens if reported else estimate
        ended = time.time()
        ctx.ledger.record(
            LedgerEntry(
                run_id=ctx.run_id,
                call_id=ctx.call_id,
                entry_id=f"gcache:{name}",
                kind="api",
                stage="cache_storage",
                provider="gemini",
                model=f"{model}:cache_storage",
                units=count * (self.ttl_s / 3600) / 1e6,  # 1M token-hours, billed for the TTL
                started_at=started,
                ended_at=ended,
                latency_ms=(ended - started) * 1000,
                thread_id=ctx.thread_id,
                metadata={
                    "cache_name": name,
                    "ttl_s": f"{self.ttl_s:g}",
                    "tokens": str(count),
                    "tokens_estimated": str(not reported).lower(),
                },
            )
        )


class GeminiImplicitCache:
    """Gemini 2.5+ caches repeated prefixes on its own; pair it with a `PromptLayout`."""

    name = "gemini_implicit_cache"
    providers = frozenset({"gemini"})

    def key_fields(self) -> dict[str, Any]:
        """No options."""
        return {}

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]:
        """The kwargs, unchanged."""
        return dict(kwargs)


def _forget(inflight: dict[str, asyncio.Future[str]], key: str, done: asyncio.Future[str]) -> None:
    if inflight.get(key) is done:
        del inflight[key]


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "model_dump"):  # a google.genai.types.GenerateContentConfig
        dumped: dict[str, Any] = config.model_dump(exclude_none=True)
        return dumped
    return {}


def _duration(seconds: float) -> str:
    """Gemini duration string ("3600s", "2.5s")."""
    if seconds == int(seconds):
        return f"{int(seconds)}s"
    return f"{seconds:.3f}".rstrip("0") + "s"
