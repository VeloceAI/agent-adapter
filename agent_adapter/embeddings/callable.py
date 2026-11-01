"""`CallableEmbedder`: any host embedding function as an `EmbeddingPort`."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.callables import is_async
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    AdapterError,
    ConfigError,
    ModelError,
    ModelOutputInvalid,
    ModelTimeout,
    ToolUsageError,
    qualified_name,
)
from agent_adapter.core.ports import CachePort, EventSink, LedgerPort
from agent_adapter.embeddings._record import EmbedCall
from agent_adapter.embeddings.cache import EmbeddingCache
from agent_adapter.embeddings.port import EmbeddingResult
from agent_adapter.runtime.current import resolve_context, use_context

logger = logging.getLogger(__name__)

# fn(texts) -> vectors (rows of numbers, or an array with `tolist()`) or an EmbeddingResult;
# sync or async.
EmbedFn = Callable[[list[str]], Any]


class CallableEmbedder:
    """Wraps `fn(texts)` with batching, an optional per-text cache, a ledger entry and events.

    A sync `fn` runs in a worker thread with the caller's contextvars. `timeout_s` bounds each
    `fn` call; a timed-out thread is abandoned, not stopped.

    Raises:
        ConfigError: a bad `model`, `provider`, `batch_size` or `timeout_s`.
        ToolUsageError: `fn` is not callable.
    """

    def __init__(
        self,
        fn: EmbedFn,
        *,
        model: str = "host",
        provider: str = "callable",
        batch_size: int = 64,
        timeout_s: float | None = None,
        cache: CachePort | None = None,
        cache_ttl: float | None = None,
        namespace: str = "aa",
        ledger: LedgerPort | None = None,
        sink: EventSink | None = None,
    ) -> None:
        if not callable(fn):
            raise ToolUsageError(f"CallableEmbedder fn must be callable, got {type(fn).__name__}")
        if not model or not provider:
            raise ConfigError("CallableEmbedder needs a non-empty model and provider")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError(f"batch_size must be an int >= 1, got {batch_size!r}")
        if timeout_s is not None and not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ConfigError(f"timeout_s must be a finite number > 0, got {timeout_s!r}")
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.ledger = ledger
        self.sink = sink
        self._fn = fn
        self._is_async = is_async(fn)
        self._cache = (
            EmbeddingCache(cache, namespace=namespace, ttl=cache_ttl) if cache is not None else None
        )

    async def embed(
        self, texts: Sequence[str], *, ctx: ToolContext | None = None
    ) -> EmbeddingResult:
        """Vectors for `texts`, in order; repeated texts are embedded once.

        Raises:
            ToolUsageError: `texts` is not a sequence of strings.
            ModelTimeout: one `fn` call took longer than `timeout_s`.
            ModelOutputInvalid: `fn` returned no valid vectors, the wrong count, or mixed sizes.
            ModelError: `fn` raised a non-adapter error (it is the `__cause__`).
        """
        items = _check_texts(texts)
        if not items:
            return EmbeddingResult(vectors=(), model=self.model)
        call = EmbedCall(
            provider=self.provider,
            model=self.model,
            ledger=self.ledger,
            sink=self.sink,
            ctx=resolve_context(ctx),
            texts=len(items),
        )
        call.started()
        try:
            result = await self._embed(items, call)
        except BaseException as exc:
            call.failed(exc)
            raise
        call.finished()
        return result

    async def _embed(self, items: list[str], call: EmbedCall) -> EmbeddingResult:
        unique = list(dict.fromkeys(items))
        found: dict[str, tuple[float, ...]] = {}
        if self._cache is not None:
            found = await self._cache.get_many(unique, provider=self.provider, model=self.model)
        fresh = await self._fresh([text for text in unique if text not in found], found, call)
        call.cache_hits = sum(text in found for text in items)
        ordered = tuple(found[text] if text in found else fresh[text] for text in items)
        if len({len(vector) for vector in ordered}) > 1:
            raise self._invalid("cached vectors of different dimensions; rename model or clear")
        return EmbeddingResult(
            vectors=ordered,
            model=call.billed_model or self.model,
            usage=call.usage,
            cache_hits=call.cache_hits,
        )

    async def _fresh(
        self, missing: list[str], found: dict[str, tuple[float, ...]], call: EmbedCall
    ) -> dict[str, tuple[float, ...]]:
        """Embed `missing` in batches; each batch is checked before it is cached.

        The first fresh batch sets the dimension: cached vectors of another one are stale, so
        they leave `found` and are embedded again (and overwritten).
        """
        fresh: dict[str, tuple[float, ...]] = {}
        queue, dimension, start = list(missing), 0, 0
        while start < len(queue):
            batch = queue[start : start + self.batch_size]
            start += len(batch)
            part = await self._call(batch, call.ctx)
            call.paid(part.usage, part.model)
            size = len(part.vectors[0])
            if not dimension:
                dimension = size
                queue.extend(self._drop_stale(found, dimension))
            elif size != dimension:
                raise self._invalid(f"fn returned {size}-dimension vectors after {dimension}")
            vectors = dict(zip(batch, part.vectors, strict=True))
            fresh.update(vectors)
            if self._cache is not None:
                # Per batch, so a later failure does not throw away vectors already paid for.
                await self._cache.set_many(vectors, provider=self.provider, model=self.model)
        return fresh

    def _drop_stale(self, found: dict[str, tuple[float, ...]], dimension: int) -> list[str]:
        """Remove and return the cached texts whose vectors are not `dimension` long."""
        stale = [text for text, vector in found.items() if len(vector) != dimension]
        for text in stale:
            del found[text]
        if stale:
            logger.warning(
                "%s %s: %d cached vectors have another dimension than %d; embedding them again",
                self.provider,
                self.model,
                len(stale),
                dimension,
            )
        return stale

    async def _call(self, batch: list[str], ctx: ToolContext) -> EmbeddingResult:
        try:
            with use_context(ctx):
                async with asyncio.timeout(self.timeout_s) as scope:
                    raw = await self._run(list(batch))
        except AdapterError:
            raise
        except TimeoutError as exc:
            # Ours only when the scope expired; else fn raised it (e.g. a socket timeout).
            problem = (
                f"embedding fn timed out after {self.timeout_s}s"
                if scope.expired()
                else f"embedding fn raised {qualified_name(type(exc))}"
            )
            raise ModelTimeout(
                f"{self.provider} {self.model}: {problem}",
                provider=self.provider,
                model=self.model,
            ) from exc
        except Exception as exc:
            raise ModelError(
                f"{self.provider} {self.model}: embedding fn failed ({qualified_name(type(exc))})",
                provider=self.provider,
                model=self.model,
            ) from exc
        return self._coerce(raw, len(batch))

    async def _run(self, batch: list[str]) -> Any:
        if self._is_async:
            raw = await self._fn(batch)
        else:
            raw = await asyncio.to_thread(self._fn, batch)  # copies the contextvars
        if inspect.isawaitable(raw):
            raw = await raw
        return raw

    def _coerce(self, raw: Any, expected: int) -> EmbeddingResult:
        if isinstance(raw, EmbeddingResult):
            result = raw
        else:
            try:
                result = EmbeddingResult(vectors=raw)
            except ValidationError as exc:
                raise self._invalid("fn returned no valid vectors") from exc
        if len(result.vectors) != expected:
            raise self._invalid(f"fn returned {len(result.vectors)} vectors for {expected} texts")
        if len({len(vector) for vector in result.vectors}) > 1:
            raise self._invalid("fn returned vectors of different dimensions")
        return result

    def _invalid(self, problem: str) -> ModelOutputInvalid:
        return ModelOutputInvalid(
            f"{self.provider} {self.model}: {problem}", provider=self.provider, model=self.model
        )


def _check_texts(texts: Any) -> list[str]:
    """`texts` as a list.

    Raises:
        ToolUsageError: not a sequence of strings (a lone string is rejected too).
    """
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise ToolUsageError(f"texts must be a sequence of str, got {type(texts).__name__}")
    items = list(texts)
    if not all(isinstance(item, str) for item in items):
        raise ToolUsageError("texts must hold only str values")
    return items
