"""The shared front of every retriever: checks, filter enforcement, events, the sync twin."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError, ConfigError, IntegrationError, log_failure
from agent_adapter.exec._sugar import blocking
from agent_adapter.retrieval.documents import (
    Document,
    Filters,
    FilterValue,
    as_document,
    check_filters,
    check_k,
    check_query,
    matches,
)
from agent_adapter.runtime.current import current_context, use_context
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)


class GuardedRetriever(ABC):
    """A `RetrieverPort` that checks its inputs and never returns a document outside `filters`.

    Backends implement `_retrieve`. Every result is re-checked here: a document whose
    metadata does not match the filters is dropped, so a backend that ignores filters fails
    closed (fewer results), never open.
    """

    backend = "retriever"

    def __init__(
        self, *, name: str = "", timeout_s: float | None = None, capture_content: bool = False
    ) -> None:
        if timeout_s is not None and not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ConfigError(f"timeout_s must be a finite number > 0, got {timeout_s!r}")
        self.name = name or self.backend
        self.timeout_s = timeout_s
        self.capture_content = bool(capture_content)

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filters: Filters | None = None,
        ctx: ToolContext | None = None,
    ) -> list[Document]:
        """Up to `k` documents for `query` whose metadata matches every filter, best first.

        Raises:
            ToolUsageError: bad query, k or filters, or the backend returned a non-document.
            IntegrationError: the backend failed or timed out (the cause is chained).
        """
        query, k, clean = check_query(query), check_k(k), check_filters(filters)
        ctx = ctx if ctx is not None else current_context()
        started = time.perf_counter()
        try:
            raw = await self._call(query, k, clean, ctx)
            found = [as_document(item) for item in raw]
        except Exception as exc:
            kind = exc.kind if isinstance(exc, AdapterError) else "integration-error"
            self._event(EventType.RETRIEVAL_FAILED, ctx, started, k, error_kind=kind)
            raise
        kept = self._kept(found, clean)
        result = kept[:k]
        content = {"arguments": {"query": query}, "result": [doc.text for doc in result]}
        self._event(
            EventType.RETRIEVAL_SEARCHED,
            ctx,
            started,
            k,
            content,
            count=len(result),
            ids=[doc.id for doc in result],
            dropped=len(found) - len(kept),  # repeated ids and documents outside the filters
        )
        return result

    def retrieve_sync(
        self, query: str, *, timeout: float | None = None, **kw: Any
    ) -> list[Document]:
        """`retrieve` from sync code, on the process-wide loop."""
        return blocking(lambda: self.retrieve(query, **kw), timeout)

    @abstractmethod
    async def _retrieve(
        self, query: str, k: int, filters: dict[str, FilterValue]
    ) -> Sequence[Any]: ...

    async def _call(
        self, query: str, k: int, filters: dict[str, FilterValue], ctx: ToolContext | None
    ) -> Sequence[Any]:
        try:
            async with asyncio.timeout(self.timeout_s) as scope:
                if ctx is None:
                    return await self._retrieve(query, k, filters)
                with use_context(ctx):
                    return await self._retrieve(query, k, filters)
        except AdapterError:
            raise
        except TimeoutError as exc:
            if not scope.expired():
                raise self._failed(exc) from exc
            logger.warning("%s retrieval timed out after %ss", self.name, self.timeout_s)
            raise IntegrationError(
                f"{self.name} retrieval timed out after {self.timeout_s}s",
                integration=self.backend,
            ) from exc
        except Exception as exc:
            raise self._failed(exc) from exc

    def _failed(self, exc: BaseException) -> IntegrationError:
        log_failure(logger, logging.ERROR, exc, "%s retrieval failed", self.name)
        return IntegrationError(f"{self.name} retrieval failed", integration=self.backend)

    def _kept(self, found: list[Document], filters: dict[str, FilterValue]) -> list[Document]:
        """`found` without repeated ids and without documents outside `filters`."""
        seen: set[str] = set()
        kept: list[Document] = []
        outside = 0
        for doc in found:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            if matches(doc, filters):
                kept.append(doc)
            else:
                outside += 1
        if outside:
            logger.warning(
                "%s retrieval returned %d documents outside the filters; dropped them",
                self.name,
                outside,
            )
        return kept

    def _event(
        self,
        type: str,
        ctx: ToolContext | None,
        started: float,
        k: int,
        content: dict[str, Any] | None = None,
        **data: Any,
    ) -> None:
        if ctx is None or ctx.sink is None:
            return
        latency = round((time.perf_counter() - started) * 1000, 3)
        extra = content if self.capture_content and content else {}
        event = make_event(
            type,
            ctx,
            retriever=self.name,
            backend=self.backend,
            k=k,
            latency_ms=latency,
            **data,
            **extra,
        )
        emit(ctx.sink, event)
