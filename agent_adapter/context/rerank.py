"""The reranker port and `CallableReranker`: any rerank API or local model as a `RerankerPort`."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

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
from agent_adapter.core.ports import EventSink, LedgerPort
from agent_adapter.core.types import Usage
from agent_adapter.embeddings._record import RERANK, EmbedCall
from agent_adapter.embeddings.port import FiniteFloat, plain
from agent_adapter.runtime.current import resolve_context, use_context

STAGE = "compress"  # rerankers here score messages for compression

# fn(query, texts) -> one score per text (a list, or an array with `tolist()`), or a
# RerankResult / mapping with `scores`, `usage` and `model`; sync or async.
RerankFn = Callable[[str, list[str]], Any]


class RerankResult(BaseModel):
    """Scores in input order, plus what the call used, for billing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[FiniteFloat, ...]
    model: str = ""  # the model that answered, if it differs from the reranker's name
    usage: Usage = Usage()  # `units` (e.g. searches), or input tokens billed per million

    @field_validator("scores", mode="before")
    @classmethod
    def _arrays_to_lists(cls, value: Any) -> Any:
        return plain(value)


@runtime_checkable
class RerankerPort(Protocol):
    """Scores how relevant each text is to `query`; one float per text, in order."""

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CallableReranker:
    """Wraps `fn(query, texts)` as a `RerankerPort` that writes a ledger entry and events.

    A sync `fn` runs in a worker thread; `timeout_s` bounds each call (a timed-out thread is
    abandoned, not stopped). The entry is priced as `provider:model` units in the price table.

    Raises:
        ConfigError: a bad `provider`, `model`, `stage` or `timeout_s`.
        ToolUsageError: `fn` is not callable.
    """

    def __init__(
        self,
        fn: RerankFn,
        *,
        provider: str = "callable",
        model: str = "host",
        timeout_s: float | None = None,
        stage: str = STAGE,
        ledger: LedgerPort | None = None,
        sink: EventSink | None = None,
    ) -> None:
        if not callable(fn):
            raise ToolUsageError(f"CallableReranker fn must be callable, got {type(fn).__name__}")
        if not provider or not model or not stage:
            raise ConfigError("CallableReranker needs a non-empty provider, model and stage")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int | float)
            or not (math.isfinite(timeout_s) and timeout_s > 0)
        ):
            raise ConfigError(f"timeout_s must be a finite number > 0, got {timeout_s!r}")
        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self.stage = stage
        self.ledger = ledger
        self.sink = sink
        self._fn = fn
        self._is_async = is_async(fn)

    async def rerank(
        self, query: str, texts: Sequence[str], *, ctx: ToolContext | None = None
    ) -> list[float]:
        """One score per text, in order; no call (and no entry) for an empty list.

        Raises:
            ToolUsageError: `query` is not a str or `texts` not a sequence of str.
            ModelTimeout: the call took longer than `timeout_s`.
            ModelOutputInvalid: `fn` returned the wrong count, a non-finite score or bad usage.
            ModelError: `fn` raised a non-adapter error (it is the `__cause__`).
        """
        items = _check_inputs(query, texts)
        if not items:
            return []
        call = EmbedCall(
            provider=self.provider,
            model=self.model,
            ledger=self.ledger,
            sink=self.sink,
            ctx=resolve_context(ctx),
            texts=len(items),
            kind=RERANK,
            stage=self.stage,
        )
        call.started()
        try:
            result = self._coerce(await self._call(query, items, call.ctx), len(items))
        except BaseException as exc:
            call.failed(exc)
            raise
        call.paid(result.usage, result.model)
        call.finished()
        return list(result.scores)

    async def _call(self, query: str, items: list[str], ctx: ToolContext) -> Any:
        try:
            with use_context(ctx):
                async with asyncio.timeout(self.timeout_s) as scope:
                    if self._is_async:
                        raw = await self._fn(query, items)
                    else:
                        raw = await asyncio.to_thread(self._fn, query, items)  # copies contextvars
                    if inspect.isawaitable(raw):
                        raw = await raw
        except AdapterError:
            raise
        except TimeoutError as exc:
            # Ours only when the scope expired; else fn raised it (e.g. a socket timeout).
            problem = (
                f"rerank fn timed out after {self.timeout_s}s"
                if scope.expired()
                else f"rerank fn raised {qualified_name(type(exc))}"
            )
            raise ModelTimeout(
                f"{self.provider} {self.model}: {problem}", provider=self.provider, model=self.model
            ) from exc
        except Exception as exc:
            raise ModelError(
                f"{self.provider} {self.model}: rerank fn failed ({qualified_name(type(exc))})",
                provider=self.provider,
                model=self.model,
            ) from exc
        return raw

    def _coerce(self, raw: Any, expected: int) -> RerankResult:
        try:
            if isinstance(raw, RerankResult):
                result = raw
            elif isinstance(raw, Mapping):
                self._check_usage_keys(raw.get("usage"))
                result = RerankResult.model_validate(dict(raw))
            else:
                values = plain(raw)
                if isinstance(values, str | bytes) or not isinstance(values, Sequence):
                    raise self._invalid(f"fn returned {type(raw).__name__}, not a list of scores")
                if any(isinstance(value, bool) for value in values):
                    raise self._invalid("fn returned a score that is not a finite number")
                result = RerankResult(scores=tuple(values))
        except ValidationError as exc:
            raise self._invalid("fn returned a score that is not a finite number") from exc
        if len(result.scores) != expected:
            raise self._invalid(f"fn returned {len(result.scores)} scores for {expected} texts")
        return result

    def _check_usage_keys(self, usage: object) -> None:
        """Reject provider-shaped usage keys, which would otherwise be dropped and bill $0."""
        if not isinstance(usage, Mapping):
            return
        unknown = sorted(str(key) for key in usage if key not in Usage.model_fields)
        if unknown:
            raise self._invalid(f"usage keys {unknown} are not Usage fields")

    def _invalid(self, problem: str) -> ModelOutputInvalid:
        return ModelOutputInvalid(
            f"{self.provider} {self.model}: {problem}", provider=self.provider, model=self.model
        )


def _check_inputs(query: object, texts: object) -> list[str]:
    """`texts` as a list.

    Raises:
        ToolUsageError: `query` is not a str or `texts` not a sequence of str.
    """
    if not isinstance(query, str):
        raise ToolUsageError(f"query must be a str, got {type(query).__name__}")
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise ToolUsageError(f"texts must be a sequence of str, got {type(texts).__name__}")
    items = list(texts)
    if not all(isinstance(item, str) for item in items):
        raise ToolUsageError("texts must hold only str values")
    return items
