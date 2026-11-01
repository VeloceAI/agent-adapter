"""`FallbackModel`: try ports in order, moving on only after transient failures."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_adapter.core.errors import (
    ConfigError,
    ModelError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import LLMRequest, LLMResponse, StreamEvent
from agent_adapter.core.ports import EventSink, LLMPort
from agent_adapter.models._port import closing
from agent_adapter.runtime.current import current_context
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import first_sink

logger = logging.getLogger(__name__)

DEFAULT_FALL_ON: tuple[type[ModelError], ...] = (ModelRateLimited, ModelUnavailable, ModelTimeout)


class FallbackModel:
    """An `LLMPort` over `models`: the next one answers after a `fall_on` error.

    Each port keeps its SDK retries, so a hop happens after those are spent; pass
    `client_kwargs={"max_retries": 0}` to the provider ports for fast failover. Ports after the
    first get the request with `model=None`, so each uses its own configured model. With
    `metadata["call_id"]`, every hop (hop 0 and a single port too) gets `{call_id}:hop{i}`.
    Hops emit `llm.fallback` to `sink`, else to the current context's sink.

    Raises:
        ConfigError: no models, or a `fall_on` entry that is not a `ModelError` subclass.
    """

    provider = "fallback"

    def __init__(
        self,
        models: Sequence[LLMPort],
        *,
        fall_on: tuple[type[ModelError], ...] = DEFAULT_FALL_ON,
        sink: EventSink | None = None,
    ) -> None:
        if not models:
            raise ConfigError("FallbackModel needs at least one model")
        bad = [
            cls for cls in fall_on if not (isinstance(cls, type) and issubclass(cls, ModelError))
        ]
        if bad:
            raise ConfigError(f"fall_on entries must be ModelError subclasses, got {bad!r}")
        self.models = tuple(models)
        self.fall_on = fall_on
        self.sink = sink
        self.model = self.models[0].model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """The first successful reply in order; the last error when every port fails.

        Raises:
            ModelError: a non-`fall_on` error (at once), or the last port's error.
        """
        errors: list[ModelError] = []
        for index, port in enumerate(self.models):
            try:
                return await port.generate(self._request(request, index))
            except self.fall_on as exc:
                self._hop(request, index, exc)
                errors.append(exc)
        raise errors[-1]

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """The first port's stream that starts; a port that fails after its first event is final.

        Raises:
            ModelError: as `generate`, or any error once events have been yielded.
        """
        errors: list[ModelError] = []
        for index, port in enumerate(self.models):
            started = False
            try:
                async with closing(port.stream(self._request(request, index))) as events:
                    async for event in events:
                        started = True
                        yield event
                return
            except self.fall_on as exc:
                if started:
                    raise
                self._hop(request, index, exc)
                errors.append(exc)
        raise errors[-1]

    def cache_key_fields(self) -> dict[str, Any]:
        """LLM-cache key material: every port in order."""
        return {
            "api": "fallback",
            "models": [
                {"provider": m.provider, "model": m.model, **m.cache_key_fields()}
                for m in self.models
            ],
        }

    def _request(self, request: LLMRequest, index: int) -> LLMRequest:
        update: dict[str, Any] = {} if index == 0 else {"model": None}
        call_id = request.metadata.get("call_id")
        if call_id:
            # Each hop gets its own ledger entry id, so an instrumented hop never overwrites
            # another hop's (or the enclosing call's) entry.
            update["metadata"] = {**request.metadata, "call_id": f"{call_id}:hop{index}"}
        return request.model_copy(update=update) if update else request

    def _hop(self, request: LLMRequest, index: int, error: ModelError) -> None:
        port = self.models[index]
        call_id = request.metadata.get("call_id", "")
        if index + 1 == len(self.models):
            logger.warning(
                "fallback: last model %s/%s failed with %s (call %s)",
                port.provider,
                port.model,
                error.kind,
                call_id or "-",
            )
            return
        target = self.models[index + 1]
        logger.warning(
            "fallback: %s/%s failed with %s; trying %s/%s (call %s)",
            port.provider,
            port.model,
            error.kind,
            target.provider,
            target.model,
            call_id or "-",
        )
        ctx = current_context()
        event = make_event(
            EventType.LLM_FALLBACK,
            ctx,
            from_provider=port.provider,
            from_model=(request.model or port.model) if index == 0 else port.model,
            to_provider=target.provider,
            to_model=target.model,
            error_kind=error.kind,
        )
        emit(first_sink(self.sink, ctx.sink if ctx is not None else None), event)
