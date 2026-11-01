"""`InstrumentedModel`: hooks, the LLM response cache, the ledger and events around any port."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_adapter.cache.llm_cache import LLMResponseCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import control_flow_member
from agent_adapter.core.errors import ApprovalPending, ModelOutputInvalid
from agent_adapter.core.messages import LLMRequest, LLMResponse, StreamDone, StreamEvent
from agent_adapter.core.ports import CachePort, EventSink, LedgerPort, LLMPort
from agent_adapter.hooks.base import Hook
from agent_adapter.hooks.dispatch import run_after, run_before, run_on_error
from agent_adapter.models._llm_call import (
    LLMCall,
    Settings,
    Source,
    as_response,
    check_hooks,
    with_ports,
)
from agent_adapter.models._port import closing, replay
from agent_adapter.runtime.current import resolve_context

logger = logging.getLogger(__name__)


class InstrumentedModel:
    """An `LLMPort` that runs `model` with hooks, a response cache, ledger entries and events.

    The model's own `ledger` / `sink` win; without them the call context's are used. The
    response cache is used only when `cache` is set (`executor.model(m)` passes the
    executor's). Each call records one `kind="llm"` ledger entry.

    Raises:
        ToolUsageError: a hook that is not a `Hook`.
    """

    def __init__(
        self,
        model: LLMPort,
        *,
        hooks: Sequence[Hook] = (),
        ledger: LedgerPort | None = None,
        cache: CachePort | None = None,
        cache_ttl: float | None = None,
        sink: EventSink | None = None,
        namespace: str = "aa",
        ctx: ToolContext | None = None,
    ) -> None:
        self.wrapped = model
        self.hooks = check_hooks(hooks)
        self.ledger = ledger
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.sink = sink
        self.namespace = namespace
        self.ctx = ctx
        self._responses = (
            LLMResponseCache(cache, namespace=namespace, ttl=cache_ttl)
            if cache is not None
            else None
        )

    @property
    def provider(self) -> str:
        return self.wrapped.provider

    @property
    def model(self) -> str:
        return self.wrapped.model

    def bind(self, ctx: ToolContext) -> InstrumentedModel:
        """A copy whose calls run in `ctx` (run id, ledger entry ids, ports)."""
        return InstrumentedModel(
            self.wrapped,
            hooks=self.hooks,
            ledger=self.ledger,
            cache=self.cache,
            cache_ttl=self.cache_ttl,
            sink=self.sink,
            namespace=self.namespace,
            ctx=ctx,
        )

    def cache_key_fields(self) -> dict[str, Any]:
        """The wrapped port's LLM-cache key material."""
        return self.wrapped.cache_key_fields()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """One instrumented call: hooks, response cache, the port, ledger entry, events.

        Raises:
            ModelError: the port's error when no `on_llm_error` hook recovers it.
            ToolUsageError: a hook returned something other than an `LLMResponse`.
            ConfigError: `request.compression` is invalid or could not run.
        """
        if request.compression is not None:
            # Compressed first, so hooks, the response cache and the ledger see what is sent.
            from agent_adapter.context.model import generate_compressed

            port, ctx = self._for_compression()
            return await generate_compressed(port, request, request.compression, ctx=ctx)
        call = LLMCall(self._settings(), request, self.ctx, stream=request.stream)
        call.started()
        try:
            response, source = await self._produce(call)
        except BaseException as exc:
            call.failed(exc)
            raise
        call.finished(response, source)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """The port's stream with the same hooks, ledger entry and events (no response cache).

        Raises:
            ModelError: as `generate`; a failure after the first event is not offered to hooks.
            ConfigError: `request.compression` is invalid or could not run.
        """
        if request.compression is not None:
            from agent_adapter.context.model import stream_compressed

            port, ctx = self._for_compression()
            compressed = stream_compressed(port, request, request.compression, ctx=ctx)
            async with closing(compressed) as events:
                async for event in events:
                    yield event
            return
        call = LLMCall(self._settings(), request, self.ctx, stream=True)
        call.started()
        try:
            async with closing(self._stream_events(call)) as events:
                async for event in events:
                    yield event
        except BaseException as exc:
            call.failed(exc)  # a closed stream (GeneratorExit) counts as cancelled
            raise

    # --- generate -----------------------------------------------------------------------

    async def _produce(self, call: LLMCall) -> tuple[LLMResponse, Source]:
        short = await run_before(self.hooks, "before_llm", call.inv)
        if short is not None:
            response: LLMResponse = as_response(short.value, "before_llm")
            source: Source = "hook"
        else:
            response, source = await self._cached_or_called(call)
        return await self._after(call, response), source

    async def _cached_or_called(self, call: LLMCall) -> tuple[LLMResponse, Source]:
        responses = self._responses if not call.request.stream else None
        fields = call.settings.fields
        if responses is not None:
            hit = await responses.get(
                call.request, provider=self.provider, model=call.model, fields=fields
            )
            if hit is not None:
                return hit, "cache"
        try:
            with call.running_port():
                reply = await self.wrapped.generate(call.request)
        except Exception as exc:
            return await self._recover(call, exc), "hook"
        call.reply = reply
        if responses is not None:
            # The provider's reply is stored, so after_llm hooks see the same input on a hit.
            await responses.set(
                call.request, reply, provider=self.provider, model=call.model, fields=fields
            )
        return reply, "model"

    async def _recover(self, call: LLMCall, exc: Exception) -> LLMResponse:
        if control_flow_member(exc) is not None or isinstance(exc, ApprovalPending):
            raise exc  # not a failure to recover from
        recovered = await run_on_error(self.hooks, "on_llm_error", call.inv, exc)
        if recovered is None:
            raise exc
        logger.info("llm call %s recovered by an on_llm_error hook", call.entry_id)
        return as_response(recovered.value, "on_llm_error")

    async def _after(self, call: LLMCall, response: LLMResponse) -> LLMResponse:
        replaced = await run_after(self.hooks, "after_llm", call.inv, response)
        return as_response(replaced, "after_llm")

    # --- stream -------------------------------------------------------------------------

    async def _stream_events(self, call: LLMCall) -> AsyncIterator[StreamEvent]:
        short = await run_before(self.hooks, "before_llm", call.inv)
        if short is not None:
            response = await self._after(call, as_response(short.value, "before_llm"))
            call.finished(response, "hook")
            for event in replay(response):
                yield event
            return
        done: LLMResponse | None = None
        started = False
        try:
            async with closing(self.wrapped.stream(call.request)) as events:
                while True:
                    try:
                        with call.running_port():  # per step: a generator never holds it
                            event = await anext(events)
                    except StopAsyncIteration:
                        break
                    if isinstance(event, StreamDone):
                        done = event.response
                        break
                    started = True
                    yield event
        except Exception as exc:
            if started:
                raise
            recovered = await self._recover(call, exc)
            response = await self._after(call, recovered)
            call.finished(response, "hook")
            for event in replay(response):
                yield event
            return
        if done is None:
            raise _no_done(call)
        call.reply = done
        response = await self._after(call, done)
        call.finished(response, "model")
        yield StreamDone(response=response)

    def _for_compression(self) -> tuple[InstrumentedModel, ToolContext]:
        """This model bound to the call's run, and that run with this model's ledger and sink."""
        ctx = resolve_context(self.ctx)
        port = self if self.ctx is not None else self.bind(ctx)
        return port, with_ports(ctx, self.ledger, self.sink)

    def _settings(self) -> Settings:
        return Settings(
            provider=self.provider,
            model=self.model,
            ledger=self.ledger,
            sink=self.sink,
            fields=self.wrapped.cache_key_fields() if self._responses is not None else {},
        )


def _no_done(call: LLMCall) -> ModelOutputInvalid:
    return ModelOutputInvalid(
        f"{call.settings.provider} {call.model}: stream ended without StreamDone",
        provider=call.settings.provider,
        model=call.model,
    )
