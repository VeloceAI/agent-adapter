"""One instrumented model call: its context, ledger entry id, ledger entry and events."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import control_flow_member
from agent_adapter.core.errors import ApprovalPending, ToolUsageError, log_failure
from agent_adapter.core.messages import LLMRequest, LLMResponse
from agent_adapter.core.ports import EventSink, LedgerPort
from agent_adapter.core.types import ErrorInfo, LedgerEntry, Usage
from agent_adapter.hooks.base import Hook, LLMInvocation
from agent_adapter.models.entries import next_entry_id
from agent_adapter.runtime.current import current_context, resolve_context, use_context
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import first_sink

logger = logging.getLogger(__name__)

CANCELLED = "cancelled"  # llm.failed error_kind for control flow and closed streams

Source = Literal["model", "cache", "hook"]

# The instrumented call whose port is running now (InstrumentedModels nested in a port).
_ENCLOSING: ContextVar[LLMCall | None] = ContextVar("agent_adapter_llm_call", default=None)


@dataclasses.dataclass(frozen=True)
class Settings:
    """What a call needs from its InstrumentedModel."""

    provider: str
    model: str
    ledger: LedgerPort | None
    sink: EventSink | None
    fields: Mapping[str, Any]  # the port's cache key fields (empty without a response cache)


@dataclasses.dataclass(frozen=True)
class Inner:
    """A response a nested instrumented call returned, with where it was billed and reported."""

    response: LLMResponse
    ledger: LedgerPort | None
    sink: EventSink | None


def same_reply(outer: LLMResponse, inner: LLMResponse, rule: int) -> bool:
    """Rule 0: the same object; 1: the same provider response id; 2: same model and usage."""
    if rule == 0:
        return outer is inner
    if rule == 1:
        return bool(outer.id) and outer.id == inner.id
    if outer.id and inner.id:  # two different provider ids are two different calls
        return False
    return outer.model == inner.model and outer.usage == inner.usage


class LLMCall:
    """State of one generate or stream call through the instrumented pipeline."""

    def __init__(
        self, settings: Settings, request: LLMRequest, bound: ToolContext | None, *, stream: bool
    ) -> None:
        self._current = current_context()  # tells whether we run directly in an outer port
        self.settings = settings
        self.request = request
        self.stream = stream
        ctx = resolve_context(bound)
        self.parent_id = ctx.call_id
        self.entry_id = next_entry_id(ctx, request)
        self.ctx = with_ports(ctx, settings.ledger, settings.sink)
        self.stage = request.stage or "llm"
        self.model = request.model or settings.model
        self.inv = LLMInvocation(request, settings.provider, self.model, self.ctx)
        self.reply: LLMResponse | None = None  # what the provider returned, for billing
        # Responses of InstrumentedModels that ran inside this call's port.
        self.inner: list[Inner] = []
        # Errors nested calls reported as llm.failed, with the sink they reported to.
        self.reported_failures: list[tuple[EventSink | None, BaseException]] = []
        self._enclosing = _ENCLOSING.get()
        self._port_ctx = self.ctx.child(call_id=self.entry_id)  # nested calls are our children
        self._settled = False
        self._event_ctx = ToolContext(run_id=self.ctx.run_id, call_id=self.parent_id)
        self._started = time.perf_counter()
        self._started_at = time.time()
        self._port_s = 0.0  # time inside the wrapped port: what per-second prices bill

    @contextmanager
    def running_port(self) -> Iterator[None]:
        """Run (one step of) the wrapped port in this call's context."""
        token = _ENCLOSING.set(self)
        started = time.perf_counter()
        try:
            with use_context(self._port_ctx):
                yield
        finally:
            self._port_s += time.perf_counter() - started
            _ENCLOSING.reset(token)

    # --- events -------------------------------------------------------------------------

    def emit(self, type: str, /, **data: Any) -> None:
        event = make_event(type, self._event_ctx, call_id=self.entry_id, **data)
        emit(self.ctx.sink, event)

    def _ids(self) -> dict[str, Any]:
        return {"provider": self.settings.provider, "model": self.model, "stage": self.stage}

    def started(self) -> None:
        self.emit(EventType.LLM_STARTED, **self._ids(), stream=self.stream)

    # --- outcomes -----------------------------------------------------------------------

    def finished(self, response: LLMResponse, source: Source) -> None:
        """Record the entry and emit `llm.finished` (or `llm.cached` for a cache hit)."""
        self._settled = True
        billed = self.reply if source == "model" and self.reply is not None else response
        inside = self._inside()
        billed_inside = inside is not None and inside.ledger is self.ctx.ledger
        nested = inside is not None and inside.sink is not None and inside.sink is self.ctx.sink
        entry = self._entry(
            billed.usage,
            provider=billed.provider or self.settings.provider,
            model=billed.model or self.model,
            cache_hit=source == "cache",
            cost_usd=None if source == "model" else 0.0,  # cache hits and hooks cost nothing
            source=source,
        )
        if billed_inside:
            # e.g. executor.model(FallbackModel([executor.model(a), ...])): the hop holds the cost.
            logger.debug("llm call %s: billed by the instrumented call inside", self.entry_id)
        else:
            self._record(entry)
        self._register(response)
        if source == "cache":
            self.emit(EventType.LLM_CACHED, **self._ids())
            return
        data: dict[str, Any] = {"nested": True} if nested else {}
        self.emit(
            EventType.LLM_FINISHED,
            **self._ids(),
            latency_ms=entry.latency_ms,
            usage=billed.usage.model_dump(mode="json"),
            cost_usd=None if nested else entry.cost_usd,
            finish_reason=response.finish_reason,
            response_model=response.model,
            response_id=response.id,
            **data,
        )

    def failed(self, exc: BaseException) -> None:
        """Record a failed entry (with the reply's usage, if any) and emit `llm.failed`.

        Control flow, a closed stream ("cancelled") and `ApprovalPending` close the span and
        record an entry only when a paid reply had already arrived.
        """
        if self._settled:
            return
        self._settled = True
        outer = self._direct_outer()
        if outer is not None:
            outer.reported_failures.append((self.ctx.sink, exc))
        reply = self._paid_reply()
        if not isinstance(exc, Exception) or control_flow_member(exc) is not None:
            self._interrupted(CANCELLED, reply, exc)
            return
        info = ErrorInfo.from_exception(exc)
        if isinstance(exc, ApprovalPending):
            self._interrupted(info.kind, reply, exc)
            return
        logger.warning(
            "llm call %s (%s %s, stage %s, run %s) failed: %s",
            self.entry_id,
            self.settings.provider,
            self.model,
            self.stage,
            self.ctx.run_id,
            info.kind,
        )
        entry = self._entry(
            reply.usage if reply is not None else Usage(),
            provider=(reply.provider if reply is not None else "") or self.settings.provider,
            model=(reply.model if reply is not None else "") or self.model,
            cache_hit=False,
            cost_usd=None if reply is not None else 0.0,  # no reply: nothing was billed
            ok=False,
            error_kind=info.kind,
        )
        self._record(entry)
        self.emit(
            EventType.LLM_FAILED,
            **self._ids(),
            error_kind=info.kind,
            latency_ms=entry.latency_ms,
            **self._nested_failure(exc),
        )

    def _interrupted(self, kind: str, reply: LLMResponse | None, exc: BaseException) -> None:
        if reply is not None:
            logger.info("llm call %s interrupted (%s) after a paid reply", self.entry_id, kind)
            self._record(
                self._entry(
                    reply.usage,
                    provider=reply.provider or self.settings.provider,
                    model=reply.model or self.model,
                    cache_hit=False,
                    cost_usd=None,
                    ok=False,
                    error_kind=kind,
                )
            )
        self.emit(EventType.LLM_FAILED, **self._ids(), error_kind=kind, **self._nested_failure(exc))

    def _nested_failure(self, exc: BaseException) -> dict[str, Any]:
        """`{"nested": True}` when `exc` is a nested call's failure already reported to our sink."""
        sink = self.ctx.sink
        reported = sink is not None and any(
            other is sink and error is exc for other, error in self.reported_failures
        )
        return {"nested": True} if reported else {}

    def _paid_reply(self) -> LLMResponse | None:
        """The provider reply this call must bill: none, or one a nested call billed."""
        inside = self._inside()
        if inside is not None and inside.ledger is self.ctx.ledger:
            return None
        return self.reply

    def _entry(
        self,
        usage: Usage,
        *,
        provider: str,
        model: str,
        cache_hit: bool,
        cost_usd: float | None,
        ok: bool = True,
        error_kind: str = "",
        source: Source = "model",
    ) -> LedgerEntry:
        ended_at = time.time()
        metadata = {"parent_call_id": self.parent_id} if self.parent_id else {}
        if source == "hook":
            metadata["source"] = "hook"
        entry = LedgerEntry.from_usage(
            usage,
            kind="llm",
            run_id=self.ctx.run_id,
            call_id=self.entry_id,
            entry_id=self.entry_id,
            stage=self.stage,
            provider=provider,
            model=model,
            cost_usd=cost_usd,
            latency_ms=(time.perf_counter() - self._started) * 1000,
            started_at=self._started_at,
            ended_at=ended_at,
            ok=ok,
            error_kind=error_kind,
            cache_hit=cache_hit,
            thread_id=self.ctx.thread_id,
            metadata=metadata,
        )
        return self._priced(entry, usage)

    def _priced(self, entry: LedgerEntry, usage: Usage) -> LedgerEntry:
        """`entry` priced here, where the port's own time is known; as is without a price."""
        cost = self._cost(entry, usage)
        if cost is None or entry.cost_usd is not None:
            return entry
        version = getattr(getattr(self.ctx.ledger, "prices", None), "version", "")
        update = {"cost_usd": cost, "price_version": version if isinstance(version, str) else ""}
        return entry.model_copy(update=update)

    def _record(self, entry: LedgerEntry) -> None:
        ledger = self.ctx.ledger
        if ledger is None:
            return
        try:
            ledger.record(entry)
        except Exception as exc:
            # The port must not raise; a broken one must not fail the call it measures.
            log_failure(
                logger, logging.WARNING, exc, "ledger.record failed for llm call %s", self.entry_id
            )

    def _register(self, response: LLMResponse) -> None:
        """Tell the enclosing call what this call returned, and where it billed and reported it."""
        outer = self._direct_outer()
        if outer is not None:
            outer.inner.append(Inner(response, self.ctx.ledger, self.ctx.sink))

    def _direct_outer(self) -> LLMCall | None:
        """The enclosing call when we run directly in its port (bound or not), else None."""
        # A tool the port ran sets its own child context: calls inside it are not ours.
        outer = self._enclosing
        if outer is not None and self._current is outer._port_ctx:
            return outer
        return None

    def _inside(self) -> Inner | None:
        """The nested call whose response is our reply (first rule that matches), or None."""
        # Not "any nested call": a failed hop followed by a raw port is still ours to bill.
        reply = self.reply
        if reply is None:
            return None
        for rule in range(3):
            for inner in self.inner:
                if same_reply(reply, inner.response, rule):
                    self.inner.remove(inner)  # each inner response matches at most once
                    return inner
        return None

    def _cost(self, entry: LedgerEntry, usage: Usage) -> float | None:
        """The entry's USD cost from the ledger's prices; None without a pricing ledger.

        Per-second prices bill the time spent inside the wrapped port only: not hooks, the
        response cache, or a stream consumer's time between chunks.
        """
        if entry.cost_usd is not None:
            return entry.cost_usd
        prices = getattr(self.ctx.ledger, "prices", None)
        cost = getattr(prices, "cost", None)
        if cost is None:
            return None
        day = datetime.fromtimestamp(entry.started_at, UTC).date()
        try:
            value = cost(  # the same arguments `Ledger` prices the entry with
                entry.model,
                usage,
                at=day,
                seconds=self._port_s,
                attempts=max(1, entry.attempts),
            )
        except Exception as exc:
            log_failure(
                logger, logging.WARNING, exc, "pricing failed for llm call %s", self.entry_id
            )
            return None
        return value if isinstance(value, int | float) else None


def as_response(value: Any, phase: str) -> LLMResponse:
    """`value` when it is an LLMResponse.

    Raises:
        ToolUsageError: a hook gave something else.
    """
    if not isinstance(value, LLMResponse):
        raise ToolUsageError(f"{phase} hooks must give an LLMResponse, got {type(value).__name__}")
    return value


def check_hooks(hooks: Sequence[Any]) -> tuple[Hook, ...]:
    """The hooks as a tuple.

    Raises:
        ToolUsageError: an item is not a Hook.
    """
    bad = [hook for hook in hooks if not isinstance(hook, Hook)]
    if bad:
        raise ToolUsageError(f"hooks must be Hook instances, got {bad!r}")
    return tuple(hooks)


def with_ports(ctx: ToolContext, ledger: LedgerPort | None, sink: EventSink | None) -> ToolContext:
    """`ctx` carrying the model's ledger and sink, or its own where the model has none."""
    ledger = ledger if ledger is not None else ctx.ledger
    sink = first_sink(sink, ctx.sink)  # an empty hub falls through to the call's sink
    if ledger is ctx.ledger and sink is ctx.sink:
        return ctx
    return dataclasses.replace(ctx, ledger=ledger, sink=sink)
