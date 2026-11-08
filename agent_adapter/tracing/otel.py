"""`OTelSink`: turns adapter events into OpenTelemetry GenAI spans and metrics."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_adapter._version import __version__
from agent_adapter.core.errors import MissingDependency, log_failure
from agent_adapter.core.types import Event
from agent_adapter.tracing import _genai as g
from agent_adapter.tracing import semconv as sc
from agent_adapter.tracing.events import EventType

logger = logging.getLogger(__name__)

_MAX_OPEN_SPANS = 10_000  # spans whose closing event never arrived are ended beyond this
_LLM, _EMBED, _TOOL, _FLOW, _NODE = "llm", "embedding", "tool", "flow", "node"
_CTX = "context"


def _ns(ts: float) -> int:
    return int(ts * 1_000_000_000)


@dataclass
class _Open:
    span: Any
    start_ts: float
    metric_attrs: g.Attrs = field(default_factory=dict)
    flow: str = ""  # flow name, for flow spans
    closing: g.Attrs = field(default_factory=dict)  # attributes set at the end


class OTelSink:
    """An `EventSink` exporting GenAI-convention spans and metrics (needs opentelemetry-api)."""

    def __init__(
        self,
        *,
        tracer_provider: Any | None = None,
        meter_provider: Any | None = None,
        capture_content: bool = False,
        instrumentation_name: str = "agent_adapter",
    ) -> None:
        """Bind a tracer and meter (global providers when None).

        Raises:
            MissingDependency: opentelemetry-api is not installed (`agent-adapter[otel]`).
        """
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except ImportError as exc:
            raise MissingDependency(
                'OTelSink needs opentelemetry-api: pip install "agent-adapter[otel]"'
            ) from exc
        self._capture = capture_content
        self._trace = trace
        self._kinds = SpanKind
        self._status = Status
        self._status_error = StatusCode.ERROR
        self._tracer = trace.get_tracer(instrumentation_name, __version__, tracer_provider)
        meter = metrics.get_meter(instrumentation_name, __version__, meter_provider)
        self._tokens = meter.create_histogram(
            sc.METRIC_TOKEN_USAGE,
            unit="{token}",
            description="Tokens used per LLM or embeddings call",
        )
        self._duration = meter.create_histogram(
            sc.METRIC_OPERATION_DURATION, unit="s", description="LLM or embeddings call duration"
        )
        self._open: OrderedDict[tuple[str, str, str], _Open] = OrderedDict()
        self._lock = threading.Lock()
        self._handlers: dict[str, Callable[[Event], object]] = {
            EventType.LLM_STARTED: self._llm_started,
            EventType.LLM_FINISHED: self._llm_finished,
            EventType.LLM_CACHED: self._llm_cached,
            EventType.LLM_FAILED: self._llm_failed,
            EventType.EMBEDDING_STARTED: self._embedding_started,
            EventType.EMBEDDING_FINISHED: self._embedding_finished,
            EventType.EMBEDDING_FAILED: lambda e: self._failed(_EMBED, e, self._embedding_started),
            EventType.TOOL_STARTED: self._tool_started,
            EventType.TOOL_FINISHED: self._tool_finished,
            EventType.TOOL_FAILED: lambda e: self._close(
                _TOOL, e, self._tool_started, g.tool_failed
            ),
            EventType.TOOL_PENDING: lambda e: self._close(_TOOL, e, self._tool_started, g.approval),
            EventType.TOOL_CACHED: self._tool_note,
            EventType.TOOL_RETRY: self._tool_note,
            EventType.APPROVAL_REQUESTED: self._tool_note,
            EventType.APPROVAL_DECIDED: self._tool_note,
            EventType.APPROVAL_EXPIRED: self._tool_note,
            EventType.GUARDRAIL_REDACTED: self._redaction_note,
            EventType.FLOW_STARTED: self._flow_started,
            EventType.FLOW_FINISHED: self._flow_finished,
            EventType.NODE_STARTED: self._node_started,
            EventType.NODE_FINISHED: lambda e: self._end_node(e, lambda _: ({}, None)),
            EventType.NODE_FAILED: lambda e: self._end_node(e, g.failed),
            EventType.NODE_PAUSED: lambda e: self._end_node(e, g.approval),
            EventType.NODE_CACHED: self._node_note,
            EventType.CONTEXT_COMPRESSED: lambda e: self._close(
                _CTX, e, self._context_started, g.context_finish
            ),
            EventType.CONTEXT_COMPRESS_FAILED: lambda e: self._close(
                _CTX, e, self._context_started, g.failed
            ),
        }

    @property
    def capture_content(self) -> bool:
        """Whether opt-in content attributes (messages, arguments, results) are exported."""
        return self._capture

    def emit(self, event: Event) -> None:
        handler = self._handlers.get(event.type)
        if handler is None:
            return
        try:
            handler(event)
        except Exception as exc:
            # Telemetry must never break the traced call.
            log_failure(
                logger,
                logging.WARNING,
                exc,
                "OTel sink failed for %s (run %s)",
                event.type,
                event.run_id,
            )

    # -- llm ---------------------------------------------------------------------------

    def _llm_started(self, event: Event) -> _Open:
        start = g.llm_start(event, self._capture)
        kind = self._kinds.CLIENT if start.client else self._kinds.INTERNAL
        return self._start(_LLM, event, start, kind, self._start_ts(event))

    def _llm_finished(self, event: Event) -> None:
        opened = self._close(
            _LLM, event, self._llm_started, lambda e: g.llm_finish(e, self._capture)
        )
        attrs = opened.closing
        metric_attrs = {
            **opened.metric_attrs,
            sc.GEN_AI_RESPONSE_MODEL: attrs[sc.GEN_AI_RESPONSE_MODEL],
        }
        if event.data.get("nested"):
            # An instrumented call inside this one already reported this provider call.
            return
        for token_type, name in (
            (sc.TOKEN_TYPE_INPUT, sc.GEN_AI_USAGE_INPUT_TOKENS),
            (sc.TOKEN_TYPE_OUTPUT, sc.GEN_AI_USAGE_OUTPUT_TOKENS),
        ):
            self._tokens.record(attrs[name], {**metric_attrs, sc.GEN_AI_TOKEN_TYPE: token_type})
        self._duration.record(float(event.data["latency_ms"]) / 1000.0, metric_attrs)

    def _llm_cached(self, event: Event) -> None:
        self._close(_LLM, event, self._llm_started, lambda _: ({sc.AA_CACHE_HIT: True}, None))

    def _llm_failed(self, event: Event) -> None:
        if event.data.get("nested"):
            # A nested instrumented call already recorded this provider call.
            self._close(_LLM, event, self._llm_started, g.failed)
            return
        self._failed(_LLM, event, self._llm_started)

    def _failed(self, family: str, event: Event, start: Callable[[Event], _Open]) -> None:
        """End the span as an error and record the duration with `error.type`."""
        opened = self._close(family, event, start, g.failed)
        latency = event.data.get("latency_ms")
        if isinstance(latency, int | float):
            seconds = float(latency) / 1000.0
        else:
            seconds = max(event.ts - opened.start_ts, 0.0)
        attrs = {**opened.metric_attrs, sc.ERROR_TYPE: str(event.data["error_kind"])}
        self._duration.record(seconds, attrs)

    # -- embeddings --------------------------------------------------------------------

    def _embedding_started(self, event: Event) -> _Open:
        start = g.embedding_start(event)
        kind = self._kinds.CLIENT if start.client else self._kinds.INTERNAL
        return self._start(_EMBED, event, start, kind, self._start_ts(event))

    def _embedding_finished(self, event: Event) -> None:
        opened = self._close(_EMBED, event, self._embedding_started, g.embedding_finish)
        attrs = opened.closing
        if attrs[sc.AA_CACHE_HIT]:
            return  # served from the cache: no provider call to measure, like llm.cached
        metric_attrs = {
            **opened.metric_attrs,
            sc.GEN_AI_RESPONSE_MODEL: attrs[sc.GEN_AI_RESPONSE_MODEL],
        }
        input_tokens = attrs[sc.GEN_AI_USAGE_INPUT_TOKENS]
        if input_tokens:  # an fn may bill only in `units` (characters, requests)
            tokens = {**metric_attrs, sc.GEN_AI_TOKEN_TYPE: sc.TOKEN_TYPE_INPUT}
            self._tokens.record(input_tokens, tokens)
        self._duration.record(float(event.data["latency_ms"]) / 1000.0, metric_attrs)

    # -- tools -------------------------------------------------------------------------

    def _tool_started(self, event: Event) -> _Open:
        start = g.tool_start(event, self._capture)
        return self._start(_TOOL, event, start, self._kinds.INTERNAL, self._start_ts(event))

    def _tool_finished(self, event: Event) -> None:
        self._close(_TOOL, event, self._tool_started, lambda e: g.tool_finish(e, self._capture))

    def _tool_note(self, event: Event) -> None:
        """Retries, cache hits and approvals become span events on the open tool span."""
        with self._lock:
            opened = self._open.get((_TOOL, event.run_id, event.call_id))
        if opened is not None:
            opened.span.add_event(event.type, attributes=g.note(event), timestamp=_ns(event.ts))

    def _redaction_note(self, event: Event) -> None:
        """A redaction is a span event on its tool span, or on the model call it rewrote."""
        family = _TOOL if event.data.get("stage") == "result" else _LLM
        with self._lock:
            opened = self._open.get((family, event.run_id, event.call_id))
        if opened is not None:
            opened.span.add_event(event.type, attributes=g.note(event), timestamp=_ns(event.ts))

    # -- context compression ------------------------------------------------------------

    def _context_started(self, event: Event) -> _Open:
        """Compression sends one closing event; its span starts `latency_ms` before it."""
        start = g.context_start(event)
        return self._start(_CTX, event, start, self._kinds.INTERNAL, self._start_ts(event))

    # -- flows -------------------------------------------------------------------------

    def _flow_key(self, event: Event) -> str:
        """The flow run's call id, or its name when the runner sets no call id."""
        return event.call_id or str(event.data["flow"])

    def _flow_started(self, event: Event) -> _Open:
        start = g.flow_start(event)
        key = self._flow_key(event)
        opened = self._start(_FLOW, event, start, self._kinds.INTERNAL, event.ts, key=key)
        opened.flow = str(event.data["flow"])
        return opened

    def _flow_finished(self, event: Event) -> None:
        self._close(_FLOW, event, self._flow_started, g.flow_finish, self._flow_key(event))

    def _node_id(self, event: Event) -> str:
        data = event.data
        return event.call_id or f"{data['flow']}:{data['node']}:{data.get('visit', 0)}"

    def _node_started(self, event: Event) -> _Open:
        start = g.node_start(event)
        node_id = self._node_id(event)
        return self._start(_NODE, event, start, self._kinds.INTERNAL, event.ts, key=node_id)

    def _end_node(self, event: Event, build: Callable[[Event], g.Built]) -> None:
        self._close(_NODE, event, self._node_started, build, self._node_id(event))

    def _node_note(self, event: Event) -> None:
        """A node cache hit becomes a span event on the open node span."""
        with self._lock:
            opened = self._open.get((_NODE, event.run_id, self._node_id(event)))
        if opened is not None:
            opened.span.add_event(event.type, attributes=g.note(event), timestamp=_ns(event.ts))

    # -- span bookkeeping --------------------------------------------------------------

    def _start_ts(self, event: Event) -> float:
        """The event time, or for a closing event with no open span, its start by latency."""
        latency = event.data.get("latency_ms")
        if event.type.endswith(".started") or not isinstance(latency, int | float):
            return event.ts
        return event.ts - float(latency) / 1000.0

    def _parent(self, event: Event, family: str) -> Any | None:
        with self._lock:
            if event.parent_id:
                for fam in (_TOOL, _LLM, _EMBED, _NODE, _FLOW):
                    opened = self._open.get((fam, event.run_id, event.parent_id))
                    if opened is not None:
                        return opened.span
            if family == _NODE:
                # The innermost open flow of that name in this run.
                flow = str(event.data["flow"])
                for (fam, run_id, _), opened in reversed(self._open.items()):
                    if fam == _FLOW and run_id == event.run_id and opened.flow == flow:
                        return opened.span
        return None

    def _start(
        self,
        family: str,
        event: Event,
        start: g.SpanStart,
        kind: Any,
        start_ts: float,
        *,
        key: str | None = None,
    ) -> _Open:
        parent = self._parent(event, family)
        context = self._trace.set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span(
            start.name, context=context, kind=kind, attributes=start.attrs, start_time=_ns(start_ts)
        )
        opened = _Open(span, start_ts, dict(start.metric_attrs))
        stale: list[_Open] = []
        with self._lock:
            self._open[(family, event.run_id, event.call_id if key is None else key)] = opened
            while len(self._open) > _MAX_OPEN_SPANS:
                stale.append(self._open.popitem(last=False)[1])
        for old in stale:
            logger.warning("OTel sink ended a span that never closed (%d open)", _MAX_OPEN_SPANS)
            old.span.end()
        return opened

    def _take(self, family: str, run_id: str, key: str) -> _Open | None:
        with self._lock:
            return self._open.pop((family, run_id, key), None)

    def _close(
        self,
        family: str,
        event: Event,
        start: Callable[[Event], _Open],
        build: Callable[[Event], g.Built],
        key: str | None = None,
    ) -> _Open:
        """End the call's span with `build`'s attributes; a malformed event still ends it.

        A closing event whose start was never seen opens the span first.
        """
        key = event.call_id if key is None else key
        opened = self._take(family, event.run_id, key)
        if opened is None:
            opened = start(event)
            self._take(family, event.run_id, key)
        try:
            attrs, error = build(event)
        except Exception:
            self._end(opened, event, {}, None)
            raise
        opened.closing = attrs
        self._end(opened, event, attrs, error)
        return opened

    def _end(self, opened: _Open, event: Event, attrs: g.Attrs, error: str | None) -> None:
        span = opened.span
        if attrs:
            span.set_attributes(attrs)
        if error is not None:
            span.set_attribute(sc.ERROR_TYPE, error)
            span.set_status(self._status(self._status_error, error))
        span.end(end_time=_ns(event.ts))
