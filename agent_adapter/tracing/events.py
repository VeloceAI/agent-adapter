"""The trace-event catalogue: types, required data keys, and helpers to build and send events."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.context import CUSTOM_PREFIX, ToolContext
from agent_adapter.core.errors import ToolUsageError, log_failure
from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event

logger = logging.getLogger(__name__)


class EventType:
    """Event type strings; anything else must start with `custom.`."""

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    TOOL_FAILED = "tool.failed"
    TOOL_RETRY = "tool.retry"
    TOOL_CACHED = "tool.cached"
    TOOL_PENDING = "tool.pending"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    LLM_STARTED = "llm.started"
    LLM_FINISHED = "llm.finished"
    LLM_CACHED = "llm.cached"
    LLM_FAILED = "llm.failed"
    LLM_FALLBACK = "llm.fallback"
    EMBEDDING_STARTED = "embedding.started"
    EMBEDDING_FINISHED = "embedding.finished"
    EMBEDDING_FAILED = "embedding.failed"
    RERANK_STARTED = "rerank.started"
    RERANK_FINISHED = "rerank.finished"
    RERANK_FAILED = "rerank.failed"
    FLOW_STARTED = "flow.started"
    FLOW_FINISHED = "flow.finished"
    NODE_STARTED = "node.started"
    NODE_FINISHED = "node.finished"
    NODE_FAILED = "node.failed"
    NODE_PAUSED = "node.paused"
    BUDGET_WARNING = "budget.warning"
    CONTEXT_COMPRESSED = "context.compressed"
    CONTEXT_COMPRESS_FAILED = "context.compress_failed"


_LLM = ("provider", "model", "stage")
_NODE = ("flow", "node", "visit")
_EMBEDDING = ("provider", "model", "stage", "texts")  # texts: a count, never the texts
_RERANK = _EMBEDDING
_CONTEXT = ("provider", "model", "trigger", "latency_ms")  # trigger: "threshold" | "overflow"

REQUIRED_DATA: Mapping[str, frozenset[str]] = {
    EventType.RUN_STARTED: frozenset(),
    EventType.RUN_FINISHED: frozenset({"ok"}),
    EventType.TOOL_STARTED: frozenset({"tool", "kind"}),  # kind "" when the tool is unknown
    EventType.TOOL_FINISHED: frozenset({"tool", "latency_ms", "attempts", "cached"}),
    EventType.TOOL_FAILED: frozenset({"tool", "error_kind", "attempts", "latency_ms"}),
    EventType.TOOL_RETRY: frozenset({"tool", "attempt", "delay_s", "error_kind"}),
    EventType.TOOL_CACHED: frozenset({"tool"}),
    EventType.TOOL_PENDING: frozenset({"tool", "approval_id"}),
    EventType.APPROVAL_REQUESTED: frozenset({"tool", "approval_id"}),
    EventType.APPROVAL_DECIDED: frozenset({"tool", "approval_id", "approved"}),
    EventType.LLM_STARTED: frozenset({*_LLM, "stream"}),
    EventType.LLM_FINISHED: frozenset({*_LLM, "latency_ms", "usage", "cost_usd", "finish_reason"}),
    EventType.LLM_CACHED: frozenset(_LLM),
    EventType.LLM_FAILED: frozenset({*_LLM, "error_kind"}),
    EventType.LLM_FALLBACK: frozenset(
        {"from_provider", "from_model", "to_provider", "to_model", "error_kind"}
    ),
    EventType.EMBEDDING_STARTED: frozenset(_EMBEDDING),
    EventType.EMBEDDING_FINISHED: frozenset(
        {*_EMBEDDING, "cache_hits", "batches", "latency_ms", "usage", "units", "cost_usd"}
    ),
    EventType.EMBEDDING_FAILED: frozenset({*_EMBEDDING, "error_kind", "latency_ms"}),
    EventType.RERANK_STARTED: frozenset(_RERANK),
    EventType.RERANK_FINISHED: frozenset({*_RERANK, "latency_ms", "usage", "units", "cost_usd"}),
    EventType.RERANK_FAILED: frozenset({*_RERANK, "error_kind", "latency_ms"}),
    EventType.FLOW_STARTED: frozenset({"flow"}),
    EventType.FLOW_FINISHED: frozenset({"flow", "status", "steps"}),
    EventType.NODE_STARTED: frozenset(_NODE),
    EventType.NODE_FINISHED: frozenset(_NODE),
    EventType.NODE_FAILED: frozenset({*_NODE, "error_kind"}),
    EventType.NODE_PAUSED: frozenset({"flow", "node", "approval_id"}),
    EventType.BUDGET_WARNING: frozenset({"limit_usd", "spent_usd"}),
    # Counts and step names only, never message text.
    EventType.CONTEXT_COMPRESSED: frozenset(
        {
            *_CONTEXT,
            "tokens_before",
            "tokens_after",
            "target_tokens",
            "messages_before",
            "messages_after",
            "steps",
            "summary_cost_usd",
        }
    ),
    EventType.CONTEXT_COMPRESS_FAILED: frozenset({*_CONTEXT, "error_kind"}),
}


def make_event(
    type: str, ctx: ToolContext | None = None, *, call_id: str | None = None, **data: Any
) -> Event:
    """An event with ids from `ctx`; an explicit `call_id` makes `ctx.call_id` the parent.

    Raises:
        ToolUsageError: unknown type (not in the catalogue, not `custom.*`) or missing data keys.
    """
    required = REQUIRED_DATA.get(type)
    if required is None and not type.startswith(CUSTOM_PREFIX):
        raise ToolUsageError(f"unknown event type {type!r}; use the catalogue or 'custom.<name>'")
    missing = sorted((required or frozenset()) - set(data))
    if missing:
        raise ToolUsageError(f"event {type!r} needs data keys {missing}")
    run_id = ctx.run_id if ctx is not None else ""
    enclosing = ctx.call_id if ctx is not None else ""
    if call_id is None:
        return Event(type, run_id=run_id, call_id=enclosing, data=data)
    parent = enclosing if enclosing != call_id else ""
    return Event(type, run_id=run_id, call_id=call_id, parent_id=parent, data=data)


def emit(sink: EventSink | None, event: Event) -> None:
    """Send `event` to `sink`; a no-op for None. Sink failures are logged, never raised."""
    if sink is None:
        return
    try:
        sink.emit(event)
    except Exception as exc:
        # Tracing must never break the traced call.
        log_failure(
            logger,
            logging.WARNING,
            exc,
            "event sink failed for %s (run %s)",
            event.type,
            event.run_id,
        )
