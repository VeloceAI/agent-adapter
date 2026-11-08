"""The trace-event catalogue: types, required data keys, and helpers to build and send events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

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
    APPROVAL_EXPIRED = "approval.expired"
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
    NODE_CACHED = "node.cached"
    CHECKPOINT_SAVED = "checkpoint.saved"
    RUN_RECOVERED = "run.recovered"
    BUDGET_WARNING = "budget.warning"
    BUDGET_RESERVED = "budget.reserved"
    BUDGET_EXCEEDED = "budget.exceeded"
    CONTEXT_COMPRESSED = "context.compressed"
    CONTEXT_COMPRESS_FAILED = "context.compress_failed"
    GUARDRAIL_REDACTED = "guardrail.redacted"
    MEMORY_ADDED = "memory.added"
    MEMORY_SEARCHED = "memory.searched"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_EXPORTED = "memory.exported"
    RETRIEVAL_SEARCHED = "retrieval.searched"
    RETRIEVAL_FAILED = "retrieval.failed"
    SANDBOX_STARTED = "sandbox.started"
    SANDBOX_FINISHED = "sandbox.finished"
    SANDBOX_FAILED = "sandbox.failed"


_LLM = ("provider", "model", "stage")
_NODE = ("flow", "node", "visit")
_EMBEDDING = ("provider", "model", "stage", "texts")  # texts: a count, never the texts
_RERANK = _EMBEDDING
_CONTEXT = ("provider", "model", "trigger", "latency_ms")  # trigger: "threshold" | "overflow"
# Counts and the scope kind (`user+session`), never ids; texts only under CONTENT_KEYS.
_MEMORY = ("backend", "count", "latency_ms", "scope_kind")
_RETRIEVAL = ("retriever", "backend", "k", "latency_ms")  # ids and counts; texts as content
_SANDBOX = ("sandbox", "backend", "language")  # sizes and exit codes; code and output as content

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
    # edited: the reviewer changed the arguments (never which values).
    EventType.APPROVAL_DECIDED: frozenset({"tool", "approval_id", "approved", "edited"}),
    # outcome "timed-out" | "stopped" (chat desk) | "ttl" (a store expired it): nobody decided.
    EventType.APPROVAL_EXPIRED: frozenset({"tool", "approval_id", "outcome"}),
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
    EventType.NODE_CACHED: frozenset(_NODE),
    # kind "flow" | "loop"; step is the stored checkpoint step, counts only (never values)
    EventType.CHECKPOINT_SAVED: frozenset({"kind", "step", "status"}),
    EventType.RUN_RECOVERED: frozenset({"kind", "step", "replayed", "rerun"}),
    EventType.BUDGET_WARNING: frozenset({"limit_usd", "spent_usd"}),
    # scope and window only: the counter key names the user or team, so it never appears.
    EventType.BUDGET_RESERVED: frozenset({"scope", "limit_usd", "spent_usd", "reserved_usd"}),
    EventType.BUDGET_EXCEEDED: frozenset({"scope", "limit_usd", "spent_usd"}),
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
    # Counts and kind labels (`email`, `card`) only, never the redacted values.
    EventType.GUARDRAIL_REDACTED: frozenset({"guard", "stage", "count", "kinds"}),
    EventType.MEMORY_ADDED: frozenset(_MEMORY),
    EventType.MEMORY_SEARCHED: frozenset({*_MEMORY, "k", "mode"}),
    EventType.MEMORY_DELETED: frozenset(_MEMORY),
    EventType.MEMORY_EXPORTED: frozenset(_MEMORY),
    EventType.RETRIEVAL_SEARCHED: frozenset({*_RETRIEVAL, "count", "ids", "dropped"}),
    EventType.RETRIEVAL_FAILED: frozenset({*_RETRIEVAL, "error_kind"}),
    EventType.SANDBOX_STARTED: frozenset(
        {*_SANDBOX, "code_bytes", "files", "network", "timeout_s"}
    ),
    EventType.SANDBOX_FINISHED: frozenset(
        {
            *_SANDBOX,
            "exit_code",
            "timed_out",
            "duration_ms",
            "stdout_bytes",
            "stderr_bytes",
            "truncated",
            "files",
        }
    ),
    EventType.SANDBOX_FAILED: frozenset({*_SANDBOX, "error_kind", "duration_ms"}),
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


@runtime_checkable
class RequiredSink(Protocol):
    """A sink a call may not run without (a fail-closed audit log).

    `require` blocks until what was emitted is stored, then raises if anything was lost;
    `settled` is True when nothing is waiting, so `require` returns at once.
    """

    def require(self) -> None: ...

    def settled(self) -> bool: ...


async def require_recorded(sink: EventSink | None) -> None:
    """Raise the first required sink's error when one under `sink` lost a record.

    The executor awaits it before a tool's side effect; a wait runs on a worker thread, so
    the event loop never blocks on the disk.
    """
    for current in members(sink):
        if isinstance(current, RequiredSink):
            if current.settled():
                current.require()
            else:
                await asyncio.to_thread(current.require)


def members(sink: EventSink | None) -> Iterator[EventSink]:
    """`sink` and every sink inside it (hubs, fan-outs, wrappers with `sinks`), each once."""
    seen: set[int] = set()
    todo = [] if sink is None else [sink]
    while todo:
        current = todo.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        todo.extend(reversed(_inner_sinks(current)))


def _inner_sinks(sink: object) -> tuple[EventSink, ...]:
    """The sinks a hub, fan-out or wrapper sends to (its `sinks`); a broken property gives none."""
    try:
        inner = getattr(sink, "sinks", ())
    except Exception as exc:
        log_failure(logger, logging.WARNING, exc, "sink %s: sinks not read", type(sink).__name__)
        return ()
    if not isinstance(inner, tuple | list):
        return ()
    return tuple(item for item in inner if item is not None)
