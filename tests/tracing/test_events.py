import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.tracing import REQUIRED_DATA, EventType, emit, make_event
from tests.fakes.sinks import RecordingSink

CATALOGUE = {
    "run.started": set(),
    "run.finished": {"ok"},
    "tool.started": {"tool", "kind"},
    "tool.finished": {"tool", "latency_ms", "attempts", "cached"},
    "tool.failed": {"tool", "error_kind", "attempts", "latency_ms"},
    "tool.retry": {"tool", "attempt", "delay_s", "error_kind"},
    "tool.cached": {"tool"},
    "tool.pending": {"tool", "approval_id"},
    "approval.requested": {"tool", "approval_id"},
    "approval.decided": {"tool", "approval_id", "approved", "edited"},
    "approval.expired": {"tool", "approval_id", "outcome"},
    "guardrail.redacted": {"guard", "stage", "count", "kinds"},
    "llm.started": {"provider", "model", "stage", "stream"},
    "llm.finished": {
        "provider",
        "model",
        "stage",
        "latency_ms",
        "usage",
        "cost_usd",
        "finish_reason",
    },
    "llm.cached": {"provider", "model", "stage"},
    "llm.failed": {"provider", "model", "stage", "error_kind"},
    "llm.fallback": {"from_provider", "from_model", "to_provider", "to_model", "error_kind"},
    "embedding.started": {"provider", "model", "stage", "texts"},
    "embedding.finished": {
        "provider",
        "model",
        "stage",
        "texts",
        "cache_hits",
        "batches",
        "latency_ms",
        "usage",
        "units",
        "cost_usd",
    },
    "embedding.failed": {"provider", "model", "stage", "texts", "error_kind", "latency_ms"},
    "rerank.started": {"provider", "model", "stage", "texts"},
    "rerank.finished": {
        "provider",
        "model",
        "stage",
        "texts",
        "latency_ms",
        "usage",
        "units",
        "cost_usd",
    },
    "rerank.failed": {"provider", "model", "stage", "texts", "error_kind", "latency_ms"},
    "memory.added": {"backend", "count", "latency_ms", "scope_kind"},
    "memory.searched": {"backend", "count", "latency_ms", "k", "mode", "scope_kind"},
    "memory.deleted": {"backend", "count", "latency_ms", "scope_kind"},
    "memory.exported": {"backend", "count", "latency_ms", "scope_kind"},
    "retrieval.searched": {"retriever", "backend", "k", "latency_ms", "count", "ids", "dropped"},
    "retrieval.failed": {"retriever", "backend", "k", "latency_ms", "error_kind"},
    "sandbox.started": {
        "sandbox",
        "backend",
        "language",
        "code_bytes",
        "files",
        "network",
        "timeout_s",
    },
    "sandbox.finished": {
        "sandbox",
        "backend",
        "language",
        "exit_code",
        "timed_out",
        "duration_ms",
        "stdout_bytes",
        "stderr_bytes",
        "truncated",
        "files",
    },
    "sandbox.failed": {"sandbox", "backend", "language", "error_kind", "duration_ms"},
    "flow.started": {"flow"},
    "flow.finished": {"flow", "status", "steps"},
    "node.started": {"flow", "node", "visit"},
    "node.finished": {"flow", "node", "visit"},
    "node.failed": {"flow", "node", "visit", "error_kind"},
    "node.paused": {"flow", "node", "approval_id"},
    "node.cached": {"flow", "node", "visit"},
    "checkpoint.saved": {"kind", "step", "status"},
    "run.recovered": {"kind", "step", "replayed", "rerun"},
    "budget.warning": {"limit_usd", "spent_usd"},
    "budget.reserved": {"scope", "limit_usd", "spent_usd", "reserved_usd"},
    "budget.exceeded": {"scope", "limit_usd", "spent_usd"},
    "context.compressed": {
        "provider",
        "model",
        "trigger",
        "latency_ms",
        "tokens_before",
        "tokens_after",
        "target_tokens",
        "messages_before",
        "messages_after",
        "steps",
        "summary_cost_usd",
    },
    "context.compress_failed": {"provider", "model", "trigger", "latency_ms", "error_kind"},
}


def test_catalogue_matches_the_contract():
    assert {k: set(v) for k, v in REQUIRED_DATA.items()} == CATALOGUE
    constants = {v for k, v in vars(EventType).items() if k.isupper()}
    assert constants == set(CATALOGUE)


def test_make_event_takes_ids_from_the_context():
    ctx = ToolContext(run_id="run_1", call_id="call_1")
    event = make_event(EventType.TOOL_CACHED, ctx, tool="t")
    assert (event.type, event.run_id, event.call_id, event.parent_id) == (
        "tool.cached",
        "run_1",
        "call_1",
        "",
    )
    assert dict(event.data) == {"tool": "t"}


def test_explicit_call_id_makes_the_context_call_the_parent():
    ctx = ToolContext(run_id="r", call_id="outer")
    event = make_event(EventType.TOOL_STARTED, ctx, call_id="inner", tool="t", kind="")
    assert (event.call_id, event.parent_id) == ("inner", "outer")
    same = make_event(EventType.TOOL_CACHED, ctx, call_id="outer", tool="t")
    assert (same.call_id, same.parent_id) == ("outer", "")


def test_make_event_without_context():
    event = make_event(EventType.RUN_STARTED)
    assert (event.run_id, event.call_id, event.parent_id) == ("", "", "")
    assert make_event(EventType.RUN_FINISHED, call_id="x", ok=True).call_id == "x"


def test_custom_events_need_no_data():
    assert make_event("custom.progress", pct=5).data == {"pct": 5}


def test_unknown_types_and_missing_keys_are_rejected():
    with pytest.raises(ToolUsageError, match=r"unknown event type 'tool.exploded'"):
        make_event("tool.exploded")
    with pytest.raises(ToolUsageError, match=r"needs data keys \['attempts', 'cached'"):
        make_event(EventType.TOOL_FINISHED, tool="t")


def test_emit_sends_and_ignores_none():
    sink = RecordingSink()
    event = make_event(EventType.RUN_STARTED)
    emit(sink, event)
    emit(None, event)
    assert sink.events == [event]


class FailingSink:
    def emit(self, event):
        raise RuntimeError("sink down")


def test_emit_logs_and_swallows_sink_errors(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.events"):
        emit(FailingSink(), make_event(EventType.RUN_STARTED, ToolContext(run_id="r")))
    assert "event sink failed for run.started (run r)" in caplog.text
