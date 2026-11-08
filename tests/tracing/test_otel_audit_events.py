"""Expired approvals and redactions become span events with safe attributes only."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from tests.tracing.otel_harness import Harness


def test_expiry_and_redaction_are_span_events_without_values() -> None:
    h = Harness()
    h.emit("tool.started", call_id="c", tool="send", kind="")
    h.emit("approval.expired", call_id="c", tool="send", approval_id="apr_1", outcome="timed-out")
    h.emit(
        "guardrail.redacted",
        call_id="c",
        guard="pii_guard",
        stage="result",
        count=2,
        kinds=["email"],
        tool="send",
    )
    h.emit("tool.finished", call_id="c", tool="send", latency_ms=1.0, attempts=1, cached=False)
    expired, redacted = h.spans()["execute_tool send"].events
    assert dict(expired.attributes) == {
        "agent_adapter.approval_id": "apr_1",
        "agent_adapter.outcome": "timed-out",
    }
    assert dict(redacted.attributes) == {
        "agent_adapter.guard": "pii_guard",
        "agent_adapter.stage": "result",
        "agent_adapter.count": 2,
    }


def test_a_reply_redaction_lands_on_the_model_span() -> None:
    h = Harness()
    h.emit("llm.started", call_id="m", provider="p", model="x", stage="s", stream=False)
    h.emit("guardrail.redacted", call_id="m", guard="g", stage="llm_response", count=1, kinds=[])
    h.emit(
        "llm.finished",
        call_id="m",
        provider="p",
        model="x",
        stage="s",
        latency_ms=1.0,
        usage={},
        cost_usd=0.0,
        finish_reason="stop",
    )
    (span,) = [s for name, s in h.spans().items() if name.startswith("chat")]
    assert [event.name for event in span.events] == ["guardrail.redacted"]
