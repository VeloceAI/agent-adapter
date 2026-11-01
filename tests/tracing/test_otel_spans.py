import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.trace import SpanKind, StatusCode

from tests.tracing.otel_harness import T0, USAGE, Harness


@pytest.fixture
def h():
    return Harness()


def test_tool_span_attributes_and_parenting_under_llm(h):
    h.emit("llm.started", call_id="e1", provider="openai", model="gpt-5", stage="s", stream=False)
    h.emit(
        "tool.started", ts=0.1, call_id="call_1", parent_id="e1", tool="search", kind="retrieval"
    )
    h.emit(
        "tool.retry",
        ts=0.2,
        call_id="call_1",
        tool="search",
        attempt=2,
        delay_s=0.5,
        error_kind="tool-timeout",
    )
    h.emit(
        "tool.finished",
        ts=0.3,
        call_id="call_1",
        tool="search",
        latency_ms=200.0,
        attempts=2,
        cached=False,
    )
    h.emit(
        "llm.finished",
        ts=1.0,
        call_id="e1",
        provider="openai",
        model="gpt-5",
        stage="s",
        latency_ms=1000.0,
        usage=USAGE,
        cost_usd=0.0,
        finish_reason="tool_calls",
    )
    spans = h.spans()
    tool, llm = spans["execute_tool search"], spans["chat gpt-5"]
    assert tool.kind is SpanKind.INTERNAL
    assert tool.parent.span_id == llm.context.span_id
    assert tool.context.trace_id == llm.context.trace_id
    attrs = dict(tool.attributes)
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "search"
    assert attrs["gen_ai.tool.call.id"] == "call_1"
    assert attrs["gen_ai.tool.type"] == "function"
    assert attrs["gen_ai.conversation.id"] == "run_1"
    assert attrs["agent_adapter.tool.kind"] == "retrieval"
    assert attrs["agent_adapter.attempts"] == 2
    assert attrs["agent_adapter.cache_hit"] is False
    [retry] = tool.events
    assert retry.name == "tool.retry"
    assert dict(retry.attributes) == {
        "agent_adapter.attempt": 2,
        "agent_adapter.delay_s": 0.5,
        "agent_adapter.error_kind": "tool-timeout",
    }


def test_tool_failed_and_pending(h):
    h.emit("tool.started", call_id="a", tool="pay", kind="")
    h.emit(
        "tool.failed",
        ts=0.1,
        call_id="a",
        tool="pay",
        error_kind="tool-failed",
        attempts=3,
        latency_ms=100.0,
    )
    h.emit("tool.started", call_id="b", tool="wire", kind="")
    h.emit("approval.requested", call_id="b", tool="wire", approval_id="apr_1")
    h.emit("tool.pending", ts=0.1, call_id="b", tool="wire", approval_id="apr_1")
    spans = h.spans()
    failed, pending = spans["execute_tool pay"], spans["execute_tool wire"]
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.attributes["error.type"] == "tool-failed"
    assert failed.attributes["agent_adapter.attempts"] == 3
    assert "agent_adapter.tool.kind" not in failed.attributes
    assert pending.status.status_code is StatusCode.UNSET
    assert pending.attributes["agent_adapter.approval_id"] == "apr_1"
    assert [e.name for e in pending.events] == ["approval.requested"]


def test_tool_notes_without_an_open_span_are_ignored(h):
    h.emit("tool.cached", call_id="zz", tool="t")
    assert h.spans() == {}


def test_tool_without_call_id_has_no_call_id_attribute(h):
    h.emit("tool.started", tool="t", kind="")
    h.emit("tool.finished", tool="t", latency_ms=1.0, attempts=1, cached=True)
    assert "gen_ai.tool.call.id" not in h.spans()["execute_tool t"].attributes


def test_flow_and_node_spans_nest(h):
    h.emit("flow.started", flow="triage")
    h.emit("node.started", ts=0.1, call_id="triage:plan:0", flow="triage", node="plan", visit=0)
    h.emit("tool.started", ts=0.2, call_id="c1", parent_id="triage:plan:0", tool="search", kind="")
    h.emit(
        "tool.finished",
        ts=0.3,
        call_id="c1",
        tool="search",
        latency_ms=100.0,
        attempts=1,
        cached=False,
    )
    h.emit("node.finished", ts=0.4, call_id="triage:plan:0", flow="triage", node="plan", visit=0)
    h.emit("node.started", ts=0.5, call_id="triage:act:0", flow="triage", node="act", visit=0)
    h.emit(
        "node.failed",
        ts=0.6,
        call_id="triage:act:0",
        flow="triage",
        node="act",
        visit=0,
        error_kind="tool-error",
    )
    h.emit("flow.finished", ts=0.7, flow="triage", status="done", steps=2)
    spans = h.spans()
    flow = spans["invoke_workflow triage"]
    plan, act, tool = (
        spans["node triage.plan"],
        spans["node triage.act"],
        spans["execute_tool search"],
    )
    assert flow.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert flow.attributes["gen_ai.workflow.name"] == "triage"
    assert flow.attributes["agent_adapter.flow.status"] == "done"
    assert flow.attributes["agent_adapter.flow.steps"] == 2
    assert plan.parent.span_id == flow.context.span_id
    assert tool.parent.span_id == plan.context.span_id
    assert plan.attributes["gen_ai.workflow.name"] == "triage"
    assert plan.attributes["agent_adapter.flow.node"] == "plan"
    assert plan.attributes["agent_adapter.flow.visit"] == 0
    assert act.status.status_code is StatusCode.ERROR
    assert act.attributes["error.type"] == "tool-error"


def test_node_paused_without_visit_uses_the_call_id(h):
    h.emit("node.started", call_id="f:gate:1", flow="f", node="gate", visit=1)
    h.emit("node.paused", ts=0.1, call_id="f:gate:1", flow="f", node="gate", approval_id="apr_9")
    span = h.spans()["node f.gate"]
    assert span.attributes["agent_adapter.approval_id"] == "apr_9"
    assert span.end_time == int((T0 + 0.1) * 1e9)


def test_node_and_flow_closing_without_start(h):
    h.emit("node.finished", flow="f", node="n", visit=2)
    h.emit("flow.finished", flow="f", status="paused", steps=1)
    spans = h.spans()
    assert spans["node f.n"].attributes["agent_adapter.flow.visit"] == 2
    assert spans["invoke_workflow f"].attributes["agent_adapter.flow.status"] == "paused"


def test_unknown_parent_falls_back_to_no_parent(h):
    h.emit("tool.started", call_id="c", parent_id="nobody", tool="t", kind="")
    h.emit("tool.finished", call_id="c", tool="t", latency_ms=1.0, attempts=1, cached=False)
    assert h.spans()["execute_tool t"].parent is None


def test_same_call_id_in_two_runs_is_two_spans(h):
    for run in ("run_a", "run_b"):
        h.emit("tool.started", run_id=run, call_id="c", tool=run, kind="")
    for run in ("run_a", "run_b"):
        h.emit(
            "tool.finished",
            run_id=run,
            call_id="c",
            tool=run,
            latency_ms=1.0,
            attempts=1,
            cached=False,
        )
    spans = h.spans()
    assert spans["execute_tool run_a"].attributes["gen_ai.conversation.id"] == "run_a"
    assert spans["execute_tool run_b"].attributes["gen_ai.conversation.id"] == "run_b"


def test_no_run_id_means_no_conversation_id(h):
    h.emit("tool.started", run_id="", call_id="c", tool="t", kind="")
    h.emit(
        "tool.finished", run_id="", call_id="c", tool="t", latency_ms=1.0, attempts=1, cached=False
    )
    assert "gen_ai.conversation.id" not in h.spans()["execute_tool t"].attributes


def test_approval_span_events_carry_no_free_text(h):
    h.emit("tool.started", call_id="b", tool="wire", kind="")
    h.emit(
        "approval.decided",
        call_id="b",
        tool="wire",
        approval_id="apr_1",
        approved=False,
        reason="customer John Doe SSN 123",
        decided_by="alice@example.com",
    )
    h.emit(
        "tool.failed",
        call_id="b",
        tool="wire",
        error_kind="approval-denied",
        attempts=0,
        latency_ms=1.0,
    )
    [event] = h.spans()["execute_tool wire"].events
    assert dict(event.attributes) == {
        "agent_adapter.approval_id": "apr_1",
        "agent_adapter.approved": False,
    }


def test_nested_flows_with_the_same_name_are_keyed_by_call_id(h):
    h.emit("flow.started", call_id="f1", flow="triage")
    h.emit("flow.started", ts=0.1, call_id="f2", parent_id="f1", flow="triage")
    h.emit("node.started", ts=0.2, call_id="triage:plan:0", flow="triage", node="plan", visit=0)
    h.emit("node.finished", ts=0.3, call_id="triage:plan:0", flow="triage", node="plan", visit=0)
    h.emit(
        "flow.finished",
        ts=0.4,
        call_id="f2",
        flow="triage",
        status="failed",
        steps=1,
        error_kind="tool-error",
    )
    h.emit("flow.finished", ts=0.5, call_id="f1", flow="triage", status="done", steps=2)
    spans = h.exporter.get_finished_spans()
    node = next(s for s in spans if s.name == "node triage.plan")
    inner, outer = (s for s in spans if s.name == "invoke_workflow triage")
    assert inner.parent.span_id == outer.context.span_id
    assert node.parent.span_id == inner.context.span_id  # innermost flow of that name
    assert inner.status.status_code is StatusCode.ERROR
    assert inner.attributes["error.type"] == "tool-error"
    assert outer.status.status_code is StatusCode.UNSET
    assert outer.attributes["agent_adapter.flow.steps"] == 2


def test_failed_flow_without_error_kind_uses_the_status(h):
    h.emit("flow.started", flow="f")
    h.emit("flow.finished", flow="f", status="error", steps=0)
    span = h.spans()["invoke_workflow f"]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "error"


def test_malformed_closing_event_still_ends_the_span(h, caplog):
    h.emit("tool.started", call_id="c", tool="t", kind="")
    with caplog.at_level("WARNING", logger="agent_adapter.tracing.otel"):
        h.emit("tool.finished", ts=0.2, call_id="c", tool="t", latency_ms=1.0, cached=False)
    span = h.spans()["execute_tool t"]
    assert span.end_time == int((T0 + 0.2) * 1e9)
    assert "OTel sink failed for tool.finished" in caplog.text
    assert h.sink._open == {}
