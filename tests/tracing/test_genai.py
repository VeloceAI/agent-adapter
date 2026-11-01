import json

import pytest

from agent_adapter.core.types import Event, Usage
from agent_adapter.tracing import _genai as g

USAGE = Usage(
    input_tokens=100,
    cached_input_tokens=50,
    cache_write_input_tokens=25,
    output_tokens=40,
    reasoning_tokens=10,
)


def ev(type, run_id="run_1", call_id="c1", **data):
    return Event(type, run_id=run_id, call_id=call_id, data=data)


def finished(**extra):
    data = dict(
        provider="openai",
        model="gpt-4.1",
        stage="plan",
        latency_ms=10.0,
        usage=USAGE,
        cost_usd=0.5,
        finish_reason="stop",
    )
    data.update(extra)
    return ev("llm.finished", **data)


def test_llm_start_names_kinds_and_metric_attributes():
    start = g.llm_start(
        ev("llm.started", provider="gemini", model="m", stage="s", stream=True), False
    )
    assert start.name == "generate_content m" and start.client is True
    assert start.metric_attrs == {
        "gen_ai.operation.name": "generate_content",
        "gen_ai.provider.name": "gcp.gemini",
        "gen_ai.request.model": "m",
    }
    assert start.attrs["gen_ai.request.stream"] is True
    assert start.attrs["gen_ai.conversation.id"] == "run_1"
    callable_start = g.llm_start(ev("llm.started", provider="callable", model="m"), False)
    assert callable_start.client is False and callable_start.name == "chat m"


def test_llm_finish_usage_counts_all_prompt_tokens():
    attrs, error = g.llm_finish(finished(), False)
    assert error is None
    assert attrs["gen_ai.usage.input_tokens"] == 175
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 50
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 25
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 10
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    assert attrs["agent_adapter.cost_usd"] == 0.5


@pytest.mark.parametrize("usage", [USAGE.model_dump(), None, "junk"])
def test_llm_finish_accepts_dicts_and_missing_usage(usage):
    attrs, _ = g.llm_finish(finished(usage=usage, cost_usd=None), False)
    assert attrs["gen_ai.usage.input_tokens"] in (175, 0)
    assert "agent_adapter.cost_usd" not in attrs


def test_llm_finish_rejects_bad_events():
    with pytest.raises(KeyError):
        g.llm_finish(ev("llm.finished", provider="p", model="m", latency_ms=1.0), False)
    with pytest.raises(ValueError):
        g.llm_finish(finished(latency_ms="slow"), False)


def test_content_only_when_captured():
    data = {"output_messages": [{"role": "assistant", "content": "hi"}]}
    assert "gen_ai.output.messages" not in g.llm_finish(finished(**data), False)[0]
    attrs, _ = g.llm_finish(finished(**data), True)
    assert json.loads(attrs["gen_ai.output.messages"]) == data["output_messages"]
    tool = g.tool_start(ev("tool.started", tool="t", kind="", arguments="raw"), True)
    assert tool.attrs["gen_ai.tool.call.arguments"] == "raw"


def test_tool_start_and_finish():
    start = g.tool_start(ev("tool.started", call_id="", tool="search", kind="retrieval"), False)
    assert start.name == "execute_tool search"
    assert "gen_ai.tool.call.id" not in start.attrs
    assert start.attrs["agent_adapter.tool.kind"] == "retrieval"
    attrs, error = g.tool_finish(ev("tool.finished", attempts=2, cached=True, result="r"), False)
    assert attrs == {"agent_adapter.attempts": 2, "agent_adapter.cache_hit": True}
    assert error is None
    assert g.tool_failed(ev("tool.failed", attempts=3, error_kind="tool-failed")) == (
        {"agent_adapter.attempts": 3},
        "tool-failed",
    )


def test_note_keeps_only_allowed_scalar_keys():
    event = ev(
        "approval.decided",
        tool="wire",
        approval_id="apr_1",
        approved=True,
        reason="customer John Doe SSN 123",
        decided_by="alice",
        arguments={"amount": 1},
    )
    assert g.note(event) == {"agent_adapter.approval_id": "apr_1", "agent_adapter.approved": True}
    retry = g.note(ev("tool.retry", tool="t", attempt=2, delay_s=0.5, error_kind="x"))
    assert set(retry) == {
        "agent_adapter.attempt",
        "agent_adapter.delay_s",
        "agent_adapter.error_kind",
    }


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"status": "done", "steps": 2}, None),
        ({"status": "paused", "steps": 1}, None),
        ({"status": "failed", "steps": 1, "error_kind": "tool-error"}, "tool-error"),
        ({"status": "error", "steps": 0}, "error"),
        ({"status": "cancelled", "steps": 1}, "cancelled"),
    ],
)
def test_flow_finish_errors_only_on_failure(data, error):
    attrs, got = g.flow_finish(ev("flow.finished", flow="f", **data))
    assert got == error
    assert attrs["agent_adapter.flow.status"] == data["status"]


def test_flow_and_node_start():
    flow = g.flow_start(ev("flow.started", run_id="", flow="triage"))
    assert flow.name == "invoke_workflow triage" and "gen_ai.conversation.id" not in flow.attrs
    node = g.node_start(ev("node.started", flow="triage", node="plan", visit=1))
    assert node.name == "node triage.plan"
    assert node.attrs["agent_adapter.flow.visit"] == 1
    assert (
        "agent_adapter.flow.visit" not in g.node_start(ev("node.paused", flow="f", node="n")).attrs
    )


def test_approval_and_failed_builders():
    assert g.approval(ev("tool.pending", approval_id="apr_2")) == (
        {"agent_adapter.approval_id": "apr_2"},
        None,
    )
    assert g.failed(ev("node.failed", error_kind="k")) == ({}, "k")
