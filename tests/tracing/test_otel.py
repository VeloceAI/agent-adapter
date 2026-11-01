import builtins
import json
import logging

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.trace import SpanKind, StatusCode

from agent_adapter.core.errors import MissingDependency
from agent_adapter.tracing import OTelSink
from agent_adapter.tracing import otel as otel_module
from tests.tracing.otel_harness import T0, USAGE, Harness, llm_call


@pytest.fixture
def h():
    return Harness()


def test_llm_span_name_kind_attributes_and_times(h):
    llm_call(h, response_id="resp_1", response_model="gpt-4.1-2025-04-14")
    span = h.spans()["chat gpt-4.1"]
    assert span.kind is SpanKind.CLIENT
    attrs = dict(span.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4.1"
    assert attrs["gen_ai.response.model"] == "gpt-4.1-2025-04-14"
    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.usage.input_tokens"] == 175  # billable + cached + cache writes
    assert attrs["gen_ai.usage.output_tokens"] == 40
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 50
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 25
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 10
    assert attrs["gen_ai.conversation.id"] == "run_1"
    assert attrs["gen_ai.request.stream"] is False
    assert attrs["agent_adapter.stage"] == "plan"
    assert attrs["agent_adapter.cost_usd"] == 0.01
    assert span.start_time == int(T0 * 1e9)
    assert span.end_time == int((T0 + 1.5) * 1e9)
    assert span.status.status_code is StatusCode.UNSET


def test_llm_usage_as_a_dict_and_response_model_default(h):
    llm_call(h, usage=USAGE.model_dump(), cost_usd=None)
    attrs = dict(h.spans()["chat gpt-4.1"].attributes)
    assert attrs["gen_ai.usage.input_tokens"] == 175
    assert attrs["gen_ai.response.model"] == "gpt-4.1"
    assert "agent_adapter.cost_usd" not in attrs
    assert "gen_ai.response.id" not in attrs


def test_llm_missing_usage_counts_zero(h):
    llm_call(h, usage=None)
    assert dict(h.spans()["chat gpt-4.1"].attributes)["gen_ai.usage.input_tokens"] == 0


@pytest.mark.parametrize(
    ("provider", "name", "kind", "provider_name"),
    [
        ("gemini", "generate_content gemini-2.5-pro", SpanKind.CLIENT, "gcp.gemini"),
        ("anthropic", "chat gemini-2.5-pro", SpanKind.CLIENT, "anthropic"),
        ("callable", "chat gemini-2.5-pro", SpanKind.INTERNAL, "callable"),
    ],
)
def test_llm_operation_kind_and_provider_per_provider(h, provider, name, kind, provider_name):
    llm_call(h, provider=provider, model="gemini-2.5-pro")
    span = h.spans()[name]
    assert span.kind is kind
    assert span.attributes["gen_ai.provider.name"] == provider_name


def test_llm_failed_sets_error_type_and_status(h):
    h.emit("llm.started", call_id="e1", provider="openai", model="gpt-5", stage="s", stream=True)
    h.emit(
        "llm.failed",
        ts=0.2,
        call_id="e1",
        provider="openai",
        model="gpt-5",
        stage="s",
        error_kind="model-rate-limited",
    )
    span = h.spans()["chat gpt-5"]
    assert span.attributes["error.type"] == "model-rate-limited"
    assert span.attributes["gen_ai.request.stream"] is True
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "model-rate-limited"


def test_llm_cached_ends_the_span_as_a_cache_hit(h):
    h.emit("llm.started", call_id="e1", provider="openai", model="gpt-5", stage="s", stream=False)
    h.emit("llm.cached", ts=0.01, call_id="e1", provider="openai", model="gpt-5", stage="s")
    span = h.spans()["chat gpt-5"]
    assert span.attributes["agent_adapter.cache_hit"] is True
    assert "gen_ai.client.token.usage" not in h.metrics()


def test_closing_event_without_a_start_still_makes_a_span(h):
    h.emit(
        "llm.finished",
        ts=2.0,
        call_id="e9",
        provider="openai",
        model="m",
        stage="",
        latency_ms=500.0,
        usage=USAGE,
        cost_usd=None,
        finish_reason="length",
    )
    span = h.spans()["chat m"]
    assert span.start_time == int((T0 + 1.5) * 1e9)
    assert span.end_time == int((T0 + 2.0) * 1e9)
    h.emit("llm.cached", call_id="e10", provider="openai", model="c", stage="")
    assert "chat c" in h.spans()


def test_metrics_token_usage_and_duration(h):
    llm_call(h)
    metrics = h.metrics()
    tokens = {
        dict(p.attributes)["gen_ai.token.type"]: p for p in metrics["gen_ai.client.token.usage"]
    }
    assert tokens["input"].sum == 175 and tokens["output"].sum == 40
    point = tokens["input"].attributes
    assert point["gen_ai.operation.name"] == "chat"
    assert point["gen_ai.provider.name"] == "openai"
    assert point["gen_ai.request.model"] == "gpt-4.1"
    assert point["gen_ai.response.model"] == "gpt-4.1"
    [duration] = metrics["gen_ai.client.operation.duration"]
    assert duration.sum == pytest.approx(1.5)


def test_content_is_off_by_default(h):
    h.emit("tool.started", call_id="c", tool="t", kind="", arguments={"q": "secret"})
    h.emit(
        "tool.finished",
        call_id="c",
        tool="t",
        latency_ms=1.0,
        attempts=1,
        cached=False,
        result="hidden",
    )
    attrs = dict(h.spans()["execute_tool t"].attributes)
    assert "gen_ai.tool.call.arguments" not in attrs and "gen_ai.tool.call.result" not in attrs


def test_content_with_capture_content():
    h = Harness(capture_content=True)
    h.emit("tool.started", call_id="c", tool="t", kind="", arguments={"q": "80C"})
    h.emit(
        "tool.finished",
        call_id="c",
        tool="t",
        latency_ms=1.0,
        attempts=1,
        cached=False,
        result="ok",
    )
    llm_call(h, output_messages=[{"role": "assistant", "content": "hi"}])
    tool = dict(h.spans()["execute_tool t"].attributes)
    assert json.loads(tool["gen_ai.tool.call.arguments"]) == {"q": "80C"}
    assert tool["gen_ai.tool.call.result"] == "ok"
    llm = dict(h.spans()["chat gpt-4.1"].attributes)
    assert json.loads(llm["gen_ai.output.messages"]) == [{"role": "assistant", "content": "hi"}]
    h.emit(
        "llm.started",
        call_id="x",
        provider="openai",
        model="in",
        stage="",
        stream=False,
        input_messages="raw text",
    )
    h.emit("llm.cached", call_id="x", provider="openai", model="in", stage="")
    assert h.spans()["chat in"].attributes["gen_ai.input.messages"] == "raw text"


def test_other_events_are_ignored(h):
    h.emit("run.started")
    h.emit("budget.warning", limit_usd=1.0, spent_usd=0.9)
    h.emit("custom.progress", pct=50)
    assert h.spans() == {}


def test_a_bad_event_is_logged_never_raised(h, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.otel"):
        h.emit("llm.started", call_id="e1")  # missing provider / model
    assert "OTel sink failed for llm.started" in caplog.text


def test_open_spans_are_bounded(h, monkeypatch, caplog):
    monkeypatch.setattr(otel_module, "_MAX_OPEN_SPANS", 2)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.otel"):
        for i in range(3):
            h.emit("tool.started", call_id=f"c{i}", tool=f"t{i}", kind="")
    assert list(h.spans()) == ["execute_tool t0"]
    assert "never closed" in caplog.text


def test_global_providers_are_used_by_default(monkeypatch):
    from opentelemetry import trace
    from opentelemetry.metrics import _internal as metrics_internal

    h = Harness()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: h.provider)
    monkeypatch.setattr(metrics_internal, "get_meter_provider", lambda: h.meters)
    h.sink = OTelSink()  # no providers passed: the (patched) global ones
    llm_call(h)
    assert list(h.spans()) == ["chat gpt-4.1"]
    assert "gen_ai.client.token.usage" in h.metrics()


def test_malformed_llm_finished_still_ends_the_span(h, caplog):
    h.emit("llm.started", call_id="e1", provider="openai", model="gpt-5", stage="s", stream=False)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing.otel"):
        h.emit("llm.finished", ts=1.0, call_id="e1", provider="openai", model="gpt-5", stage="s")
    assert "chat gpt-5" in h.spans()
    assert "OTel sink failed for llm.finished" in caplog.text
    assert "gen_ai.client.token.usage" not in h.metrics()


def test_llm_failed_records_duration_with_error_type(h):
    h.emit("llm.started", call_id="e1", provider="openai", model="gpt-5", stage="s", stream=False)
    h.emit(
        "llm.failed",
        ts=0.25,
        call_id="e1",
        provider="openai",
        model="gpt-5",
        stage="s",
        error_kind="model-timeout",
    )
    h.emit(
        "llm.failed",
        ts=0.5,
        call_id="e2",
        provider="openai",
        model="gpt-5",
        stage="s",
        error_kind="model-timeout",
        latency_ms=400.0,
    )
    points = h.metrics()["gen_ai.client.operation.duration"]
    [point] = points
    assert point.attributes["error.type"] == "model-timeout"
    assert point.attributes["gen_ai.request.model"] == "gpt-5"
    assert point.count == 2
    assert point.sum == pytest.approx(0.25 + 0.4)


def test_missing_opentelemetry_raises_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[otel\]"):
        OTelSink()
