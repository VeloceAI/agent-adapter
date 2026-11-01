"""`context.compressed` / `context.compress_failed` events and the OTel span: counts, no text."""

import json

import pytest

from agent_adapter.context import CompressingModel, CompressionPolicy, MyOwnStep, RemoveOldMessages
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import Event
from agent_adapter.runtime.current import use_context
from agent_adapter.tracing import _genai as g
from tests.context.test_wiring_model import HISTORY, PAD, POLICY, _overflow, _request
from tests.fakes.models import ScriptedModel, text
from tests.fakes.sinks import RecordingSink


def _in_run():
    sink = RecordingSink()
    return sink, ToolContext(run_id="run_1", call_id="outer", sink=sink)


async def test_compressed_event_carries_counts_and_steps_never_text():
    sink, ctx = _in_run()
    with use_context(ctx):
        await CompressingModel(ScriptedModel([text("ok")]), POLICY).generate(_request())
    (event,) = sink.of_type("context.")
    assert event.type == "context.compressed"
    assert (event.run_id, event.parent_id) == ("run_1", "outer")
    assert event.call_id.startswith("cmp_")
    data = event.data
    assert (data["provider"], data["model"], data["trigger"]) == (
        "scripted",
        "scripted",
        "threshold",
    )
    assert data["tokens_after"] < data["tokens_before"] and data["target_tokens"] == 120
    assert data["messages_before"] == len(HISTORY) and data["steps"] == ["remove_old_messages"]
    assert data["summary_cost_usd"] == 0.0 and data["latency_ms"] >= 0
    assert PAD[:20] not in json.dumps(data, default=str)


async def test_an_overflow_retry_is_its_own_event():
    sink, ctx = _in_run()
    with use_context(ctx):
        await CompressingModel(ScriptedModel([_overflow(), text("ok")]), POLICY).generate(
            _request()
        )
    triggers = [e.data["trigger"] for e in sink.of_type("context.compressed")]
    assert triggers == ["threshold", "overflow"]


async def test_no_event_when_under_the_threshold():
    sink, ctx = _in_run()
    short = (Message(role="user", content="hi"),)
    with use_context(ctx):
        await CompressingModel(ScriptedModel([text("ok")]), POLICY).generate(
            LLMRequest(messages=short)
        )
    assert sink.of_type("context.") == []


async def test_a_failing_step_emits_compress_failed_and_raises():
    def broken(messages):
        raise ConfigError("my step broke")

    policy = CompressionPolicy(
        start_when_tokens_over=10, shrink_to_tokens=5, steps=[MyOwnStep(broken)]
    )
    sink, ctx = _in_run()
    model = ScriptedModel([text("never")])
    with use_context(ctx), pytest.raises(ConfigError, match="my step broke"):
        await model.generate(_request(compression=policy))
    (event,) = sink.of_type("context.")
    assert event.type == "context.compress_failed"
    assert event.data["error_kind"] == "config-invalid" and model.requests == []


async def test_an_unknown_window_at_call_time_is_a_config_error():
    percent = CompressionPolicy(steps=[RemoveOldMessages()])
    sink, ctx = _in_run()
    with use_context(ctx), pytest.raises(ConfigError, match="no context window"):
        await ScriptedModel([]).generate(_request(compression=percent))
    assert sink.of_type("context.")[0].data["error_kind"] == "config-invalid"


def test_span_builders_name_the_model_and_copy_only_counts():
    data = {
        "provider": "openai",
        "model": "gpt-4.1",
        "trigger": "overflow",
        "latency_ms": 5.0,
        "tokens_before": 900,
        "tokens_after": 300,
        "target_tokens": 400,
        "messages_before": 20,
        "messages_after": 6,
        "steps": ["shrink_tool_results", "summarize"],
        "summary_cost_usd": 0.002,
    }
    event = Event("context.compressed", run_id="run_1", call_id="cmp_1", data=data)
    start = g.context_start(event)
    assert start.name == "compress_context gpt-4.1"
    assert start.attrs["agent_adapter.context.trigger"] == "overflow"
    attrs, error = g.context_finish(event)
    assert error is None
    assert attrs["agent_adapter.context.tokens_after"] == 300
    assert attrs["agent_adapter.context.steps"] == "shrink_tool_results,summarize"
    assert attrs["agent_adapter.cost_usd"] == 0.002


def test_otel_sink_makes_one_internal_span_per_compression():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.trace import SpanKind, StatusCode

    from tests.tracing.otel_harness import Harness

    h = Harness()
    ids = {"provider": "openai", "model": "gpt-4.1", "trigger": "threshold"}
    counts = {
        "tokens_before": 9,
        "tokens_after": 3,
        "target_tokens": 4,
        "messages_before": 5,
        "messages_after": 2,
        "steps": ["remove_old_messages"],
        "summary_cost_usd": 0.0,
    }
    h.emit("context.compressed", ts=2.0, call_id="cmp_1", latency_ms=500.0, **ids, **counts)
    h.emit(
        "context.compress_failed", ts=3.0, call_id="cmp_2", latency_ms=1.0, error_kind="x", **ids
    )
    spans = h.exporter.get_finished_spans()
    ok, failed = spans
    assert ok.name == "compress_context gpt-4.1" and ok.kind == SpanKind.INTERNAL
    assert ok.attributes["agent_adapter.context.tokens_after"] == 3
    assert (ok.end_time - ok.start_time) == 500_000_000
    assert failed.status.status_code == StatusCode.ERROR
