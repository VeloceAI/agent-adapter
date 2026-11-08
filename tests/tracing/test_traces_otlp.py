"""OTLP recipes: the example runs offline, exporters are wired right, spans carry GenAI names."""

import asyncio
import importlib.util
import re
import runpy
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "traces_otlp.py"
PAGE = ROOT / "docs" / "how-to" / "send-traces.md"
SEMCONV = ROOT / "agent_adapter" / "tracing" / "semconv.py"
HTTP_EXPORTER = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
HAS_SDK = importlib.util.find_spec("opentelemetry") is not None and (
    importlib.util.find_spec("opentelemetry.sdk") is not None
)


@pytest.fixture(scope="module")
def example() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE))


class FakeExporter:
    """Records the keyword arguments an `OTLPSpanExporter` was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_http_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(HTTP_EXPORTER)
    module.OTLPSpanExporter = FakeExporter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, HTTP_EXPORTER, module)


def test_the_example_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    if HAS_SDK:
        assert lines == ["chat house-7b chat run_shop", "execute_tool stock execute_tool run_shop"]
    else:
        assert lines == ["opentelemetry-sdk not installed"]


def test_langfuse_exporter_uses_basic_auth_on_the_http_endpoint(example, fake_http_exporter):
    exporter = example["langfuse_exporter"]("pk-lf-1", "sk-lf-2", "https://cloud.langfuse.com/")
    assert exporter.kwargs["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    # base64("pk-lf-1:sk-lf-2")
    assert exporter.kwargs["headers"] == {"Authorization": "Basic cGstbGYtMTpzay1sZi0y"}


def test_phoenix_exporter_defaults_to_a_local_collector(example, fake_http_exporter):
    assert example["phoenix_exporter"]().kwargs == {"endpoint": "http://localhost:6006/v1/traces"}
    hosted = example["phoenix_exporter"]("https://phoenix.example.com/v1/traces")
    assert hosted.kwargs["endpoint"] == "https://phoenix.example.com/v1/traces"


def test_openllmetry_sink_initialises_traceloop_then_uses_the_global_provider(
    example, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, Any]] = []
    sdk = types.ModuleType("traceloop.sdk")
    sdk.Traceloop = types.SimpleNamespace(init=lambda **kw: calls.append(kw))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "traceloop.sdk", sdk)
    built: list[dict[str, Any]] = []
    monkeypatch.setitem(
        example["openllmetry_sink"].__globals__, "OTelSink", lambda **kw: built.append(kw)
    )
    example["openllmetry_sink"]("museum-guide")
    assert calls == [{"app_name": "museum-guide"}]
    assert built == [{}]  # no tracer_provider: the global one Traceloop installed


@pytest.mark.skipif(not HAS_SDK, reason="opentelemetry-sdk is not installed")
def test_spans_carry_the_genai_attributes_we_emit(example) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import SpanKind

    from agent_adapter.tracing import OTelSink

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    asyncio.run(example["run_once"](OTelSink(tracer_provider=provider)))
    spans = {span.name: span for span in exporter.get_finished_spans()}
    chat, tool = spans["chat house-7b"], spans["execute_tool stock"]
    assert chat.kind is SpanKind.CLIENT
    assert {
        key: chat.attributes[key]
        for key in (
            "gen_ai.operation.name",
            "gen_ai.provider.name",
            "gen_ai.request.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.conversation.id",
        )
    } == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "house",
        "gen_ai.request.model": "house-7b",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 8,
        "gen_ai.conversation.id": "run_shop",
    }
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "stock"
    assert tool.attributes["gen_ai.tool.type"] == "function"
    assert tool.attributes["gen_ai.tool.call.id"]
    for span in (chat, tool):  # content is opt-in
        assert not any(
            key.endswith((".messages", ".arguments", ".result")) for key in span.attributes
        )


@pytest.mark.skipif(not HAS_SDK, reason="opentelemetry-sdk is not installed")
def test_otlp_provider_names_the_service(example) -> None:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = example["otlp_provider"](InMemorySpanExporter(), service="ticket-desk")
    assert provider.resource.attributes["service.name"] == "ticket-desk"
    provider.shutdown()


def test_the_page_states_the_semconv_version_and_uses_every_section() -> None:
    text = PAGE.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 150
    version = re.search(
        r"opentelemetry-semantic-conventions (\S+?)\.?$", SEMCONV.read_text(encoding="utf-8"), re.M
    )
    assert version is not None
    assert f"`opentelemetry-semantic-conventions` {version.group(1)}" in text
    source = EXAMPLE.read_text(encoding="utf-8")
    sections = re.findall(r'^--8<-- "examples/traces_otlp.py:(\w+)"$', text, re.M)
    assert sections == ["provider", "wire", "langfuse", "phoenix", "openllmetry"]
    for section in sections:
        assert f"[start:{section}]" in source and f"[end:{section}]" in source
