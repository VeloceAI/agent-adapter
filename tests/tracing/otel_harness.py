"""In-memory OTel SDK harness for the OTelSink tests (needs opentelemetry-sdk)."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent_adapter.core.types import Event, Usage
from agent_adapter.tracing import OTelSink

T0 = 1_800_000_000.0
USAGE = Usage(
    input_tokens=100,
    cached_input_tokens=50,
    cache_write_input_tokens=25,
    output_tokens=40,
    reasoning_tokens=10,
)


class Harness:
    def __init__(self, **kwargs):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.reader = InMemoryMetricReader()
        self.meters = MeterProvider(metric_readers=[self.reader])
        self.sink = OTelSink(tracer_provider=self.provider, meter_provider=self.meters, **kwargs)

    def emit(self, type, *, ts=0.0, run_id="run_1", call_id="", parent_id="", **data):
        self.sink.emit(
            Event(type, run_id=run_id, call_id=call_id, parent_id=parent_id, ts=T0 + ts, data=data)
        )

    def spans(self):
        return {span.name: span for span in self.exporter.get_finished_spans()}

    def metrics(self):
        data = self.reader.get_metrics_data()
        found = {}
        for resource in data.resource_metrics if data is not None else ():
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    found[metric.name] = list(metric.data.data_points)
        return found


def llm_call(h, provider="openai", model="gpt-4.1", call_id="e1", parent_id="", **finished):
    h.emit(
        "llm.started",
        call_id=call_id,
        parent_id=parent_id,
        provider=provider,
        model=model,
        stage="plan",
        stream=False,
    )
    data = dict(
        provider=provider,
        model=model,
        stage="plan",
        latency_ms=1500.0,
        usage=USAGE,
        cost_usd=0.01,
        finish_reason="stop",
    )
    data.update(finished)
    h.emit("llm.finished", ts=1.5, call_id=call_id, parent_id=parent_id, **data)
