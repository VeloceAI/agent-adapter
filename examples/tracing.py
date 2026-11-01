"""Send trace events to several sinks; export them as OpenTelemetry spans when installed."""

import logging

from agent_adapter import ToolContext
from agent_adapter.tracing import (
    CallbackEventSink,
    EventLog,
    FanoutSink,
    LoggingSink,
    OTelSink,
    make_event,
)


def sinks_demo() -> EventLog:
    # --8<-- [start:sinks]
    log = EventLog(max_events=1_000)  # bounded, in memory
    seen: list[str] = []
    sink = FanoutSink(
        log,
        LoggingSink(logging.getLogger("app.trace")),  # DEBUG by default
        CallbackEventSink(lambda event: seen.append(event.type)),
    )
    ctx = ToolContext(run_id="run_1", call_id="call_1", sink=sink)
    sink.emit(make_event("tool.started", ctx, tool="search", kind="retrieval"))
    ctx.emit("progress", pct=50)  # custom.progress
    sink.emit(
        make_event("tool.finished", ctx, tool="search", latency_ms=12.0, attempts=1, cached=False)
    )
    print([event.type for event in log.of_type("tool.")])
    print(seen)
    # --8<-- [end:sinks]
    return log


def otel_demo(log: EventLog) -> None:
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError:
        print("opentelemetry-sdk not installed")
        return
    # --8<-- [start:otel]
    exporter = InMemorySpanExporter()  # in production: an OTLP exporter
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OTelSink(tracer_provider=provider)  # content attributes stay off
    for event in log.events:  # or FanoutSink(otel, ...) as the executor's sink
        otel.emit(event)
    span = exporter.get_finished_spans()[0]
    print(span.name, span.attributes["gen_ai.tool.name"], span.attributes["gen_ai.conversation.id"])
    # --8<-- [end:otel]


def main() -> None:
    otel_demo(sinks_demo())


if __name__ == "__main__":
    main()
