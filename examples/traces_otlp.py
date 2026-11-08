"""Send the adapter's OpenTelemetry spans to Langfuse, Arize Phoenix or OpenLLMetry over OTLP."""

import asyncio
import base64
from typing import Any

from agent_adapter import CallableModel, Executor, LLMRequest, Message, Registry, Usage
from agent_adapter.tracing import OTelSink

registry = Registry()


@registry.tool
def stock(item: str) -> int:
    """Units of an item left in the shop."""
    return {"mug": 12, "poster": 3}.get(item, 0)


async def answer(request: LLMRequest) -> dict[str, Any]:
    """A fake model: one reply with token usage."""
    return {"text": "12 mugs left", "usage": Usage(input_tokens=120, output_tokens=8)}


# --8<-- [start:provider]
def otlp_provider(exporter: Any, service: str = "shop-agent") -> Any:
    """A tracer provider that batches spans to `exporter`; needs opentelemetry-sdk."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


# --8<-- [end:provider]


# --8<-- [start:langfuse]
def langfuse_exporter(public_key: str, secret_key: str, host: str) -> Any:
    """Langfuse's OTLP endpoint (HTTP only), authenticated with the project's key pair."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {token}"},
    )


# host: "https://cloud.langfuse.com" (EU), "https://us.cloud.langfuse.com" (US), or your own
# --8<-- [end:langfuse]


# --8<-- [start:phoenix]
def phoenix_exporter(endpoint: str = "http://localhost:6006/v1/traces") -> Any:
    """A Phoenix server's OTLP/HTTP collector (the default port of a local Phoenix)."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint)


# --8<-- [end:phoenix]


# --8<-- [start:openllmetry]
def openllmetry_sink(app_name: str = "shop-agent") -> OTelSink:
    """OpenLLMetry (traceloop-sdk) sets the global provider; the sink then writes to it."""
    from traceloop.sdk import Traceloop

    Traceloop.init(app_name=app_name)  # its own exporter settings: TRACELOOP_* variables
    return OTelSink()  # no tracer_provider: the global one


# --8<-- [end:openllmetry]


async def run_once(sink: OTelSink) -> None:
    """One tool call and one model call, traced under one run."""
    # --8<-- [start:wire]
    executor = Executor(registry, sink=sink)
    model = executor.model(CallableModel(answer, provider="house", model="house-7b"))
    async with executor.run_scope(run_id="run_shop") as ctx:
        await executor.run("stock", {"item": "mug"}, ctx)
        ask = LLMRequest(messages=(Message(role="user", content="How many mugs?"),))
        await model.bind(ctx).generate(ask)
    # --8<-- [end:wire]


async def main() -> None:
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError:
        print("opentelemetry-sdk not installed")
        return
    exporter = InMemorySpanExporter()  # offline stand-in for langfuse_exporter(...) and friends
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    await run_once(OTelSink(tracer_provider=provider))
    for span in sorted(exporter.get_finished_spans(), key=lambda s: s.name):
        attrs = span.attributes or {}
        print(span.name, attrs["gen_ai.operation.name"], attrs["gen_ai.conversation.id"])
    provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
