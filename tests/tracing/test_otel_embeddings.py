import logging

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.trace import SpanKind, StatusCode

from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.types import Usage
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from tests.tracing.otel_harness import T0, Harness


@pytest.fixture
def h():
    return Harness()


def embedding_call(
    h, provider="ollama", model="nomic", call_id="c1:embed:0", parent_id="", **finished
):
    ids = dict(call_id=call_id, provider=provider, model=model, stage="embed", texts=3)
    h.emit("embedding.started", parent_id=parent_id, **ids)
    data = dict(
        cache_hits=1,
        batches=1,
        latency_ms=250.0,
        usage={"input_tokens": 12},
        units=12 / 1e6,
        cost_usd=0.002,
        response_model=f"{model}:v1.5",
    )
    data.update(finished)
    h.emit("embedding.finished", ts=0.25, parent_id=parent_id, **ids, **data)


def test_embedding_span_name_kind_attributes_and_times(h):
    embedding_call(h)
    span = h.spans()["embeddings nomic"]
    assert span.kind is SpanKind.CLIENT
    assert (span.start_time, span.end_time) == (int(T0 * 1e9), int((T0 + 0.25) * 1e9))
    assert dict(span.attributes) == {
        "gen_ai.conversation.id": "run_1",
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": "nomic",
        "gen_ai.response.model": "nomic:v1.5",
        "gen_ai.usage.input_tokens": 12,
        "agent_adapter.stage": "embed",
        "agent_adapter.embedding.texts": 3,
        "agent_adapter.embedding.batches": 1,
        "agent_adapter.cache_hits": 1,
        "agent_adapter.cache_hit": False,
        "agent_adapter.cost_usd": 0.002,
    }
    assert span.status.status_code is not StatusCode.ERROR


def test_callable_provider_is_an_internal_span_and_unknown_cost_is_left_out(h):
    embedding_call(h, provider="callable", model="m", cost_usd=None, response_model="")
    span = h.spans()["embeddings m"]
    assert span.kind is SpanKind.INTERNAL
    assert "agent_adapter.cost_usd" not in span.attributes
    assert span.attributes["gen_ai.response.model"] == "m"


def test_embedding_metrics_record_input_tokens_and_duration(h):
    embedding_call(h)
    metrics = h.metrics()
    [tokens] = metrics["gen_ai.client.token.usage"]
    assert tokens.sum == 12
    assert dict(tokens.attributes) == {
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": "nomic",
        "gen_ai.response.model": "nomic:v1.5",
        "gen_ai.token.type": "input",
    }
    [duration] = metrics["gen_ai.client.operation.duration"]
    assert duration.sum == pytest.approx(0.25)


def test_a_call_served_from_the_cache_is_a_span_without_metrics(h):
    embedding_call(h, batches=0, cache_hits=3, usage={}, cost_usd=0.0)
    span = h.spans()["embeddings nomic"]
    assert span.attributes["agent_adapter.cache_hit"] is True
    assert span.attributes["gen_ai.usage.input_tokens"] == 0
    assert "gen_ai.client.token.usage" not in h.metrics()


def test_embedding_failed_sets_error_type_status_and_duration(h):
    ids = dict(call_id="e", provider="ollama", model="nomic", stage="embed", texts=2)
    h.emit("embedding.started", **ids)
    h.emit("embedding.failed", ts=0.5, **ids, error_kind="model-timeout", latency_ms=500.0)
    span = h.spans()["embeddings nomic"]
    assert span.attributes["error.type"] == "model-timeout"
    assert span.status.status_code is StatusCode.ERROR
    [duration] = h.metrics()["gen_ai.client.operation.duration"]
    assert duration.sum == pytest.approx(0.5)
    assert duration.attributes["error.type"] == "model-timeout"


def test_closing_event_without_a_start_still_makes_a_span(h):
    h.emit(
        "embedding.finished",
        ts=1.0,
        call_id="lonely",
        provider="ollama",
        model="nomic",
        stage="embed",
        texts=1,
        cache_hits=0,
        batches=1,
        latency_ms=400.0,
        usage={"input_tokens": 1},
        units=1e-6,
        cost_usd=None,
    )
    span = h.spans()["embeddings nomic"]
    assert span.start_time == int((T0 + 0.6) * 1e9)


def test_embedding_span_nests_under_the_tool_that_called_it(h):
    h.emit("tool.started", call_id="call_1", tool="search", kind="retrieval")
    embedding_call(h, call_id="call_1:embed:0", parent_id="call_1")
    finished = dict(tool="search", latency_ms=300.0, attempts=1, cached=False)
    h.emit("tool.finished", ts=0.3, call_id="call_1", **finished)
    spans = h.spans()
    assert spans["embeddings nomic"].parent.span_id == spans["execute_tool search"].context.span_id


def test_malformed_embedding_finished_still_ends_the_span(h, caplog):
    h.emit("embedding.started", call_id="x", provider="p", model="m", stage="embed", texts=1)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.tracing"):
        h.emit("embedding.finished", call_id="x", provider="p", model="m", latency_ms="bad")
    assert "embeddings m" in h.spans()
    assert any("OTel sink failed for embedding.finished" in r.getMessage() for r in caplog.records)


async def test_a_real_embedder_exports_spans_without_the_texts():
    h = Harness(capture_content=True)

    async def fn(texts):
        return EmbeddingResult(vectors=[[1.0, 2.0]] * len(texts), usage=Usage(input_tokens=4))

    embedder = CallableEmbedder(
        fn, model="mini", provider="local", sink=h.sink, cache=MemoryCache()
    )
    ctx = ToolContext(run_id="r", call_id="tool_1")
    await embedder.embed(["private words", "more private words"], ctx=ctx)
    await embedder.embed(["private words"], ctx=ctx)
    first, second = h.exporter.get_finished_spans()
    assert first.name == second.name == "embeddings mini"
    assert first.attributes["gen_ai.usage.input_tokens"] == 4
    assert second.attributes["agent_adapter.cache_hit"] is True
    for span in (first, second):
        assert all("private" not in str(value) for value in span.attributes.values())
        assert not span.events


def test_usage_without_tokens_records_no_token_metric_but_still_a_duration(h):
    embedding_call(h, usage={"input_tokens": 0, "units": 40.0})
    metrics = h.metrics()
    assert "gen_ai.client.token.usage" not in metrics
    [duration] = metrics["gen_ai.client.operation.duration"]
    assert duration.sum == pytest.approx(0.25)
