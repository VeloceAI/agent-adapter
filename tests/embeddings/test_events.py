import asyncio

import pytest

from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.types import Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.tracing import REQUIRED_DATA, EventType
from tests.embeddings.support import Recorder
from tests.fakes.sinks import RecordingSink

PRICES = Prices.default().merged(
    {"units": {"local:mini": {"provider": "local", "per_unit": 0.5, "unit": "1M tokens"}}}
)


def _has_required(event) -> bool:
    return set(event.data) >= REQUIRED_DATA[event.type]


async def test_a_call_emits_started_then_finished_with_counts_units_and_cost():
    sink = RecordingSink()
    fn = Recorder(tokens_per_text=1_000_000, model="mini")
    embedder = CallableEmbedder(
        fn.run, model="mini", provider="local", ledger=Ledger(prices=PRICES), sink=sink
    )
    await embedder.embed(["a", "b", "a"], ctx=ToolContext(run_id="r", call_id="tool_1"))
    started, finished = sink.events
    assert (started.type, finished.type) == (EventType.EMBEDDING_STARTED, "embedding.finished")
    assert all(_has_required(e) for e in sink.events)
    for event in sink.events:
        assert (event.run_id, event.call_id, event.parent_id) == ("r", "tool_1:embed:0", "tool_1")
    assert dict(started.data) == {
        "provider": "local",
        "model": "mini",
        "stage": "embed",
        "texts": 3,
    }
    data = finished.data
    assert (data["texts"], data["cache_hits"], data["batches"]) == (3, 0, 1)
    assert (data["units"], data["cost_usd"], data["response_model"]) == (2.0, 1.0, "mini")
    assert data["usage"]["input_tokens"] == 2_000_000 and data["latency_ms"] >= 0


async def test_a_cache_served_call_reports_zero_cost():
    sink = RecordingSink()
    embedder = CallableEmbedder(Recorder().run, model="m", cache=MemoryCache(), sink=sink)
    await embedder.embed(["a"])
    await embedder.embed(["a"])
    last = sink.events[-1]
    assert (last.data["cache_hits"], last.data["batches"], last.data["cost_usd"]) == (1, 0, 0.0)


async def test_cost_is_none_without_a_pricing_ledger():
    sink = RecordingSink()
    await CallableEmbedder(Recorder().run, model="m", sink=sink).embed(["a"])
    assert sink.events[-1].data["cost_usd"] is None


async def test_the_context_sink_is_used_and_the_embedders_own_wins():
    ctx_sink, own_sink = RecordingSink(), RecordingSink()
    ctx = ToolContext(run_id="r", sink=ctx_sink)
    await CallableEmbedder(Recorder().run, model="m").embed(["a"], ctx=ctx)
    await CallableEmbedder(Recorder().run, model="m", sink=own_sink).embed(["a"], ctx=ctx)
    assert len(ctx_sink.events) == 2 and len(own_sink.events) == 2


async def test_a_failure_emits_failed_with_its_kind():
    sink = RecordingSink()
    embedder = CallableEmbedder(lambda texts: [[1.0]], model="m", sink=sink)
    with pytest.raises(ModelOutputInvalid):
        await embedder.embed(["a", "b"])
    assert sink.types() == ["embedding.started", "embedding.failed"]
    failed = sink.events[-1]
    assert _has_required(failed) and failed.data["error_kind"] == "model-output-invalid"


async def test_a_cancelled_call_emits_failed_cancelled():
    sink = RecordingSink()
    gate = asyncio.Event()

    async def fn(texts):
        gate.set()
        await asyncio.sleep(5)

    task = asyncio.create_task(CallableEmbedder(fn, model="m", sink=sink).embed(["a"]))
    await gate.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sink.types() == ["embedding.started", "embedding.failed"]
    assert sink.events[-1].data["error_kind"] == "cancelled"


async def test_events_never_carry_the_texts():
    sink = RecordingSink()

    async def fn(texts):
        return EmbeddingResult(vectors=[[1.0]] * len(texts), usage=Usage(input_tokens=2))

    await CallableEmbedder(fn, model="m", sink=sink).embed(["private words"])
    assert sink.events and all("private" not in repr(e) for e in sink.events)


async def test_a_broken_sink_never_fails_the_call():
    class Broken:
        def emit(self, event):
            raise RuntimeError("sink down")

    result = await CallableEmbedder(Recorder().run, model="m", sink=Broken()).embed(["a"])
    assert len(result.vectors) == 1


async def test_a_pricing_failure_leaves_the_event_cost_unknown(caplog):
    class BadPrices:
        def unit_cost(self, item, units, *, at=None):
            raise RuntimeError("bad table")

    class PricingLedger:
        prices = BadPrices()

        def record(self, entry):
            pass

        def total_usd(self, run_id=None, *, thread_id=None):
            return 0.0

        def entries(self, run_id=None, *, thread_id=None):
            return []

    sink = RecordingSink()
    embedder = CallableEmbedder(Recorder().run, model="m", ledger=PricingLedger(), sink=sink)
    await embedder.embed(["a"])
    assert sink.events[-1].data["cost_usd"] is None
    assert any("pricing failed" in r.getMessage() for r in caplog.records)
