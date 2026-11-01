import asyncio
import logging

import pytest

from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelError, ModelTimeout
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.runtime.current import use_context
from tests.embeddings.support import Recorder, vector

PRICES = Prices.default().merged(
    {"units": {"local:mini": {"provider": "local", "per_unit": 0.5, "unit": "1M tokens"}}}
)


def ledger() -> Ledger:
    return Ledger(prices=PRICES)


async def test_one_api_entry_per_call_with_tokens_as_millions_of_units():
    book = ledger()
    fn = Recorder(tokens_per_text=1_000_000)
    embedder = CallableEmbedder(fn.run, model="mini", provider="local", ledger=book, batch_size=1)
    ctx = ToolContext(run_id="r1", call_id="c1", thread_id="t1")
    await embedder.embed(["a", "b"], ctx=ctx)
    await embedder.embed(["c"], ctx=ctx)
    first, second = book.entries("r1")
    assert (first.entry_id, second.entry_id) == ("c1:embed:0", "c1:embed:1")
    assert first.call_id == "c1:embed:0" and first.metadata["parent_call_id"] == "c1"
    assert (first.kind, first.stage, first.provider, first.model) == (
        "api",
        "embed",
        "local",
        "mini",
    )
    assert (first.input_tokens, first.units, first.cost_usd) == (2_000_000, 2.0, 1.0)
    assert first.price_version == PRICES.version
    assert first.metadata | {} == {
        "parent_call_id": "c1",
        "texts": "2",
        "batches": "2",
        "cache_hits": "0",
    }
    assert first.thread_id == "t1" and first.ok and not first.cache_hit
    assert first.ended_at >= first.started_at > 0 and first.latency_ms >= 0


async def test_units_given_by_fn_are_kept():
    book = ledger()

    async def fn(texts):
        return EmbeddingResult(vectors=[[1.0]] * len(texts), usage=Usage(units=4.0))

    embedder = CallableEmbedder(fn, model="mini", provider="local", ledger=book)
    await embedder.embed(["a"], ctx=ToolContext(run_id="r"))
    (entry,) = book.entries("r")
    assert (entry.units, entry.cost_usd, entry.entry_id) == (4.0, 2.0, "r:embed:0")


async def test_billed_model_is_the_one_fn_reports():
    book = ledger()
    fn = Recorder(tokens_per_text=10, model="mini-2026")
    await CallableEmbedder(fn.run, model="mini", ledger=book).embed(
        ["a"], ctx=ToolContext(run_id="r")
    )
    assert book.entries("r")[0].model == "mini-2026"


async def test_all_cache_hits_cost_nothing():
    book = ledger()
    fn = Recorder(tokens_per_text=5)
    embedder = CallableEmbedder(fn.run, model="unpriced", ledger=book, cache=MemoryCache())
    ctx = ToolContext(run_id="r")
    await embedder.embed(["a"], ctx=ctx)
    await embedder.embed(["a"], ctx=ctx)
    miss, hit = book.entries("r")
    assert miss.cost_usd is None  # no unit price for callable:unpriced
    assert (hit.cache_hit, hit.cost_usd, hit.input_tokens, hit.units) == (True, 0.0, 0, 0.0)
    assert hit.metadata["cache_hits"] == "1" and hit.metadata["batches"] == "0"


async def test_context_ledger_is_used_and_the_embedders_own_wins():
    ctx_book, own_book = ledger(), ledger()
    ctx = ToolContext(run_id="r", ledger=ctx_book)
    fn = Recorder()
    with use_context(ctx):
        await CallableEmbedder(fn.run, model="m").embed(["a"])
    await CallableEmbedder(fn.run, model="m", ledger=own_book).embed(["a"], ctx=ctx)
    assert len(ctx_book.entries("r")) == 1 and len(own_book.entries("r")) == 1


async def test_without_any_context_a_fresh_run_is_recorded():
    book = ledger()
    await CallableEmbedder(Recorder().run, model="m", ledger=book).embed(["a"])
    (entry,) = book.entries()
    assert entry.run_id.startswith("run_") and entry.entry_id.startswith("call_")
    assert entry.entry_id.endswith(":embed:0")


async def test_a_context_without_run_id_gets_a_fresh_one():
    book = ledger()
    ctx = ToolContext(call_id="c9")
    await CallableEmbedder(Recorder().run, model="m", ledger=book).embed(["a"], ctx=ctx)
    (entry,) = book.entries()
    assert entry.run_id.startswith("run_") and entry.entry_id == "c9:embed:0"


async def test_no_ledger_anywhere_is_fine():
    result = await CallableEmbedder(Recorder().run, model="m").embed(["a"])
    assert result.vectors == (vector("a"),)


async def test_failure_after_a_paid_batch_records_its_usage_with_unknown_cost():
    book = ledger()

    async def fn(texts):
        if texts == ["b"]:
            raise RuntimeError("down")
        return EmbeddingResult(vectors=[[1.0]], usage=Usage(input_tokens=7))

    embedder = CallableEmbedder(fn, model="m", ledger=book, batch_size=1)
    with pytest.raises(ModelError):
        await embedder.embed(["a", "b"], ctx=ToolContext(run_id="r"))
    (entry,) = book.entries("r")
    assert (entry.ok, entry.error_kind, entry.input_tokens) == (False, "model-error", 7)
    assert entry.cost_usd is None and entry.metadata["batches"] == "1"


async def test_failure_before_any_batch_costs_nothing():
    book = ledger()

    async def fn(texts):
        await asyncio.sleep(1)

    embedder = CallableEmbedder(fn, model="m", ledger=book, timeout_s=0.01)
    with pytest.raises(ModelTimeout):
        await embedder.embed(["a"], ctx=ToolContext(run_id="r"))
    (entry,) = book.entries("r")
    assert (entry.ok, entry.error_kind, entry.cost_usd) == (False, "model-timeout", 0.0)


async def test_cancelled_call_records_only_when_a_batch_was_paid():
    book = ledger()
    gate = asyncio.Event()

    async def fn(texts):
        if texts == ["b"]:
            gate.set()
            await asyncio.sleep(5)
        return EmbeddingResult(vectors=[[1.0]], usage=Usage(input_tokens=3))

    embedder = CallableEmbedder(fn, model="m", ledger=book, batch_size=1)
    task = asyncio.create_task(embedder.embed(["a", "b"], ctx=ToolContext(run_id="paid")))
    await gate.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (entry,) = book.entries("paid")
    assert (entry.ok, entry.error_kind, entry.input_tokens) == (False, "cancelled", 3)

    task = asyncio.create_task(embedder.embed(["b"], ctx=ToolContext(run_id="unpaid")))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert book.entries("unpaid") == []


async def test_a_broken_ledger_never_fails_the_call(caplog):
    class Broken:
        def record(self, entry: LedgerEntry) -> None:
            raise RuntimeError("disk full")

        def total_usd(self, run_id=None, *, thread_id=None):
            return 0.0

        def entries(self, run_id=None, *, thread_id=None):
            return []

    embedder = CallableEmbedder(Recorder().run, model="m", ledger=Broken())
    with caplog.at_level(logging.WARNING, logger="agent_adapter.embeddings"):
        result = await embedder.embed(["a"])
    assert result.vectors == (vector("a"),)
    assert any("ledger.record failed" in r.getMessage() for r in caplog.records)


async def test_logs_never_carry_texts(caplog):
    book = ledger()
    embedder = CallableEmbedder(Recorder().run, model="m", ledger=book, cache=MemoryCache())
    with caplog.at_level(logging.DEBUG, logger="agent_adapter"):
        await embedder.embed(["private words"])
        await embedder.embed(["private words"])
    assert caplog.records and all("private" not in r.getMessage() for r in caplog.records)
