"""Scorer billing: CallableReranker writes priced ledger entries and events; embedders bill too."""

import pytest

from agent_adapter.context import (
    CallableReranker,
    EmbeddingScorer,
    KeepMostRelevant,
    RerankerScorer,
    RerankResult,
    compress,
)
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelError, ModelOutputInvalid
from agent_adapter.core.types import Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from agent_adapter.tracing import REQUIRED_DATA, EventType
from tests.context.conversations import assistant, policy, user
from tests.fakes.sinks import RecordingSink

PRICES = Prices.default().merged(
    {
        "units": {
            "cohere:rerank-v3": {"provider": "cohere", "per_unit": 0.002, "unit": "search"},
            "local:tokens": {"provider": "local", "per_unit": 0.5, "unit": "1M tokens"},
            "local:bow": {"provider": "local", "per_unit": 1.0, "unit": "1M tokens"},
        }
    }
)


def priced():
    return Ledger(prices=PRICES)


def chars(messages):
    return sum(len(m.content) for m in messages)


async def test_scores_with_units_are_billed_through_the_unit_price():
    ledger, sink = priced(), RecordingSink()

    def fn(query, texts):
        return RerankResult(scores=[0.9] * len(texts), usage=Usage(units=1))

    reranker = CallableReranker(fn, provider="cohere", model="rerank-v3", ledger=ledger, sink=sink)
    ctx = ToolContext(run_id="r", call_id="tool_1")
    assert await reranker.rerank("q", ["a", "b"], ctx=ctx) == [0.9, 0.9]
    (entry,) = ledger.entries("r")
    assert (entry.kind, entry.stage, entry.provider, entry.model) == (
        "api",
        "compress",
        "cohere",
        "rerank-v3",
    )
    assert entry.units == 1 and entry.cost_usd == pytest.approx(0.002)
    assert entry.entry_id == "tool_1:rerank:0" and entry.metadata["texts"] == "2"
    started, finished = sink.events
    assert (started.type, finished.type) == (EventType.RERANK_STARTED, EventType.RERANK_FINISHED)
    assert all(set(e.data) >= REQUIRED_DATA[e.type] for e in sink.events)
    assert (finished.call_id, finished.parent_id) == ("tool_1:rerank:0", "tool_1")
    assert finished.data["cost_usd"] == pytest.approx(0.002) and finished.data["texts"] == 2


async def test_token_usage_is_billed_in_millions_and_the_reported_model_wins():
    ledger = priced()

    async def fn(query, texts):
        return {
            "scores": [0.1] * len(texts),
            "usage": {"input_tokens": 2_000_000},
            "model": "tokens",
        }

    reranker = CallableReranker(fn, provider="local", model="host", ledger=ledger, stage="rank")
    await reranker.rerank("q", ["a"], ctx=ToolContext(run_id="r"))
    (entry,) = ledger.entries("r")
    assert (entry.stage, entry.model, entry.units) == ("rank", "tokens", 2.0)
    assert entry.cost_usd == pytest.approx(1.0) and entry.input_tokens == 2_000_000


async def test_plain_scores_still_leave_an_entry_and_the_run_ledger_is_used():
    ledger = Ledger(prices=PRICES)
    reranker = CallableReranker(lambda q, t: [0.5] * len(t), provider="cohere", model="rerank-v3")
    await reranker.rerank("q", ["a"], ctx=ToolContext(run_id="r", ledger=ledger))
    (entry,) = ledger.entries("r")
    assert entry.units == 0 and entry.cost_usd == 0.0


async def test_a_failed_rerank_emits_failed_and_records_an_unpaid_entry():
    ledger, sink = priced(), RecordingSink()

    def fn(query, texts):
        raise RuntimeError("down")

    reranker = CallableReranker(fn, provider="cohere", model="rerank-v3", ledger=ledger, sink=sink)
    with pytest.raises(ModelError):
        await reranker.rerank("q", ["a"], ctx=ToolContext(run_id="r"))
    assert [e.type for e in sink.events] == [EventType.RERANK_STARTED, EventType.RERANK_FAILED]
    (entry,) = ledger.entries("r")
    assert (entry.ok, entry.error_kind, entry.cost_usd) == (False, "model-error", 0.0)


@pytest.mark.parametrize(
    "raw",
    [
        {"scores": [0.1], "usage": {"prompt_tokens": 5}},  # a provider-shaped key would bill $0
        {"usage": {"units": 1}},
        RerankResult(scores=[0.1, 0.2]),
    ],
)
async def test_bad_results_are_rejected(raw):
    with pytest.raises(ModelOutputInvalid):
        await CallableReranker(lambda q, t: raw).rerank("q", ["a"])


async def test_a_reranker_scorer_step_bills_the_rerank_and_marks_the_step():
    ledger = priced()

    def fn(query, texts):
        return RerankResult(scores=[len(t) / 1000 for t in texts], usage=Usage(units=1))

    scorer = RerankerScorer(CallableReranker(fn, provider="cohere", model="rerank-v3"))
    messages = [user("x" * 300), assistant("y" * 300), user("z" * 300), assistant("w"), user("q")]
    p = policy(500, 400, [KeepMostRelevant(scorer, keep_count=1)], count_tokens=chars)
    await compress(messages, p, ctx=ToolContext(run_id="r", call_id="c", ledger=ledger))
    entries = {e.entry_id: e for e in ledger.entries("r")}
    assert set(entries) == {"c:rerank:0", "c:compress:0"}
    assert entries["c:rerank:0"].cost_usd == pytest.approx(0.002)
    assert entries["c:compress:0"].model == "context:keep_most_relevant"
    assert ledger.summary("r").by_stage["compress"].cost_usd == pytest.approx(0.002)


async def test_an_embedding_scorer_bills_through_the_embedder():
    ledger = priced()

    def fn(texts):
        vectors = [[float(len(t)), 1.0] for t in texts]
        return EmbeddingResult(vectors=vectors, usage=Usage(input_tokens=1_000_000))

    scorer = EmbeddingScorer(CallableEmbedder(fn, provider="local", model="bow"))
    messages = [user("x" * 300), assistant("y" * 300), user("z" * 300), assistant("w"), user("q")]
    p = policy(500, 400, [KeepMostRelevant(scorer, keep_count=1)], count_tokens=chars)
    await compress(messages, p, ctx=ToolContext(run_id="r", call_id="c", ledger=ledger))
    embed = [e for e in ledger.entries("r") if e.stage == "embed"]
    assert len(embed) == 1 and embed[0].cost_usd == pytest.approx(1.0)
    assert embed[0].provider == "local" and embed[0].model == "bow"


async def test_a_mapping_may_carry_a_usage_object_or_no_usage():
    ledger = priced()
    usage = Usage(units=2)
    fn = CallableReranker(
        lambda q, t: {"scores": [0.3], "usage": usage}, provider="cohere", model="rerank-v3"
    )
    assert await fn.rerank("q", ["a"], ctx=ToolContext(run_id="r", ledger=ledger)) == [0.3]
    assert ledger.entries("r")[0].cost_usd == pytest.approx(0.004)
    assert await CallableReranker(lambda q, t: {"scores": [0.7]}).rerank("q", ["a"]) == [0.7]
