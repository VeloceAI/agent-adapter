"""Scorers and the reranker port: EmbeddingScorer, RerankerScorer, MyScorer, CallableReranker."""

import asyncio
import math

import pytest

from agent_adapter.context import (
    CallableReranker,
    EmbeddingScorer,
    MyScorer,
    RerankerPort,
    RerankerScorer,
    Scorer,
)
from agent_adapter.context.scoring import ScoringSession, unit_shape, unit_text
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ConfigError,
    ModelError,
    ModelOutputInvalid,
    ModelTimeout,
    ToolUsageError,
)
from agent_adapter.embeddings import CallableEmbedder, EmbeddingResult
from tests.context.ai_fakes import embedder, logistic, overlap
from tests.context.conversations import context_for, tool_round, user
from tests.fakes.sinks import RecordingLedger

# --- CallableReranker ------------------------------------------------------------------------


async def test_callable_reranker_wraps_a_sync_function_in_a_thread():
    seen = []

    def fn(query, texts):
        seen.append((query, list(texts)))
        return [0.5] * len(texts)

    reranker = CallableReranker(fn, provider="local", model="mini")
    assert isinstance(reranker, RerankerPort)
    assert await reranker.rerank("q", ("a", "b")) == [0.5, 0.5]
    assert seen == [("q", ["a", "b"])]
    assert (reranker.provider, reranker.model) == ("local", "mini")


async def test_callable_reranker_wraps_an_async_function_and_arrays():
    class Array:
        def tolist(self):
            return [0, 1]

    async def fn(query, texts):
        return Array()

    assert await CallableReranker(fn).rerank("q", ["a", "b"]) == [0.0, 1.0]


async def test_callable_reranker_does_not_call_for_no_texts():
    def fn(query, texts):
        raise AssertionError("not called")

    assert await CallableReranker(fn).rerank("q", []) == []


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        ([0.1], "1 scores for 2 texts"),
        ("ab", "not a list"),
        ([0.1, "x"], "not a finite number"),
        ([True, 0.1], "not a finite number"),
        ([0.1, math.nan], "not a finite number"),
    ],
)
async def test_callable_reranker_rejects_bad_output(raw, problem):
    with pytest.raises(ModelOutputInvalid, match=problem):
        await CallableReranker(lambda q, t: raw).rerank("q", ["a", "b"])


async def test_callable_reranker_maps_failures_and_timeouts():
    def broken(query, texts):
        raise RuntimeError("secret detail")

    with pytest.raises(ModelError, match="rerank fn failed") as caught:
        await CallableReranker(broken).rerank("q", ["a"])
    assert "secret detail" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)

    async def slow(query, texts):
        await asyncio.sleep(5)

    with pytest.raises(ModelTimeout, match="timed out"):
        await CallableReranker(slow, timeout_s=0.01).rerank("q", ["a"])

    async def socket_timeout(query, texts):
        raise TimeoutError

    with pytest.raises(ModelTimeout, match="raised"):
        await CallableReranker(socket_timeout, timeout_s=5).rerank("q", ["a"])

    async def adapter_error(query, texts):
        raise ConfigError("mine")

    with pytest.raises(ConfigError, match="mine"):
        await CallableReranker(adapter_error).rerank("q", ["a"])


async def test_callable_reranker_checks_its_settings_and_inputs():
    with pytest.raises(ToolUsageError):
        CallableReranker("not callable")
    with pytest.raises(ConfigError):
        CallableReranker(overlap, model="")
    for bad in (0, -1, math.inf, True, "1"):
        with pytest.raises(ConfigError, match="timeout_s"):
            CallableReranker(overlap, timeout_s=bad)
    reranker = CallableReranker(overlap)
    with pytest.raises(ToolUsageError):
        await reranker.rerank(1, ["a"])
    with pytest.raises(ToolUsageError):
        await reranker.rerank("q", "abc")
    with pytest.raises(ToolUsageError):
        await reranker.rerank("q", ["a", 2])


# --- the scorers ---------------------------------------------------------------------------


async def test_embedding_scorer_is_cosine_similarity_clamped_to_0_1():
    scorer = EmbeddingScorer(embedder())
    assert isinstance(scorer, Scorer)
    same, other, partial = await scorer.score("tax rate", ["tax rate", "node file", "tax node"])
    assert same == pytest.approx(1.0)
    assert other < 0.01
    assert 0.3 < partial < 0.7
    assert await scorer.score("q", []) == []


async def test_embedding_scorer_embeds_each_text_once():
    calls = []

    def fn(texts):
        calls.append(list(texts))
        return [[1.0, float(len(t))] for t in texts]

    scorer = EmbeddingScorer(CallableEmbedder(fn))
    await scorer.score("a", ["bb", "a", "bb"])
    await scorer.score("bb", ["a", "ccc"])
    assert calls == [["a", "bb"], ["ccc"]]


async def test_embedding_scorer_handles_zero_and_negative_vectors_and_bad_sizes():
    class Port:
        provider, model = "p", "m"

        def __init__(self, vectors):
            self.vectors = vectors

        async def embed(self, texts, *, ctx=None):
            return EmbeddingResult(vectors=[self.vectors[t] for t in texts])

    port = Port({"q": (1.0, 0.0), "zero": (0.0, 0.0), "opposite": (-1.0, 0.0), "odd": (1.0,)})
    scorer = EmbeddingScorer(port)
    assert await scorer.score("q", ["zero", "opposite"]) == [0.0, 0.0]
    with pytest.raises(ModelOutputInvalid, match="dimensions"):
        await scorer.score("q", ["odd"])

    class Short(Port):
        async def embed(self, texts, *, ctx=None):
            return EmbeddingResult(vectors=[(1.0,)])

    with pytest.raises(ModelOutputInvalid, match="wrong number"):
        await EmbeddingScorer(Short({})).score("q", ["a", "b"])
    with pytest.raises(ConfigError):
        EmbeddingScorer(object())


async def test_reranker_scorer_and_my_scorer():
    reranker = CallableReranker(lambda q, texts: [logistic(len(t)) for t in texts])
    scores = await RerankerScorer(reranker).score("q", ["a", "abc"])
    assert scores == [logistic(1), logistic(3)]
    assert await RerankerScorer(reranker).score("q", []) == []
    with pytest.raises(ConfigError, match="RerankerPort"):
        RerankerScorer(object())

    assert await MyScorer(overlap).score("tax rate", ["tax", "row"]) == [0.5, 0.0]

    async def async_fn(query, texts):
        return [1.0 for _ in texts]

    assert await MyScorer(async_fn).score("q", ["a"]) == [1.0]
    assert await MyScorer(async_fn).score("q", []) == []
    with pytest.raises(ConfigError):
        MyScorer(3)


# --- ScoringSession: what every scoring step relies on ----------------------------------


def test_unit_text_and_shape_include_tool_calls():
    unit_a = context_for(tool_round("c1", "result one"), 0).units(tool_round("c1", "result one"))
    text = unit_text(unit_a[0])
    assert "lookup(" in text and "result one" in text
    other = tool_round("c2", "x", name="other")
    assert unit_shape(unit_a[0]) != unit_shape(context_for(other, 0).units(other)[0])


async def test_scoring_session_skips_empty_texts_and_checks_scores():
    messages = [user("q")]
    sent = []

    async def fn(query, texts):
        sent.append(list(texts))
        return [0.25] * len(texts)

    session = ScoringSession(MyScorer(fn), context_for(messages, 0), step="s")
    assert await session.score("q", ["", "a", ""]) == [0.0, 0.25, 0.0]
    assert await session.score("q", ["", ""]) == [0.0, 0.0]
    assert sent == [["a"]]
    for bad in ([0.1, 0.2], [1.5], [-0.5], [math.nan], ["x"], [True], "1"):
        session = ScoringSession(MyScorer(lambda q, t, bad=bad: bad), context_for([], 0), step="s")
        with pytest.raises(ConfigError, match="scorer"):
            await session.score("q", ["a"])


async def test_scoring_session_tolerates_float_noise_at_the_edges():
    session = ScoringSession(MyScorer(lambda q, t: [1 + 1e-9, -1e-9]), context_for([], 0), step="s")
    assert await session.score("q", ["a", "b"]) == [1.0, 0.0]


async def test_scoring_session_writes_one_compress_entry_and_runs_in_the_run_context():
    ledger = RecordingLedger()
    ctx = ToolContext(run_id="r1", call_id="c1", ledger=ledger)
    context = context_for([user("q")], 0)
    context.ctx = ctx
    emb = embedder()
    async with ScoringSession(EmbeddingScorer(emb), context, step="keep_most_relevant") as s:
        await s.score("tax", ["tax rate", "row"])
        await s.score("rate", ["file"])
    entries = ledger.entries("r1")
    stages = sorted(e.stage for e in entries)
    assert stages == ["compress", "embed", "embed"]  # the embedder bills under the same run
    entry = next(e for e in entries if e.stage == "compress")
    assert (entry.kind, entry.provider, entry.cost_usd) == ("api", "agent_adapter", 0.0)
    assert entry.model == "context:keep_most_relevant"
    assert entry.metadata["calls"] == "2" and entry.metadata["texts"] == "3"
    assert entry.entry_id == "c1:compress:0"


async def test_scoring_session_without_calls_writes_nothing():
    ledger = RecordingLedger()
    context = context_for([], 0)
    context.ctx = ToolContext(run_id="r1", ledger=ledger)
    async with ScoringSession(MyScorer(overlap), context, step="s"):
        pass
    assert ledger.entries() == []
