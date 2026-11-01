"""Edge paths of the AI steps: broken ledgers, pricing faults, memo limits, odd callables."""

import logging

import pytest

from agent_adapter.context import (
    CallableReranker,
    EmbeddingScorer,
    MyScorer,
    RemoveDuplicates,
    Summarize,
    SummarizeLongMessage,
)
from agent_adapter.context import entries as entries_module
from agent_adapter.context import scorer as scorer_module
from agent_adapter.context.redaction import message_view
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.embeddings import CallableEmbedder
from tests.context.ai_fakes import FakeSummariser
from tests.context.conversations import assistant, context_for, user
from tests.fakes.sinks import RecordingLedger


def chars(messages):
    return sum(len(m.content) for m in messages)


def chat():
    return [user("u" * 300), assistant("a" * 300), user("v" * 300), assistant("b" * 300), user("q")]


async def run(step, messages, target=0, ctx=None):
    context = context_for(messages, target, chars)
    context.ctx = ctx
    return await step.apply(list(messages), context), context


class BrokenLedger:
    def record(self, entry):
        raise RuntimeError("ledger down")

    def entries(self, run_id=None, *, thread_id=None):
        raise RuntimeError("ledger down")

    def total_usd(self, run_id=None, *, thread_id=None):
        return 0.0


async def test_a_broken_ledger_never_fails_a_summary(caplog):
    ctx = ToolContext(run_id="r", ledger=BrokenLedger())
    with caplog.at_level(logging.WARNING):
        result, context = await run(
            Summarize(FakeSummariser(), keep_recent_messages=1), chat(), ctx=ctx
        )
    assert result[0].content.startswith("[Summary") and context.summary_cost_usd == 0.0
    assert "ledger.record failed" in caplog.text and "ledger.entries failed" in caplog.text
    assert "ledger down" not in caplog.text


async def test_a_pricing_fault_counts_as_no_cost(monkeypatch, caplog):
    class Faulty:
        def cost(self, *args, **kwargs):
            raise ValueError("bad table")

    monkeypatch.setattr(entries_module.Prices, "default", classmethod(lambda cls: Faulty()))
    with caplog.at_level(logging.WARNING):
        _, context = await run(Summarize(FakeSummariser(), keep_recent_messages=1), chat())
    assert context.summary_cost_usd == 0.0 and "pricing failed" in caplog.text


async def test_a_summary_longer_than_what_it_covers_is_not_used():
    reply = '{"summary": "' + "w" * 3000 + '", "facts": []}'
    messages = chat()
    result, _ = await run(Summarize(FakeSummariser(reply=reply), keep_recent_messages=1), messages)
    assert result == messages


async def test_summarize_long_message_under_target_does_nothing():
    model = FakeSummariser()
    result, _ = await run(SummarizeLongMessage(model, bigger_than_tokens=10), chat(), 10_000)
    assert result == chat() and model.requests == []


async def test_similar_matching_refuses_a_step_whose_scorer_was_removed():
    step = RemoveDuplicates(match="similar", scorer=MyScorer(lambda q, t: [0.0] * len(t)))
    object.__setattr__(step, "scorer", None)
    with pytest.raises(ConfigError, match="needs a scorer"):
        await run(step, chat())


async def test_embedding_scorer_memo_is_bounded(monkeypatch):
    monkeypatch.setattr(scorer_module, "MEMO_SIZE", 2)
    scorer = EmbeddingScorer(CallableEmbedder(lambda texts: [[1.0, len(t)] for t in texts]))
    await scorer.score("a", ["bb", "ccc"])
    assert list(scorer._memo) == ["bb", "ccc"]


async def test_callable_reranker_awaits_a_coroutine_from_a_sync_function():
    async def scores(texts):
        return [0.5] * len(texts)

    reranker = CallableReranker(lambda query, texts: scores(texts))
    assert await reranker.rerank("q", ["a"]) == [0.5]


def test_message_view_redacts_strings_inside_lists():
    call = ToolCall(name="send", arguments={"lines": ["token=abc", {"cookie": "c"}, 3]})
    view = message_view(Message(role="assistant", tool_calls=(call,)))
    assert view["tool_calls"][0]["arguments"] == {"lines": ["token=***", {"cookie": "***"}, 3]}


async def test_a_failing_scorer_still_leaves_a_failed_compress_entry():
    def broken(query, texts):
        raise ConfigError("scorer down")

    ledger = RecordingLedger()
    step = RemoveDuplicates(match="similar", scorer=MyScorer(broken))
    with pytest.raises(ConfigError, match="scorer down"):
        await run(step, chat(), ctx=ToolContext(run_id="r", ledger=ledger))
    (entry,) = ledger.entries("r")
    assert (entry.stage, entry.ok, entry.error_kind) == ("compress", False, "config-invalid")
