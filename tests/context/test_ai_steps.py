"""Scorer steps: RemoveDuplicates(match="similar") and KeepMostRelevant."""

import pytest

from agent_adapter.context import (
    EmbeddingScorer,
    KeepMostRelevant,
    MyScorer,
    RemoveDuplicates,
    compress,
)
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from tests.context.ai_fakes import embedder, overlap
from tests.context.conversations import (
    DENIED,
    assistant,
    context_for,
    invariant_problems,
    policy,
    system,
    tool_round,
    user,
)
from tests.fakes.sinks import RecordingLedger


def chars(messages):
    """One token per character: exact arithmetic in tests."""
    return sum(len(m.content) for m in messages)


def run(step, messages, target):
    return step.apply(list(messages), context_for(messages, target, chars))


class Table:
    """A scorer from a fixed table of (query, text) -> score; records every call."""

    def __init__(self, table, default=0.0):
        self.table = table
        self.default = default
        self.calls = []

    async def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [self.table.get((query, t), self.default) for t in texts]


# --- RemoveDuplicates(match="similar") -------------------------------------------------------


async def test_similar_duplicates_drop_the_older_near_copy_only():
    old = user("the tax rate is 5 percent")
    near = user("the tax rate is 5 %")
    other = user("node file row")
    messages = [system(), old, assistant("a"), other, near, assistant("b"), user("latest")]
    scorer = Table({(old.content, near.content): 0.95})
    result = await run(RemoveDuplicates(match="similar", scorer=scorer), messages, 0)
    assert result == [messages[0], assistant("a"), other, near, assistant("b"), user("latest")]
    assert result[3] is near


async def test_similar_uses_the_threshold_strictly():
    old, new = user("a b c"), user("a b d")
    messages = [old, assistant("x"), new, assistant("y")]
    at = Table({(old.content, new.content): 0.92})
    assert await run(RemoveDuplicates(match="similar", scorer=at), messages, 0) == messages
    low = RemoveDuplicates(match="similar", scorer=at, same_if_score_above=0.5)
    assert old not in await run(low, messages, 0)


async def test_similar_never_compares_units_of_another_shape():
    question = user("tax rate")
    answer = assistant("tax rate")
    messages = [question, answer, user("next"), assistant("done")]
    scorer = Table({("tax rate", "tax rate"): 1.0})
    result = await run(RemoveDuplicates(match="similar", scorer=scorer), messages, 0)
    assert result == messages  # a user message is never a copy of an assistant one
    assert ("tax rate", ["tax rate"]) not in scorer.calls


async def test_similar_keeps_a_representative_of_every_chain():
    a, b, c = user("v1"), user("v2"), user("v3")
    messages = [a, assistant("1"), b, assistant("2"), c, assistant("end")]
    # a ~ b and b ~ c, but a is not like c: b goes, so a has no kept twin and stays.
    scorer = Table({("v1", "v2"): 0.99, ("v2", "v3"): 0.99})
    result = await run(RemoveDuplicates(match="similar", scorer=scorer), messages, 0)
    assert a in result and b not in result and c in result


async def test_similar_never_drops_protected_units_and_skips_empty_texts():
    pinned = user("same text", pinned=True)
    denied = tool_round("d1", DENIED)
    copy_of_denied = tool_round("d2", DENIED)
    empty = Message(role="assistant", content="")
    messages = [pinned, *denied, empty, *copy_of_denied, empty.model_copy(), user("same text")]
    step = RemoveDuplicates(match="similar", scorer=Table({}, default=1.0))
    result = await run(step, messages, 0)
    assert pinned in result
    assert all(m in result for m in denied)
    assert invariant_problems(messages, result) == []


async def test_similar_stops_once_under_target():
    first, second, third = user("x" * 30), user("y" * 30), user("z" * 30)
    messages = [first, assistant("a"), second, assistant("b"), third, assistant("c")]
    step = RemoveDuplicates(match="similar", scorer=Table({}, default=1.0))
    total = chars(messages)
    result = await run(step, messages, total - 30)
    assert first not in result and second in result


async def test_similar_with_an_embedding_scorer_end_to_end():
    old = user("alpha beta gamma tax")
    new = user("alpha beta gamma tax rate")  # close, not equal
    messages = [system(), old, assistant("ok"), user("row node file"), assistant("k"), new]
    ledger = RecordingLedger()
    ctx = ToolContext(run_id="r", ledger=ledger)
    scorer = EmbeddingScorer(embedder())
    step = RemoveDuplicates(match="similar", scorer=scorer, same_if_score_above=0.8)
    after, report = await compress(messages, policy(10, 5, [step], count_tokens=chars), ctx=ctx)
    assert old not in after and new in after
    assert report.steps_used == ("remove_duplicates",)
    assert invariant_problems(messages, after) == []
    assert {e.stage for e in ledger.entries("r")} == {"compress", "embed"}


async def test_similar_still_needs_a_scorer():
    with pytest.raises(ConfigError, match="scorer"):
        RemoveDuplicates(match="similar", scorer=object())


async def test_similar_bad_scores_fail_clearly():
    step = RemoveDuplicates(match="similar", scorer=MyScorer(lambda q, t: [2.0] * len(t)))
    with pytest.raises(ConfigError, match=r"outside 0\.\.1"):
        await run(step, [user("a"), assistant("b"), user("a!"), assistant("c")], 0)


# --- KeepMostRelevant ------------------------------------------------------------------------

TOPICS = [
    user("tax rate for 2026"),
    assistant("tax rate is 5"),
    user("node file row"),
    assistant("row value"),
    user("alpha beta"),
    assistant("gamma delta"),
]


async def test_keep_most_relevant_drops_the_least_relevant_first():
    messages = [system(), *TOPICS, user("tax rate row")]
    step = KeepMostRelevant(MyScorer(overlap), keep_count=2)
    result = await run(step, messages, chars(messages) - len("alpha beta") - len("gamma delta"))
    assert result[0] is messages[0] and result[-1] is messages[-1]
    assert TOPICS[0] in result and TOPICS[1] in result  # the tax lines are kept
    assert TOPICS[4] not in result and TOPICS[5] not in result
    assert TOPICS[2] in result  # only as many as needed


async def test_keep_most_relevant_keeps_keep_count_units_and_breaks_ties_by_age():
    messages = [*TOPICS, user("zzz")]
    step = KeepMostRelevant(MyScorer(lambda q, t: [0.0] * len(t)), keep_count=2)
    result = await run(step, messages, 0)
    assert result == [TOPICS[4], TOPICS[5], messages[-1]]


async def test_keep_most_relevant_with_a_fixed_text_and_an_embedder():
    messages = [*TOPICS, user("anything")]
    step = KeepMostRelevant(EmbeddingScorer(embedder()), compare_to="node row file", keep_count=1)
    result = await run(step, messages, 0)
    assert result == [TOPICS[2], messages[-1]]


async def test_keep_most_relevant_does_nothing_when_few_units_or_no_query():
    messages = [*TOPICS, user("q")]
    scorer = Table({})
    assert await run(KeepMostRelevant(scorer, keep_count=6), messages, 0) == messages
    no_user = [assistant("a"), assistant("b"), assistant("c")]
    assert await run(KeepMostRelevant(scorer, keep_count=0), no_user, 0) == no_user
    blank = [*TOPICS, user("")]
    assert await run(KeepMostRelevant(scorer, keep_count=0), blank, 0) == blank
    assert await run(KeepMostRelevant(scorer), messages, 10_000) == messages
    assert scorer.calls == []


async def test_keep_most_relevant_never_drops_protected_units():
    pinned = assistant("unrelated but pinned", pinned=True)
    denied = tool_round("d", DENIED)
    messages = [system(), pinned, *denied, *TOPICS, user("tax")]
    result = await run(KeepMostRelevant(MyScorer(overlap), keep_count=0), messages, 0)
    assert invariant_problems(messages, result) == []
    assert pinned in result and denied[0] in result and denied[1] in result
    assert all(m.role == "system" or m in (pinned, *denied, messages[-1]) for m in result)


def test_keep_most_relevant_checks_its_settings():
    with pytest.raises(ConfigError, match="scorer"):
        KeepMostRelevant(object())
    for bad in ("", "  ", 3):
        with pytest.raises(ConfigError, match="compare_to"):
            KeepMostRelevant(MyScorer(overlap), compare_to=bad)
    with pytest.raises(ConfigError, match="keep_count"):
        KeepMostRelevant(MyScorer(overlap), keep_count=-1)
    assert KeepMostRelevant(MyScorer(overlap)).config()["keep_count"] == 10
