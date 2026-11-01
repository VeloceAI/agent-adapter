"""Property-style invariant tests of the AI steps over many seeded random conversations."""

import random

import pytest

from agent_adapter.context import (
    EmbeddingScorer,
    KeepMostRelevant,
    MyScorer,
    RemoveDuplicates,
    Summarize,
    SummarizeLongMessage,
    compress,
    estimate_tokens,
)
from agent_adapter.context.reference import keeps_instructions
from agent_adapter.core.messages import Message
from tests.context.ai_fakes import FakeSummariser, embedder, overlap
from tests.context.conversations import (
    invariant_problems,
    policy,
    seeded_conversation,
    system_problems,
)

SEEDS = range(150)


def _everything_alike(query, texts):
    return [1.0] * len(texts)


STEPS = {
    "similar_embedder": lambda: RemoveDuplicates(
        match="similar", scorer=EmbeddingScorer(embedder()), same_if_score_above=0.9
    ),
    "similar_everything": lambda: RemoveDuplicates(
        match="similar", scorer=MyScorer(_everything_alike)
    ),
    "relevant_overlap": lambda: KeepMostRelevant(MyScorer(overlap), keep_count=2),
    "relevant_none_kept": lambda: KeepMostRelevant(EmbeddingScorer(embedder()), keep_count=0),
    "relevant_fixed_text": lambda: KeepMostRelevant(
        MyScorer(overlap), compare_to="tax rate", keep_count=1
    ),
    "summarize": lambda: Summarize(FakeSummariser(), keep_recent_messages=2, summary_max_tokens=20),
    "summarize_all": lambda: Summarize(FakeSummariser(), keep_recent_messages=0),
    "summarize_paragraph": lambda: Summarize(FakeSummariser(), summary_style="paragraph"),
    "long_message": lambda: SummarizeLongMessage(FakeSummariser(), bigger_than_tokens=50),
}
SYSTEM_STEPS = {
    "long_message_system": lambda: SummarizeLongMessage(
        FakeSummariser(reply='{"summary": "rule", "facts": []}'),
        bigger_than_tokens=50,
        include_system_prompt=True,
    ),
}


RULES = "Rule: answer briefly.\nRule: never refund.\n"


def _with_big_system(messages, rng):
    if messages and messages[0].role == "system":
        reference = "fact " * rng.randint(50, 400)
        big = Message(role="system", content=f"{RULES}<reference>\n{reference}\n</reference>")
        return [big, *messages[1:]]
    return messages


async def _compress(messages, step_factory):
    total = estimate_tokens(messages)
    start, target = max(2, total // 2), max(1, total // 4)
    return await compress(
        messages, policy(start, target, [step_factory()], count_tokens=estimate_tokens)
    )


def _summaries(messages):
    return [m for m in messages if m.content.startswith("[Summary of")]


@pytest.mark.parametrize("step", list(STEPS))
async def test_every_ai_step_keeps_every_invariant_and_never_touches_the_system(step):
    changed = 0
    for seed in SEEDS:
        before = _with_big_system(list(seeded_conversation(seed)), random.Random(seed))
        after, report = await _compress(before, STEPS[step])
        assert invariant_problems(before, after) == [], (step, seed)
        assert system_problems(before, after, may_shrink=False) == [], (step, seed)
        for summary in _summaries(after):
            assert summary.role in ("user", "assistant", "tool") and not summary.pinned
            assert not summary.tool_calls
        changed += bool(report.steps_used)
    assert changed > 20, step  # the step really worked on most conversations


@pytest.mark.parametrize("step", list(SYSTEM_STEPS))
async def test_include_system_prompt_only_shortens_the_system_text(step):
    shortened = 0
    for seed in SEEDS:
        before = _with_big_system(list(seeded_conversation(seed)), random.Random(seed))
        after, _ = await _compress(before, SYSTEM_STEPS[step])
        assert invariant_problems(before, after, system_may_shrink=True) == [], (step, seed)
        if before and before[0].role == "system":
            assert keeps_instructions(before[0].content, after[0].content), (step, seed)
            shortened += after[0] != before[0]
    assert shortened > 20


@pytest.mark.parametrize("step", list(STEPS) + list(SYSTEM_STEPS))
async def test_every_ai_step_is_deterministic(step):
    factory = {**STEPS, **SYSTEM_STEPS}[step]
    for seed in range(0, 150, 5):
        before = _with_big_system(list(seeded_conversation(seed)), random.Random(seed))
        copy = [m.model_copy() for m in before]
        first, first_report = await _compress(before, factory)
        second, second_report = await _compress(copy, factory)
        assert first == second and first_report == second_report, (step, seed)
