"""Summarize never reorders history: each run between protected messages is summarised in place."""

import random
import re

from agent_adapter.context import Summarize, compress
from agent_adapter.context.summary_prompt import ContextSummary
from agent_adapter.core.messages import LLMResponse, Message
from tests.context.ai_fakes import FakeSummariser, covered
from tests.context.conversations import DENIED, assistant, policy, system, tool_round, user

TAG = re.compile(r"#(\d+)")


class TagSummariser(FakeSummariser):
    """Names the `#n` tags of the messages it covers, so a test can see what it summarised."""

    def _answer(self, request):
        self.requests.append(request)
        tags = sorted({int(t) for view in covered(request) for t in TAG.findall(str(view))})
        text = ContextSummary(
            summary="covered " + " ".join(f"@{t}" for t in tags)
        ).model_dump_json()
        message = Message(role="assistant", content=text)
        return LLMResponse(provider="fake", model=self.model, text=text, message=message)


def chars(messages):
    return sum(len(m.content) for m in messages)


def tagged(n, make, pad=200, **extra):
    return make(f"#{n} " + "w" * pad, **extra)


def _summaries(messages):
    return [m for m in messages if m.content.startswith("[Summary of earlier")]


def _covered_tags(summary):
    return [int(t) for t in re.findall(r"@(\d+)", summary.content)]


def _tag(message):
    found = TAG.search(message.content)
    return int(found.group(1)) if found else None


async def test_the_review_repro_keeps_the_pinned_message_between_its_neighbours():
    pinned = tagged(3, user, pinned=True)
    messages = [
        system(),
        tagged(0, user),
        *tool_round("c1", "#1 " + "r" * 200),
        pinned,
        tagged(4, assistant),
        tagged(5, user),
        tagged(6, assistant),
        user("#7 latest"),
    ]
    model = TagSummariser()
    p = policy(300, 50, [Summarize(model, keep_recent_messages=2)], count_tokens=chars)
    after, _ = await compress(messages, p)
    first, second = _summaries(after)
    assert after.index(first) < after.index(pinned) < after.index(second)
    assert max(_covered_tags(first)) < 3 < min(_covered_tags(second))
    assert len(model.requests) == 2  # one summary per run


def _random_history(rng):
    out = [system()]
    n = 0
    for _ in range(rng.randint(3, 12)):
        n += 1
        kind = rng.random()
        if kind < 0.15:
            out.append(tagged(n, user, pinned=True))
        elif kind < 0.25:
            call, result = tool_round(f"d{n}", DENIED)
            out += [call.model_copy(update={"content": f"#{n}"}), result]
        elif kind < 0.45:
            out += tool_round(f"c{n}", f"#{n} " + "r" * rng.randint(50, 300))
        else:
            out.append(tagged(n, rng.choice((user, assistant)), pad=rng.randint(50, 300)))
    out.append(user(f"#{n + 1} latest"))
    return out


def _protected_tag(message):
    approval = any(call.id.startswith("d") for call in message.tool_calls)
    return _tag(message) if message.pinned or approval else None


async def test_summaries_never_move_history_across_a_protected_message():
    summarised = 0
    for seed in range(120):
        rng = random.Random(seed)
        before = _random_history(rng)
        step = Summarize(TagSummariser(), keep_recent_messages=rng.randint(0, 3))
        after, _ = await compress(before, policy(50, 10, [step], count_tokens=chars))
        kept = [m for m in after if any(m is b for b in before)]
        assert kept == [b for b in before if any(b is m for m in kept)], seed  # same order
        marks = [(i, _protected_tag(m)) for i, m in enumerate(after)]
        for position, summary in enumerate(after):
            if not summary.content.startswith("[Summary of earlier"):
                continue
            summarised += 1
            for index, mark in marks:
                if mark is None:
                    continue
                tags = _covered_tags(summary)
                assert all(t < mark for t in tags) if position < index else True, seed
                assert all(t > mark for t in tags) if position > index else True, seed
    assert summarised > 50
