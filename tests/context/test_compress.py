"""compress(): thresholds, step order, early stop, the report and failures."""

import logging

import pytest

from agent_adapter.context import (
    CompressionPolicy,
    CompressionReport,
    MyOwnStep,
    RemoveOldMessages,
    compress,
)
from agent_adapter.core.errors import ConfigError
from tests.context.conversations import assistant, policy, system, tool_round, user


def chars(messages):
    return sum(len(m.content) for m in messages)


CHAT = [system("s"), user("task"), *[assistant(str(n) * 100) for n in range(5)]]
CHAT += [user("latest"), assistant("last")]
TOTAL = chars(CHAT)  # 515


class Spy:
    """A step that records its calls and drops the first unprotected message it sees."""

    def __init__(self, name):
        self.name = name
        self.calls = 0

    async def apply(self, messages, context):
        self.calls += 1
        for i, m in enumerate(messages):
            if not context.is_protected(m) and m.role == "assistant":
                return messages[:i] + messages[i + 1 :]
        return messages


async def test_compress_under_the_threshold_returns_the_input():
    smaller, report = await compress(CHAT, policy(TOTAL, 100, count_tokens=chars))
    assert smaller == CHAT and report.skipped_reason == "under_threshold"
    assert report.tokens_before == report.tokens_after == TOTAL and report.steps_used == ()


async def test_compress_runs_steps_in_order_and_stops_once_under_target():
    first, second = Spy("first"), Spy("second")
    _, report = await compress(CHAT, policy(500, 420, steps=[first, second], count_tokens=chars))
    assert first.calls == 1 and second.calls == 0
    assert report.steps_used == ("first",) and report.tokens_after == TOTAL - 100
    assert report == CompressionReport(
        tokens_before=TOTAL,
        tokens_after=TOTAL - 100,
        steps_used=("first",),
        summary_cost_usd=0.0,
        skipped_reason="",
        target_tokens=420,
        messages_before=9,
        messages_after=8,
    )
    assert report.reached_target


async def test_compress_moves_on_when_a_step_is_not_enough(caplog):
    first, second = Spy("first"), Spy("second")
    with caplog.at_level(logging.WARNING, logger="agent_adapter.context.compress"):
        _, report = await compress(
            CHAT, policy(300, 200, steps=[first, second], count_tokens=chars)
        )
    assert report.steps_used == ("first", "second") and not report.reached_target
    assert "still over target" in caplog.text


async def test_compress_reports_when_nothing_could_change():
    _, report = await compress(CHAT, policy(10, 5, steps=[MyOwnStep(list)], count_tokens=chars))
    assert report.skipped_reason == "nothing_to_compress" and report.steps_used == ()


async def test_compress_force_ignores_the_start_threshold():
    _, report = await compress(
        CHAT, policy(10_000, 400, steps=[Spy("s")], count_tokens=chars), force=True
    )
    assert report.steps_used == ("s",)
    _, report = await compress(
        CHAT, policy(10_000, 9_000, steps=[Spy("s")], count_tokens=chars), force=True
    )
    assert report.skipped_reason == "under_target"


async def test_compress_uses_the_given_model_size_for_percents():
    p = CompressionPolicy(steps=[RemoveOldMessages(keep_last=2)], count_tokens=chars)
    _, report = await compress(CHAT, p, model_max_tokens=500)
    assert report.target_tokens == 300 and report.tokens_after <= 300
    with pytest.raises(ConfigError, match="model_max_tokens"):
        await compress(CHAT, p)


async def test_compress_default_steps_shrink_then_remove():
    messages = [system("s"), user("task")]
    for n in range(6):
        messages += tool_round(f"c{n}", "x" * 8000)
    messages += [user("latest"), assistant("last")]
    p = CompressionPolicy(model_max_tokens=20_000, count_tokens=chars)
    _, report = await compress(messages, p)
    assert report.steps_used[0] == "shrink_tool_results" and report.reached_target


async def test_compress_guards_a_step_that_breaks_invariants():
    def wipe(messages):
        return []

    smaller, report = await compress(
        CHAT, policy(10, 5, steps=[MyOwnStep(wipe)], count_tokens=chars)
    )
    assert smaller == [CHAT[0], CHAT[-2], CHAT[-1]] and report.steps_used == ("my_own_step",)


async def test_compress_never_mutates_the_input():
    messages = list(CHAT)

    def mutate(received):
        received.clear()
        return received

    await compress(messages, policy(10, 5, steps=[MyOwnStep(mutate)], count_tokens=chars))
    assert messages == CHAT


async def test_compress_logs_and_reraises_a_failing_step(caplog):
    def boom(messages):
        raise RuntimeError("secret detail")

    step = MyOwnStep(boom)
    with (
        caplog.at_level(logging.WARNING, logger="agent_adapter.context.compress"),
        pytest.raises(RuntimeError),
    ):
        await compress(CHAT, policy(10, 5, steps=[step], count_tokens=chars, on_step_error="raise"))
    loud = next(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "compression step my_own_step failed" in loud and "secret detail" not in loud


async def test_compress_logs_counts_never_text(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.context"):
        await compress(CHAT, policy(10, 5, steps=[Spy("s")], count_tokens=chars))
    assert "context compressed" in caplog.text
    assert "task" not in caplog.text and "latest" not in caplog.text


async def test_compress_rejects_bad_inputs():
    with pytest.raises(ConfigError, match="CompressionPolicy"):
        await compress(CHAT, {"start": 1})
    with pytest.raises(ConfigError, match="list of Message"):
        await compress([{"role": "user"}], policy(10, 5))


async def test_compress_an_empty_conversation():
    smaller, report = await compress([], policy(10, 5))
    assert smaller == [] and report.skipped_reason == "under_threshold"
