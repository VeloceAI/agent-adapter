"""StepContext: token helpers, protection and the summary cost AI steps add."""

import pytest

from agent_adapter.core.errors import ConfigError
from tests.context.conversations import assistant, context_for, user


def test_step_context_adds_summary_costs():
    context = context_for([user("a"), assistant("b")], 10)
    context.add_summary_cost(0.25)
    context.add_summary_cost(1)
    assert context.summary_cost_usd == 1.25


@pytest.mark.parametrize(
    ("bad", "message"), [(-1, ">= 0"), (float("inf"), "finite"), (True, "finite"), ("1", "finite")]
)
def test_step_context_rejects_bad_costs(bad, message):
    with pytest.raises(ConfigError, match=message):
        context_for([], 10).add_summary_cost(bad)


def test_step_context_helpers():
    messages = [user("a"), assistant("b")]
    context = context_for(messages, 5, lambda ms: len(ms))
    assert context.tokens(messages) == 2 and context.message_tokens(messages[0]) == 1
    assert context.is_protected(messages[1]) and context.under_target(5)
    assert not context.under_target(6) and [u.kind for u in context.units(messages)] == [
        "user",
        "assistant",
    ]
