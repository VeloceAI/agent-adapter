"""CompressionPolicy: percents or token counts, validation, limits and the default steps."""

import pytest

from agent_adapter.context import (
    CompressionLimits,
    CompressionPolicy,
    MyOwnStep,
    RemoveOldMessages,
    ShrinkToolResults,
)
from agent_adapter.core.errors import ConfigError


def test_policy_percent_defaults_are_90_and_60():
    policy = CompressionPolicy(model_max_tokens=1000)
    assert policy.limits() == CompressionLimits(start_when_tokens_over=900, shrink_to_tokens=600)


def test_policy_percents_apply_to_a_given_model_size():
    policy = CompressionPolicy(start_when_full_percent=80, shrink_to_percent=50)
    assert policy.limits(200_000) == CompressionLimits(160_000, 100_000)


def test_policy_own_model_size_wins_over_the_given_one():
    assert CompressionPolicy(model_max_tokens=100).limits(1_000_000).shrink_to_tokens == 60


def test_policy_token_counts_ignore_the_model_size():
    policy = CompressionPolicy(start_when_tokens_over=5000, shrink_to_tokens=3000)
    assert policy.limits() == CompressionLimits(5000, 3000)
    assert policy.limits(10) == CompressionLimits(5000, 3000)


def test_policy_without_a_model_size_explains_what_to_set():
    with pytest.raises(ConfigError, match="model_max_tokens"):
        CompressionPolicy().limits()


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"start_when_full_percent": 0}, "1 to 100"),
        ({"start_when_full_percent": 101}, "1 to 100"),
        ({"shrink_to_percent": True}, "1 to 100"),
        ({"shrink_to_percent": 60.5}, "1 to 100"),
        ({"start_when_full_percent": 50, "shrink_to_percent": 50}, "less than"),
        ({"shrink_to_percent": 95}, "less than"),
        ({"start_when_tokens_over": 100}, "go together"),
        ({"shrink_to_tokens": 100}, "go together"),
        ({"start_when_tokens_over": 100, "shrink_to_tokens": 100}, "less than"),
        ({"start_when_tokens_over": 0, "shrink_to_tokens": 0}, ">= 1"),
        (
            {"start_when_tokens_over": 100, "shrink_to_tokens": 50, "shrink_to_percent": 40},
            "not both",
        ),
        (
            {"start_when_tokens_over": 200, "shrink_to_tokens": 50, "model_max_tokens": 100},
            "exceed",
        ),
        ({"model_max_tokens": 0}, ">= 1"),
        ({"model_max_tokens": "big"}, ">= 1"),
        ({"steps": [object()]}, "compression steps"),
        ({"count_tokens": 5}, "callable"),
        ({"if_still_too_long": "stronger"}, "CompressionPolicy"),
    ],
)
def test_policy_rejects_bad_settings(settings, message):
    with pytest.raises(ConfigError, match=message):
        CompressionPolicy(**settings)


def test_policy_rejects_a_bad_given_model_size():
    with pytest.raises(ConfigError, match=">= 1"):
        CompressionPolicy().limits(0)


def test_policy_is_frozen_and_keeps_steps_as_a_tuple():
    step = RemoveOldMessages()
    policy = CompressionPolicy(model_max_tokens=10, steps=[step])
    assert policy.steps == (step,)
    with pytest.raises(AttributeError):
        policy.model_max_tokens = 5


def test_policy_default_steps_without_a_summary_model():
    steps = CompressionPolicy(model_max_tokens=10).resolved_steps()
    assert [type(s) for s in steps] == [ShrinkToolResults, RemoveOldMessages]
    assert (steps[1].keep_first, steps[1].keep_last) == (1, 8)


def test_policy_given_steps_replace_the_defaults():
    step = MyOwnStep(lambda messages: messages)
    assert CompressionPolicy(model_max_tokens=10, steps=[step]).resolved_steps() == (step,)


def test_policy_accepts_a_stronger_fallback_policy():
    strong = CompressionPolicy(model_max_tokens=10, shrink_to_percent=20)
    assert (
        CompressionPolicy(model_max_tokens=10, if_still_too_long=strong).if_still_too_long is strong
    )
