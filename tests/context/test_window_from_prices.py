"""A percent policy needs no setup for a model in the packaged price table."""

import pytest

from agent_adapter.context import CompressingModel, CompressionPolicy, RemoveOldMessages
from agent_adapter.context.run import context_window
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import ModelCaps
from agent_adapter.cost import Prices
from tests.fakes.models import ScriptedModel, text

PERCENT = CompressionPolicy(
    start_when_full_percent=90,
    shrink_to_percent=60,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
)


@pytest.mark.parametrize(
    ("model", "window"),
    [
        ("gpt-5", 400_000 - 128_000),  # room left for the longest reply
        ("gpt-5.4-2026-03-05", 1_050_000 - 128_000),
        ("gemini-2.5-flash", 1_048_576 - 65_536),
        ("claude-sonnet-5-5", 1_000_000 - 128_000),
    ],
)
def test_a_known_model_gets_its_prompt_window_from_the_table(model, window):
    assert context_window(PERCENT, model) == window
    limits = PERCENT.limits(context_window(PERCENT, model))
    assert limits.start_when_tokens_over == window * 90 // 100
    assert limits.shrink_to_tokens == window * 60 // 100


@pytest.mark.parametrize("model", ["house-13b", "gpt-5.6", "claude-sonnet-4-5"])
def test_an_unknown_window_is_a_clear_config_error(model):
    with pytest.raises(ConfigError, match=f"no context window is known for model {model!r}"):
        context_window(PERCENT, model)


def test_the_policy_and_token_counts_still_win_over_the_table():
    own = CompressionPolicy(model_max_tokens=8_000)
    assert context_window(own, "gpt-5") == 8_000
    tokens = CompressionPolicy(start_when_tokens_over=100, shrink_to_tokens=50)
    assert context_window(tokens, "house-13b") is None


def test_a_user_table_with_only_a_window_keeps_all_of_it():
    table = Prices.load(
        {
            "version": "mine",
            "models": {
                "house-13b": {"provider": "house", "free": True, "caps": {"context_window": 32_000}}
            },
        }
    )
    assert context_window(PERCENT, "house-13b", table.caps) == 32_000
    both = ModelCaps(context_window=32_000, max_output_tokens=4_000)
    assert context_window(PERCENT, "house-13b", lambda _: both) == 28_000


async def test_compressing_model_on_a_table_model_needs_no_window_setting():
    inner = ScriptedModel([text("ok")], model="gpt-5")
    model = CompressingModel(inner, PERCENT)  # no caps, no model_max_tokens: the table's
    history = (Message(role="user", content="a"), Message(role="user", content="b"))
    assert (await model.generate(LLMRequest(messages=history))).text == "ok"
    assert inner.requests[0].messages == history  # far under 90% of 272,000 tokens
