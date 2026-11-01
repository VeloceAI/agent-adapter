"""Token counting: the chars/4 estimate, the one-time warning and the user's counter."""

import logging

import pytest

from agent_adapter.context import TokenMeter, estimate_tokens
from agent_adapter.context import tokens as tokens_module
from agent_adapter.context.tokens import MESSAGE_OVERHEAD_TOKENS, message_chars
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall


def test_estimate_tokens_is_chars_over_four_plus_overhead():
    message = Message(role="user", content="x" * 10)
    assert estimate_tokens([message]) == 3 + MESSAGE_OVERHEAD_TOKENS
    assert estimate_tokens([]) == 0


def test_estimate_tokens_counts_tool_calls_and_names():
    call = ToolCall(name="lookup", arguments={"q": "abc"})
    message = Message(role="assistant", tool_calls=(call,))
    assert message_chars(message) == len("lookup") + len('{"q": "abc"}')
    tool = Message(role="tool", content="1234", name="lookup")
    assert message_chars(tool) == 4 + 6


def test_estimate_tokens_handles_non_json_arguments():
    call = ToolCall(name="t", arguments={"when": object()})
    assert estimate_tokens([Message(role="assistant", tool_calls=(call,))]) > 0


def test_token_meter_warns_once_without_a_counter(monkeypatch, caplog):
    monkeypatch.setattr(tokens_module._EstimateWarning, "warned", False)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.context.tokens"):
        TokenMeter()
        TokenMeter()
    warnings = [r for r in caplog.records if "chars/4" in r.getMessage()]
    assert len(warnings) == 1


def test_token_meter_with_a_counter_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(tokens_module._EstimateWarning, "warned", False)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.context.tokens"):
        TokenMeter(len)
    assert caplog.records == []


def test_token_meter_uses_the_user_counter_and_caches_single_messages():
    calls = []

    def counter(messages):
        calls.append(len(messages))
        return 7 * len(messages)

    meter = TokenMeter(counter)
    message = Message(role="user", content="hi")
    assert meter.total([message, message]) == 14
    assert meter.of(message) == 7 and meter.of(message) == 7
    assert calls == [2, 1]


def test_token_meter_cache_is_per_object_not_per_value():
    meter = TokenMeter(lambda messages: len(messages))
    first = Message(role="user", content="same")
    second = first.model_copy()
    assert meter.of(first) == meter.of(second) == 1


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True, None])
def test_token_meter_rejects_a_bad_count(bad):
    meter = TokenMeter(lambda messages: bad)
    with pytest.raises(ConfigError, match="int >= 0"):
        meter.total([Message(role="user")])
    with pytest.raises(ConfigError, match="int >= 0"):
        meter.of(Message(role="user"))


def test_token_meter_rejects_a_non_callable_counter():
    with pytest.raises(ConfigError, match="callable"):
        TokenMeter(42)


def test_token_meter_lets_counter_errors_propagate():
    def broken(messages):
        raise RuntimeError("tokenizer missing")

    with pytest.raises(RuntimeError, match="tokenizer missing"):
        TokenMeter(broken).total([])
