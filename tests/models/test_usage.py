from types import SimpleNamespace

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import Usage
from agent_adapter.models.usage import (
    normalise_usage,
    usage_anthropic,
    usage_gemini,
    usage_openai_chat,
    usage_openai_responses,
    usage_voyage,
)
from tests.models.conftest import fixture


def test_usage_openai_chat_every_cell_from_full_response():
    usage = usage_openai_chat(fixture("openai/chat_tool_calls"))
    assert usage == Usage(
        input_tokens=1200 - 1024 - 100,  # cached reads and cache writes excluded (§23 #2)
        cached_input_tokens=1024,
        cache_write_input_tokens=100,
        output_tokens=20,
        reasoning_tokens=4,
    )


def test_usage_openai_chat_from_usage_dict_missing_details_count_as_zero():
    usage = usage_openai_chat({"prompt_tokens": 50, "completion_tokens": 7})
    assert usage == Usage(input_tokens=50, output_tokens=7)


def test_usage_openai_chat_never_negative_when_details_exceed_total():
    raw = {
        "prompt_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 9},
    }
    usage = usage_openai_chat(raw)
    assert (usage.input_tokens, usage.cached_input_tokens, usage.cache_write_input_tokens) == (
        0,
        6,
        9,
    )


def test_usage_openai_responses_every_cell_and_ttl_split():
    usage = usage_openai_responses(fixture("openai/responses_text"))
    assert usage == Usage(
        input_tokens=1000,  # the vendor's worked example: 12000 - 4000 - 7000
        cached_input_tokens=4000,
        cache_write_input_tokens=7000,
        cache_write_by_ttl={"30m": 3000, "12h": 4000},
        output_tokens=40,
        reasoning_tokens=0,
    )


def test_usage_openai_responses_reasoning_and_cached():
    usage = usage_openai_responses(fixture("openai/responses_tool_call")["usage"])
    assert (usage.input_tokens, usage.cached_input_tokens, usage.reasoning_tokens) == (244, 256, 10)
    assert usage.cache_write_by_ttl == {}


def test_usage_gemini_camel_case_full_response():
    usage = usage_gemini(fixture("gemini/function_call"))
    assert usage == Usage(
        input_tokens=3000 - 2048,
        cached_input_tokens=2048,
        output_tokens=12 + 40,
        reasoning_tokens=40,
    )


def test_usage_gemini_snake_case_and_missing_fields():
    assert usage_gemini(fixture("gemini/text")) == Usage(input_tokens=5, output_tokens=3)
    assert usage_gemini({}) == Usage()


def test_usage_anthropic_every_cell():
    usage = usage_anthropic(fixture("anthropic/tool_use"))
    assert usage == Usage(
        input_tokens=100,
        cached_input_tokens=2000,
        cache_write_input_tokens=1500,
        cache_write_by_ttl={"5m": 1000, "1h": 500},
        output_tokens=30,
    )


def test_usage_anthropic_without_cache_creation_split():
    usage = usage_anthropic({"input_tokens": 3, "output_tokens": 4})
    assert usage == Usage(input_tokens=3, output_tokens=4)


def test_usage_accepts_objects_with_attributes():
    raw = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4, cache_write_tokens=None),
            completion_tokens_details=None,
        )
    )
    assert usage_openai_chat(raw) == Usage(input_tokens=6, cached_input_tokens=4, output_tokens=2)


def test_usage_voyage_counts_tokens_and_million_token_units():
    usage = usage_voyage({"usage": {"total_tokens": 2_000_000}})
    assert usage.input_tokens == 2_000_000
    assert usage.units == pytest.approx(2.0)
    assert usage_voyage(SimpleNamespace(total_tokens=500)).input_tokens == 500


def test_usage_none_is_empty():
    assert usage_openai_chat(None) == Usage()
    assert usage_anthropic(None) == Usage()


@pytest.mark.parametrize(
    ("provider", "api", "raw", "expected_input"),
    [
        ("openai", "chat", {"prompt_tokens": 9}, 9),
        ("openai", "responses", {"input_tokens": 8}, 8),
        ("openai", "", {"prompt_tokens": 7}, 7),
        ("openai", "", {"input_tokens": 6}, 6),
        ("openai", "", {"object": "chat.completion", "usage": {"prompt_tokens": 5}}, 5),
        ("gemini", "", {"prompt_token_count": 4}, 4),
        ("anthropic", "", {"input_tokens": 3}, 3),
        ("voyage", "", {"total_tokens": 2}, 2),
    ],
)
def test_normalise_usage_dispatches_by_provider_and_api(provider, api, raw, expected_input):
    assert normalise_usage(provider, raw, api=api).input_tokens == expected_input


def test_normalise_usage_unknown_provider_or_api():
    with pytest.raises(ConfigError, match="mistral"):
        normalise_usage("mistral", {})
    with pytest.raises(ConfigError, match="batch"):
        normalise_usage("openai", {}, api="batch")


def test_usage_gemini_tool_use_prompt_tokens_are_input():
    raw = {
        "usageMetadata": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 40,
            "toolUsePromptTokenCount": 25,
            "candidatesTokenCount": 5,
        }
    }
    assert usage_gemini(raw).input_tokens == 100 - 40 + 25
    snake = {"prompt_token_count": 10, "tool_use_prompt_token_count": 3}
    assert usage_gemini(snake).input_tokens == 13
