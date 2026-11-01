"""Plug-ins without `caps=` read the capability flags of the packaged price table."""

from types import SimpleNamespace

import pytest

from agent_adapter.cache.providers import (
    AnthropicPromptCache,
    GeminiExplicitCache,
    OpenAIPromptCache,
    OpenAIPromptCacheV2,
)
from agent_adapter.cache.providers._base import caps_for
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import ModelCaps
from agent_adapter.cost import Prices

SYSTEM = "S" * 8200  # ~2050 estimated tokens
REQUEST = LLMRequest(
    messages=(Message(role="system", content=SYSTEM), Message(role="user", content="q"))
)


@pytest.mark.parametrize(
    "model", ["gpt-4.1", "gpt-5.6", "gemini-2.5-pro", "claude-sonnet-4-5", "unknown-model"]
)
def test_no_lookup_means_the_default_price_table(model):
    assert caps_for(None, model) == Prices.default().caps(model)


def test_an_explicit_lookup_wins():
    assert caps_for(lambda model: ModelCaps(cache_control=True), "gpt-4.1").cache_control


async def test_openai_v1_default_caps_allow_the_key_and_24h_on_gpt_4_1():
    kwargs = {"model": "gpt-4.1", "input": []}
    out = await OpenAIPromptCache(key="k", retention="24h").apply(
        REQUEST, kwargs, model="gpt-4.1", client=None
    )
    assert out["prompt_cache_key"] == "k" and out["prompt_cache_retention"] == "24h"
    mini = await OpenAIPromptCache(key="k", retention="24h").apply(
        REQUEST, kwargs, model="gpt-4.1-mini", client=None
    )
    assert mini["prompt_cache_key"] == "k" and "prompt_cache_retention" not in mini


async def test_openai_v2_default_caps_gate_on_gpt_5_6_only():
    kwargs = {"model": "x", "input": []}
    plugin = OpenAIPromptCacheV2(key="k")
    on = await plugin.apply(REQUEST, kwargs, model="gpt-5.6", client=None)
    assert on["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert await plugin.apply(REQUEST, kwargs, model="gpt-4.1", client=None) == kwargs


async def test_anthropic_default_caps_cover_the_claude_family():
    kwargs = {"model": "claude-x", "system": "S", "messages": []}
    out = await AnthropicPromptCache().apply(REQUEST, kwargs, model="claude-opus-4-1", client=None)
    assert out["system"] == [
        {"type": "text", "text": "S", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
    ]
    assert await AnthropicPromptCache().apply(REQUEST, kwargs, model="other", client=None) == kwargs


async def test_gemini_default_caps_gate_explicit_caching_and_min_tokens():
    created = []

    async def create(*, model, config):
        created.append(model)
        return SimpleNamespace(name="cachedContents/1", usage_metadata=None)

    client = SimpleNamespace(aio=SimpleNamespace(caches=SimpleNamespace(create=create)))
    kwargs = {"model": "m", "contents": [], "config": {"system_instruction": SYSTEM}}
    plugin = GeminiExplicitCache()
    out = await plugin.apply(REQUEST, kwargs, model="gemini-2.5-pro", client=client)
    assert out["config"] == {"cached_content": "cachedContents/1"}
    # gemini-4-argon has implicit caching only; gemini-2.5-flash-lite needs 4096 tokens
    for model in ("gemini-4-argon", "gemini-2.5-flash-lite"):
        assert await plugin.apply(REQUEST, kwargs, model=model, client=client) == kwargs
    assert created == ["gemini-2.5-pro"]
