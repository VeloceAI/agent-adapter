"""Cache backends, the LLM response cache, a stable prompt layout and a prompt-cache plug-in."""

import asyncio
from typing import Any

from agent_adapter import LLMRequest, LLMResponse, MemoryCache, Message
from agent_adapter.cache import LLMResponseCache, PromptLayout, RedisCache
from agent_adapter.cache.providers import AnthropicPromptCache, OpenAIPromptCache
from agent_adapter.models import ModelCaps


async def backends() -> None:
    # --8<-- [start:memory]
    cache = MemoryCache(max_entries=1000)  # LRU; TTLs on the monotonic clock
    await cache.set("greeting", b"hello", ttl=60)
    print(await cache.get("greeting"), await cache.get("missing"))
    # --8<-- [end:memory]


# --8<-- [start:redis]
def shared_cache() -> RedisCache:
    """Needs the redis extra; connects lazily, one client per event loop."""
    return RedisCache("redis://localhost:6379/0", prefix="myapp:", socket_timeout=2.0)


# --8<-- [end:redis]


async def response_cache() -> None:
    # --8<-- [start:llm_cache]
    llm_cache = LLMResponseCache(MemoryCache(), ttl=3600)
    request = LLMRequest(messages=(Message(role="user", content="Capital of France?"),))
    answer = LLMResponse(
        provider="openai",
        model="gpt-4.1",
        text="Paris",
        message=Message(role="assistant", content="Paris"),
        latency_ms=640.0,
    )
    where: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-4.1",
        "fields": {"api": "responses"},
    }
    await llm_cache.set(request, answer, **where)
    hit = await llm_cache.get(request, **where)
    assert hit is not None  # a miss returns None
    print(hit.text, hit.cached, hit.latency_ms)
    # --8<-- [end:llm_cache]


def layout() -> PromptLayout:
    # --8<-- [start:layout]
    layout = PromptLayout(
        static="You answer customer-support questions in one sentence.",
        sections={"tools": "Tools: search_articles, shipping_rate."},
        dynamic="Customer plan: premium.",
    )
    for message in layout.messages("What is the returns policy?"):
        print(message.role, message.cache_breakpoint, repr(message.content))
    # --8<-- [end:layout]
    return layout


async def plugins(layout: PromptLayout) -> None:
    # --8<-- [start:plugins]
    own_caps = {  # or caps=Prices.default().caps, the packaged table
        "gpt-4.1": ModelCaps(prompt_cache_key=True, prompt_cache_retention_24h=True),
        "claude-sonnet": ModelCaps(cache_control=True),
    }

    def caps(model: str) -> ModelCaps:
        return own_caps.get(model, ModelCaps())  # an unknown model gets every flag off

    request = LLMRequest(messages=layout.messages("What is the returns policy?"))

    openai = OpenAIPromptCache(retention="24h", caps=caps)
    out = await openai.apply(request, {"model": "gpt-4.1"}, model="gpt-4.1", client=None)
    print(sorted(out))

    anthropic = AnthropicPromptCache(ttl="1h", caps=caps)
    kwargs = {"system": layout.system_text(), "messages": []}
    out = await anthropic.apply(request, kwargs, model="claude-sonnet", client=None)
    print(out["system"][-1]["cache_control"])
    unknown = await anthropic.apply(request, kwargs, model="other", client=None)
    print(unknown == kwargs)  # no capability, no change
    # --8<-- [end:plugins]


async def main() -> None:
    await backends()
    await response_cache()
    await plugins(layout())


if __name__ == "__main__":
    asyncio.run(main())
