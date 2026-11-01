import logging

import pytest

from agent_adapter.cache import encode, llm_key
from agent_adapter.cache.llm_cache import LLMResponseCache
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.types import ToolCall, Usage
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.clock import FakeClock

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),), temperature=0.0)
KEY = {"provider": "openai", "model": "gpt-4.1", "fields": {"api": "responses"}}


def response(**changes) -> LLMResponse:
    base = {
        "provider": "openai",
        "model": "gpt-4.1-2025-04-14",
        "text": "hello",
        "tool_calls": (ToolCall(name="search", arguments={"q": "x"}, id="c1"),),
        "usage": Usage(input_tokens=10, output_tokens=2),
        "message": Message(role="assistant", content="hello"),
        "latency_ms": 812.0,
        "parsed": {"a": 1},
    }
    base.update(changes)
    return LLMResponse(**base)


async def test_hit_returns_the_response_marked_cached_with_zero_latency():
    port = DictCache()
    cache = LLMResponseCache(port)
    stored = response()
    assert await cache.set(REQUEST, stored, **KEY)
    hit = await cache.get(REQUEST, **KEY)
    assert hit is not None
    assert hit.cached is True and hit.latency_ms == 0.0
    assert hit.model_copy(update={"cached": False, "latency_ms": 812.0}) == stored
    assert list(port.data) == [llm_key("aa", REQUEST, **KEY)]


async def test_miss_and_key_isolation():
    cache = LLMResponseCache(DictCache(), namespace="app")
    assert await cache.get(REQUEST, **KEY) is None
    await cache.set(REQUEST, response(), **KEY)
    assert await cache.get(REQUEST, **{**KEY, "model": "gpt-5"}) is None
    assert await cache.get(REQUEST, **{**KEY, "fields": {"api": "chat"}}) is None
    other = REQUEST.model_copy(update={"temperature": 0.5})
    assert await cache.get(other, **KEY) is None
    # Labels outside the key share the entry.
    labelled = REQUEST.model_copy(update={"stage": "draft", "metadata": {"call_id": "x"}})
    assert await cache.get(labelled, **KEY) is not None


async def test_ttl_is_passed_to_the_port():
    clock = FakeClock()
    cache = LLMResponseCache(DictCache(clock), ttl=30)
    await cache.set(REQUEST, response(), **KEY)
    clock.advance(31)
    assert await cache.get(REQUEST, **KEY) is None


async def test_error_responses_and_cached_responses_are_not_stored():
    port = DictCache()
    cache = LLMResponseCache(port)
    assert not await cache.set(REQUEST, response(finish_reason="error"), **KEY)
    assert not await cache.set(REQUEST, response(cached=True), **KEY)
    assert port.data == {}


async def test_requests_that_opt_out_are_neither_read_nor_stored():
    port = DictCache()
    cache = LLMResponseCache(port)
    opted_out = REQUEST.model_copy(update={"response_cache": False})
    assert not await cache.set(opted_out, response(), **KEY)
    await cache.set(REQUEST, response(), **KEY)
    assert await cache.get(opted_out, **KEY) is None
    assert port.calls == [("set", llm_key("aa", REQUEST, **KEY))]


async def test_uncacheable_request_or_payload_is_skipped(caplog):
    port = DictCache()
    cache = LLMResponseCache(port)
    odd = REQUEST.model_copy(update={"extra": {"x": float("nan")}})
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.cache.llm_cache"):
        assert await cache.get(odd, **KEY) is None
        assert not await cache.set(odd, response(), **KEY)
        assert not await cache.set(REQUEST, response(parsed=object()), **KEY)
    assert port.data == {}
    assert "not cacheable" in caplog.text


async def test_broken_backend_is_a_miss_with_a_warning(caplog):
    cache = LLMResponseCache(BrokenCache())
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.llm_cache"):
        assert await cache.get(REQUEST, **KEY) is None
        assert not await cache.set(REQUEST, response(), **KEY)
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.WARNING]


@pytest.mark.parametrize(
    "data",
    [b"not json", encode("tool", {"provider": "x"}), encode("llm", {"provider": "x"})],
)
async def test_bad_entries_are_misses(data):
    port = DictCache()
    await port.set(llm_key("aa", REQUEST, **KEY), data)
    assert await LLMResponseCache(port).get(REQUEST, **KEY) is None
