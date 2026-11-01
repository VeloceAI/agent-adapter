"""`LLMRequest.compression` is adapter-only: never in a payload, a dump or a cache key."""

import json

import pytest

from agent_adapter.cache.keys import llm_key
from agent_adapter.context import CompressionPolicy
from agent_adapter.core.messages import LLMRequest, Message
from tests.models.test_pinned_not_on_the_wire import BUILDERS, HISTORY

POLICY = CompressionPolicy(model_max_tokens=1000)


@pytest.mark.parametrize("provider", list(BUILDERS))
def test_every_provider_payload_is_the_same_with_or_without_compression(provider):
    build = BUILDERS[provider]
    plain = build(LLMRequest(messages=HISTORY))
    with_policy = build(LLMRequest(messages=HISTORY, compression=POLICY))
    assert with_policy == plain
    text = json.dumps(with_policy, default=str)
    assert "compression" not in text and "shrink" not in text


def test_compression_is_not_dumped_nor_part_of_the_cache_key():
    request = LLMRequest(messages=HISTORY, compression=POLICY)
    assert "compression" not in request.model_dump()
    assert "compression" not in request.model_dump_json()
    assert "compression" not in repr(request)
    plain = LLMRequest(messages=HISTORY)
    keys = [llm_key("aa", r, provider="p", model="m", fields={}) for r in (request, plain)]
    assert keys[0] == keys[1]


def test_compression_must_be_a_policy():
    with pytest.raises(ValueError, match="CompressionPolicy"):
        LLMRequest(messages=(Message(role="user", content="x"),), compression="aggressive")
    assert LLMRequest(messages=(), compression=POLICY).compression is POLICY
