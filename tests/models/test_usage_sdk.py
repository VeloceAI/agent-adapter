"""The normalisers give the same Usage for SDK objects as for the recorded dicts."""

import pytest

from agent_adapter.models.usage import (
    usage_anthropic,
    usage_gemini,
    usage_openai_chat,
    usage_openai_responses,
)
from tests.models.conftest import fixture


def test_openai_sdk_objects_match_dicts():
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion
    from openai.types.responses import Response

    for name, cls, fn in [
        ("openai/chat_tool_calls", ChatCompletion, usage_openai_chat),
        ("openai/responses_text", Response, usage_openai_responses),
        ("openai/responses_tool_call", Response, usage_openai_responses),
    ]:
        raw = fixture(name)
        obj = cls.model_validate(raw)
        assert fn(obj) == fn(raw)
        assert fn(obj.usage) == fn(raw)


def test_openai_responses_ttl_details_survive_sdk_validation():
    pytest.importorskip("openai")
    from openai.types.responses import Response

    usage = usage_openai_responses(Response.model_validate(fixture("openai/responses_text")))
    assert usage.cache_write_by_ttl == {"30m": 3000, "12h": 4000}


def test_gemini_sdk_object_matches_dict():
    pytest.importorskip("google.genai")
    from google.genai import types

    for name in ("gemini/function_call", "gemini/text"):
        raw = fixture(name)
        assert usage_gemini(types.GenerateContentResponse.model_validate(raw)) == usage_gemini(raw)


def test_anthropic_sdk_object_matches_dict():
    pytest.importorskip("anthropic")
    from anthropic.types import Message

    raw = fixture("anthropic/tool_use")
    assert usage_anthropic(Message.model_validate(raw)) == usage_anthropic(raw)
