"""`Message.pinned` is adapter-only: every provider payload is the same with or without it."""

import json

import pytest

from agent_adapter.core.messages import LLMRequest, Message, ProviderRaw
from agent_adapter.core.types import ToolCall
from agent_adapter.models._anthropic_convert import build_anthropic_request
from agent_adapter.models._gemini_convert import build_gemini_request
from agent_adapter.models._openai_chat import build_chat_request
from agent_adapter.models._openai_responses import build_responses_request

CALL = ToolCall(name="lookup", arguments={"q": "x"}, id="c1")
HISTORY = (
    Message(role="system", content="be brief"),
    Message(role="user", content="find x"),
    Message(role="assistant", content="", tool_calls=(CALL,)),
    Message(role="tool", content="found", tool_call_id="c1", name="lookup"),
    Message(
        role="assistant",
        content="done",
        raw=ProviderRaw(provider="gemini", items=[{"text": "done"}]),
    ),
    Message(role="user", content="thanks"),
)

BUILDERS = {
    "openai-chat": lambda r: build_chat_request(r, model="m"),
    "openai-responses": lambda r: build_responses_request(r, model="m"),
    "anthropic": lambda r: build_anthropic_request(r, model="m", max_tokens=10),
    "gemini": lambda r: build_gemini_request(r, model="m"),
}


def _pinned(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    return tuple(m.model_copy(update={"pinned": True}) for m in messages)


@pytest.mark.parametrize("provider", list(BUILDERS))
def test_pinned_messages_build_the_same_provider_payload(provider):
    build = BUILDERS[provider]
    plain = build(LLMRequest(messages=HISTORY))
    pinned = build(LLMRequest(messages=_pinned(HISTORY)))
    assert pinned == plain
    assert "pinned" not in json.dumps(pinned, default=str)
