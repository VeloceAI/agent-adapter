"""Message.pinned: default off, frozen, and dumped only when True (old data still loads)."""

import json

import pytest
from pydantic import ValidationError

from agent_adapter.cache.keys import llm_key
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import ToolCall

LEGACY = {
    "role": "assistant",
    "content": "hi",
    "tool_calls": [],
    "tool_call_id": "",
    "name": "",
    "raw": None,
    "cache_breakpoint": False,
}


def test_message_pinned_defaults_to_false_and_is_frozen():
    message = Message(role="user", content="task")
    assert message.pinned is False
    with pytest.raises(ValidationError):
        message.pinned = True


def test_message_unpinned_dump_is_unchanged():
    assert Message(role="assistant", content="hi").model_dump(mode="json") == LEGACY
    assert "pinned" not in Message(role="user").model_dump_json()


def test_message_pinned_dump_round_trips():
    pinned = Message(role="user", content="task", pinned=True)
    dumped = pinned.model_dump(mode="json")
    assert dumped["pinned"] is True
    assert Message.model_validate(dumped) == pinned
    assert Message.model_validate_json(pinned.model_dump_json()) == pinned


def test_message_legacy_data_without_pinned_loads_unpinned():
    assert Message.model_validate(LEGACY).pinned is False
    assert Message.model_validate_json(json.dumps(LEGACY)).pinned is False


def test_message_pinned_rejects_non_bool():
    with pytest.raises(ValidationError):
        Message(role="user", pinned="maybe")


def test_message_pinned_copy_keeps_other_fields():
    message = Message(role="assistant", tool_calls=[ToolCall(name="t", id="1")])
    pinned = message.model_copy(update={"pinned": True})
    assert pinned.tool_calls == message.tool_calls and pinned.pinned


def test_llm_key_of_unpinned_request_matches_the_pre_pinned_payload():
    request = LLMRequest(messages=(Message(role="user", content="q"),))
    legacy = LLMRequest.model_validate({"messages": [{**LEGACY, "role": "user", "content": "q"}]})
    dumped = request.model_dump(exclude={"stream", "stage", "response_cache", "metadata"})
    assert "pinned" not in dumped["messages"][0]
    key = llm_key("aa", request, provider="p", model="m", fields={})
    assert key == llm_key("aa", legacy, provider="p", model="m", fields={})
