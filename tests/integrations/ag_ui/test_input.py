"""`parse_input` validates and bounds a RunAgentInput; `to_messages` converts the conversation."""

import json

import pytest

pytest.importorskip("ag_ui")

from agent_adapter.core.messages import Message
from agent_adapter.core.types import ToolCall
from agent_adapter.integrations.ag_ui import AGUIInputInvalid, parse_input, to_messages
from agent_adapter.integrations.ag_ui._input import MAX_ID_CHARS, MAX_RESUME

pytestmark = pytest.mark.contract

BASE = {"threadId": "t", "runId": "r", "messages": []}
TOO_MANY = [{"interruptId": f"i{n}", "status": "cancelled"} for n in range(MAX_RESUME + 1)]


def test_parse_accepts_bytes_text_mappings_and_models():
    parsed = parse_input(json.dumps(BASE).encode())
    assert parsed.thread_id == "t"
    assert parse_input(json.dumps(BASE)).run_id == "r"
    assert parse_input(parsed) is parsed
    assert parse_input(BASE).messages == []


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (b"{not json", "not JSON"),
        (b"\xff", "not JSON"),
        (b"[]", "JSON object"),
        (json.dumps({"threadId": "t"}).encode(), "invalid"),
        ({**BASE, "threadId": ""}, "thread_id"),
        ({**BASE, "runId": "x" * (MAX_ID_CHARS + 1)}, "run_id"),
        ({**BASE, "resume": TOO_MANY}, "resume"),
        ({**BASE, "resume": [{"interruptId": "i", "status": "cancelled"}] * 2}, "more than once"),
        ({**BASE, "resume": [{"interruptId": "", "status": "cancelled"}]}, "interruptId"),
    ],
)
def test_parse_refuses_bad_input(body, fragment):
    with pytest.raises(AGUIInputInvalid, match=fragment):
        parse_input(body)


def test_parse_refuses_a_body_over_the_limit():
    with pytest.raises(AGUIInputInvalid, match="limit"):
        parse_input(json.dumps(BASE), max_bytes=10)


def test_to_messages_maps_roles_and_tool_calls():
    call = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": '{"a": 1}'}}
    body = {
        **BASE,
        "messages": [
            {"id": "1", "role": "developer", "content": "be brief"},
            {"id": "2", "role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"id": "3", "role": "assistant", "toolCalls": [call]},
            {"id": "4", "role": "tool", "toolCallId": "c1", "content": "ok"},
            {"id": "5", "role": "reasoning", "content": "hmm"},
        ],
    }
    assert to_messages(parse_input(body)) == [
        Message(role="system", content="be brief"),
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=(ToolCall(name="t", arguments={"a": 1}, id="c1"),)),
        Message(role="tool", content="ok", tool_call_id="c1"),
    ]


@pytest.mark.parametrize("arguments", ["{bad", "[1]"])
def test_to_messages_refuses_bad_tool_call_arguments(arguments):
    call = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": arguments}}
    body = {**BASE, "messages": [{"id": "3", "role": "assistant", "toolCalls": [call]}]}
    with pytest.raises(AGUIInputInvalid, match="c1"):
        to_messages(parse_input(body))
