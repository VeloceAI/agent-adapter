"""parse_envelope: strict keys, safe ids, stable ids for redelivered messages."""

import json

import pytest

from agent_adapter.errors import ToolUsageError
from agent_adapter.handlers import MAX_ENVELOPE_BYTES, Envelope, parse_envelope
from agent_adapter.runtime.ids import scoped_id


def test_parse_envelope_call_with_ids_keeps_them():
    payload = {"tool": "t", "arguments": {"a": 1}, "run_id": "run_1", "call_id": "call_1"}
    assert parse_envelope(payload) == Envelope("t", {"a": 1}, "run_1", "call_1")


def test_parse_envelope_json_text_and_bytes():
    text = json.dumps({"tool": "t"})
    assert parse_envelope(text).tool == "t"
    assert parse_envelope(text.encode()).arguments == {}


def test_parse_envelope_without_ids_derives_them_from_the_message_id():
    first = parse_envelope({"tool": "t"}, message_id="m-1", source="sqs")
    again = parse_envelope({"tool": "t"}, message_id="m-1", source="sqs")
    assert first == again
    assert first.call_id == scoped_id("call_", "sqs", "m-1")
    assert first.run_id == scoped_id("run_", "sqs", "m-1")
    other = parse_envelope({"tool": "t"}, message_id="m-1", source="eventbridge")
    assert other.call_id != first.call_id


def test_parse_envelope_without_ids_or_message_id_gets_fresh_ids():
    one, two = parse_envelope({"tool": "t"}), parse_envelope({"tool": "t"})
    assert one.call_id.startswith("call_") and one.call_id != two.call_id
    assert one.run_id.startswith("run_")


def test_parse_envelope_resume():
    envelope = parse_envelope({"approval_id": "apr_1"})
    assert envelope.is_resume and envelope.approval_id == "apr_1" and envelope.tool == ""


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"tool": "t", "approved": True}, r"unknown keys \['approved'\]"),
        ({"approval_id": "apr_1", "decision": {"approved": True}}, "unknown keys"),
        ({"approval_id": "apr_1", "tool": "t"}, r"unknown keys \['tool'\]"),
        ({}, "non-empty string 'tool'"),
        ({"tool": ""}, "non-empty string 'tool'"),
        ({"tool": 3}, "non-empty string 'tool'"),
        ({"tool": "t", "arguments": [1]}, "'arguments' must be an object, got list"),
        ({"tool": "t", "run_id": ""}, "'run_id' must be a string of 1-256"),
        ({"tool": "t", "call_id": "x" * 257}, "'call_id' must be a string of 1-256"),
        ({"tool": "t", "call_id": 7}, "'call_id' must be a string"),
        ({"tool": "t", "call_id": "a\nb"}, "'call_id' must be printable"),
        ({"approval_id": None}, "'approval_id' must be a string"),
        ("[1, 2]", "must be a JSON object, got list"),
        ("{not json", "not valid JSON"),
        (b"\xff\xfe", "not valid JSON"),
        (42, "must be a JSON object, got int"),
    ],
)
def test_parse_envelope_refuses_bad_payloads(payload, match):
    with pytest.raises(ToolUsageError, match=match):
        parse_envelope(payload)


def test_parse_envelope_refuses_oversized_payloads_before_parsing():
    big = '{"tool": "t", "arguments": {"x": "' + "a" * MAX_ENVELOPE_BYTES + '"}}'
    with pytest.raises(ToolUsageError, match="the limit is"):
        parse_envelope(big)
    with pytest.raises(ToolUsageError, match="the limit is"):
        parse_envelope(big.encode())
