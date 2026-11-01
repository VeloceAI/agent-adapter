import logging

import pytest

from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import (
    LLMRequest,
    Message,
    OutputSchema,
    ProviderRaw,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.core.types import ToolCall
from agent_adapter.models import _convert
from agent_adapter.models._anthropic_convert import (
    AnthropicStreamAssembler,
    build_anthropic_request,
    parse_anthropic_message,
)
from tests.models.conftest import fixture

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={"type": "object", "title": "T", "properties": {"city": {"type": "string"}}},
)


def _request(*messages, **fields):
    return LLMRequest(messages=messages or (Message(role="user", content="hi"),), **fields)


def test_build_minimal_uses_blocks_and_default_max_tokens():
    assert build_anthropic_request(_request(), model="claude", max_tokens=100) == {
        "model": "claude",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }


def test_build_system_tools_schema_effort_sampling():
    schema = OutputSchema(name="A", json_schema={"type": "object", "title": "A", "properties": {}})
    kwargs = build_anthropic_request(
        _request(
            Message(role="system", content="S1"),
            Message(role="system", content="S2"),
            Message(role="user", content="hi"),
            tools=(TOOL,),
            output_schema=schema,
            reasoning="minimal",
            stop=("Z",),
            max_output_tokens=9,
        ),
        model="claude",
        max_tokens=100,
    )
    assert kwargs["system"] == [{"type": "text", "text": "S1"}, {"type": "text", "text": "S2"}]
    assert kwargs["tools"] == [
        {
            "name": "get_weather",
            "description": "Weather.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert kwargs["tool_choice"] == {"type": "auto"}
    assert kwargs["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "title": "A",
                "properties": {},
                "additionalProperties": False,
            },
        },
        "effort": "low",
    }
    assert kwargs["stop_sequences"] == ["Z"]
    assert kwargs["max_tokens"] == 9


@pytest.mark.parametrize(
    ("choice", "parallel", "wire"),
    [
        ("none", False, {"type": "none"}),
        ("required", None, {"type": "any"}),
        ("required", False, {"type": "any", "disable_parallel_tool_use": True}),
        ("get_weather", True, {"type": "tool", "name": "get_weather"}),
    ],
)
def test_build_tool_choice(choice, parallel, wire):
    request = _request(tools=(TOOL,), tool_choice=choice, parallel_tool_calls=parallel)
    assert build_anthropic_request(request, model="c", max_tokens=1)["tool_choice"] == wire


@pytest.mark.parametrize(
    ("effort", "wire"), [("low", "low"), ("medium", "medium"), ("high", "high")]
)
def test_build_effort(effort, wire):
    kwargs = build_anthropic_request(_request(reasoning=effort), model="c", max_tokens=1)
    assert kwargs["output_config"] == {"effort": wire}


def test_build_drops_sampling_fields_with_one_warning_each(caplog, monkeypatch):
    monkeypatch.setattr(_convert, "_warned", set())
    request = _request(seed=1, temperature=0.2, top_p=0.9)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            kwargs = build_anthropic_request(request, model="c", max_tokens=1)
    for field in ("seed", "temperature", "top_p"):
        assert field not in kwargs
        assert caplog.text.count(f"'{field}'") == 1


def test_build_history_raw_echo_and_merged_tool_results():
    raw = ProviderRaw(
        provider="anthropic",
        items=[
            {"type": "thinking", "thinking": "t", "signature": "s"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
        ],
    )
    history = (
        Message(role="user", content="hi"),
        Message(
            role="assistant", tool_calls=(ToolCall(id="toolu_1", name="get_weather"),), raw=raw
        ),
        Message(role="tool", tool_call_id="toolu_1", content="20C"),
        Message(role="tool", tool_call_id="toolu_2", content='{"error": {"kind": "x"}}'),
        Message(
            role="assistant",
            content="Done.",
            tool_calls=(ToolCall(id="t3", name="get_time", arguments={"tz": "UTC"}),),
        ),
        Message(role="assistant"),
    )
    messages = build_anthropic_request(_request(*history), model="c", max_tokens=1)["messages"]
    assert messages[1] == {"role": "assistant", "content": raw.items}
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "20C"},
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": '{"error": {"kind": "x"}}',
                "is_error": True,
            },
        ],
    }
    assert messages[3] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Done."},
            {"type": "tool_use", "id": "t3", "name": "get_time", "input": {"tz": "UTC"}},
        ],
    }
    assert len(messages) == 4


def test_parse_tool_use_thinking_raw_and_usage():
    response = parse_anthropic_message(fixture("anthropic/tool_use"))
    assert response.text == "Let me check."
    assert response.tool_calls == (
        ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"}),
    )
    assert (response.finish_reason, response.raw_finish_reason) == ("tool_calls", "tool_use")
    assert [item["type"] for item in response.message.raw.items] == ["thinking", "text", "tool_use"]
    assert response.usage.cache_write_by_ttl == {"5m": 1000, "1h": 500}
    assert (response.model, response.id) == ("claude-sonnet-4-5", "msg_1")


@pytest.mark.parametrize(
    ("stop", "finish"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "content_filter"),
        ("pause_turn", "other"),
        (None, "other"),
    ],
)
def test_parse_stop_reason_map(stop, finish):
    response = parse_anthropic_message({"content": [], "stop_reason": stop})
    assert response.finish_reason == finish
    assert response.raw_finish_reason == (stop or "")


def test_stream_assembler_input_json_deltas_and_thinking():
    assembler = AnthropicStreamAssembler(model="claude")
    deltas = [d for event in fixture("anthropic/stream") for d in assembler.feed(event)]
    assert deltas == [
        TextDelta(text="Checking"),
        ToolCallDelta(index=0, id="toolu_1", name="get_weather"),
        ToolCallDelta(index=0, arguments_delta='{"city": '),
        ToolCallDelta(index=0, arguments_delta='"Paris"}'),
        ToolCallDelta(index=1, id="toolu_2", name="get_time"),
    ]
    response = assembler.response()
    assert response.text == "Checking"
    assert response.tool_calls == (
        ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="toolu_2", name="get_time", arguments={}),
    )
    assert response.finish_reason == "tool_calls"
    assert (response.usage.input_tokens, response.usage.cached_input_tokens) == (50, 10)
    assert response.usage.output_tokens == 42
    thinking = response.message.raw.items[0]
    assert thinking == {"type": "thinking", "thinking": "Let me think", "signature": "sig1"}


def test_stream_assembler_text_in_block_start_and_bad_json():
    assembler = AnthropicStreamAssembler(model="claude")
    start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": "Hi"},
    }
    assert assembler.feed(start) == [TextDelta(text="Hi")]
    assert assembler.feed({"type": "ping"}) == []
    bad = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": "{"},
    }
    assert assembler.feed(bad) == [ToolCallDelta(index=0, arguments_delta="{")]
    with pytest.raises(ModelOutputInvalid):
        assembler.response()


def test_build_skips_empty_user_turns():
    request = _request(Message(role="user", content=""), Message(role="user", content="hi"))
    messages = build_anthropic_request(request, model="c", max_tokens=1)["messages"]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
