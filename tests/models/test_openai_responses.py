import logging

import pytest

from agent_adapter.core.errors import ModelError, ModelRateLimited, ModelUnavailable
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
from agent_adapter.models._openai_responses import (
    ResponsesStreamAssembler,
    build_responses_request,
    parse_response,
)
from tests.models.conftest import fixture

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def _request(*messages, **fields):
    return LLMRequest(messages=messages or (Message(role="user", content="hi"),), **fields)


def test_build_minimal():
    assert build_responses_request(_request(), model="gpt-5") == {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "hi"}],
    }


def test_build_flat_tools_choice_schema_and_reasoning():
    schema = OutputSchema(name="Answer", json_schema={"type": "object", "properties": {}})
    kwargs = build_responses_request(
        _request(
            tools=(TOOL,),
            tool_choice="get_weather",
            parallel_tool_calls=True,
            output_schema=schema,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=10,
            reasoning="high",
        ),
        model="m",
        strict_tools=False,
    )
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather.",
            "parameters": TOOL.parameters,
            "strict": False,
        }
    ]
    assert kwargs["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert kwargs["parallel_tool_calls"] is True
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["name"] == "Answer"
    assert kwargs["text"]["format"]["schema"]["additionalProperties"] is False
    assert kwargs["reasoning"] == {"effort": "high"}
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["max_output_tokens"]) == (0.0, 1.0, 10)


def test_build_tool_choice_keywords():
    kwargs = build_responses_request(_request(tools=(TOOL,), tool_choice="required"), model="m")
    assert kwargs["tool_choice"] == "required"
    assert kwargs["tools"][0]["strict"] is True


def test_build_drops_seed_and_stop_with_one_warning(caplog, monkeypatch):
    monkeypatch.setattr(_convert, "_warned", set())
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            kwargs = build_responses_request(_request(seed=1, stop=("x",)), model="m")
    assert "seed" not in kwargs and "stop" not in kwargs
    assert caplog.text.count("'seed'") == 1 and caplog.text.count("'stop'") == 1


def test_build_history_items_and_raw_echo():
    raw = ProviderRaw(
        provider="openai",
        api="responses",
        items=[
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "call_id": "c1"},
        ],
    )
    foreign = ProviderRaw(provider="gemini", items=[{"text": "x"}])
    history = (
        Message(role="system", content="S"),
        Message(role="user", content="U"),
        Message(role="assistant", tool_calls=(ToolCall(id="c1", name="t"),), raw=raw),
        Message(role="tool", tool_call_id="c1", content="out"),
        Message(
            role="assistant",
            content="A",
            tool_calls=(ToolCall(id="c2", name="t", arguments={"a": 1}),),
            raw=foreign,
        ),
    )
    assert build_responses_request(_request(*history), model="m")["input"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "out"},
        {"role": "assistant", "content": "A"},
        {"type": "function_call", "call_id": "c2", "name": "t", "arguments": '{"a":1}'},
    ]


def test_parse_function_call_with_reasoning_raw():
    response = parse_response(fixture("openai/responses_tool_call"))
    assert response.tool_calls == (
        ToolCall(id="call_b", name="get_weather", arguments={"city": "Oslo"}),
    )
    assert (response.finish_reason, response.raw_finish_reason) == ("tool_calls", "completed")
    raw = response.message.raw
    assert raw is not None and (raw.provider, raw.api) == ("openai", "responses")
    assert [item["type"] for item in raw.items] == ["reasoning", "function_call"]
    assert response.usage.reasoning_tokens == 10


def test_parse_text_joins_parts():
    response = parse_response(fixture("openai/responses_text"))
    assert response.text == "Hello there"
    assert response.finish_reason == "stop"
    assert response.usage.cache_write_by_ttl == {"30m": 3000, "12h": 4000}


@pytest.mark.parametrize(
    ("status", "reason", "finish", "raw_finish"),
    [
        ("incomplete", "max_output_tokens", "length", "max_output_tokens"),
        ("incomplete", "content_filter", "content_filter", "content_filter"),
        ("incomplete", "", "other", "incomplete"),
        ("failed", "", "error", "failed"),
        ("cancelled", "", "other", "cancelled"),
    ],
)
def test_parse_status_map(status, reason, finish, raw_finish):
    data = {"model": "m", "status": status, "output": []}
    if reason:
        data["incomplete_details"] = {"reason": reason}
    response = parse_response(data)
    assert (response.finish_reason, response.raw_finish_reason) == (finish, raw_finish)


def test_parse_refusal_part():
    data = {
        "model": "m",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}],
    }
    response = parse_response(data)
    assert (response.finish_reason, response.text) == ("content_filter", "No.")


def test_stream_assembler_keys_by_item_id():
    assembler = ResponsesStreamAssembler(model="gpt-5")
    deltas = [d for event in fixture("openai/responses_stream") for d in assembler.feed(event)]
    assert deltas == [
        TextDelta(text="Let me "),
        TextDelta(text="check."),
        ToolCallDelta(index=0, id="call_b", name="get_weather"),
        ToolCallDelta(index=1, id="call_c", name="get_time"),
        ToolCallDelta(index=1, arguments_delta="{}"),
        ToolCallDelta(index=0, arguments_delta='{"city":'),
        ToolCallDelta(index=0, arguments_delta='"Oslo"}'),
    ]
    response = assembler.response()
    assert response.text == "Let me check."
    assert [c.name for c in response.tool_calls] == ["get_weather", "get_time"]
    assert response.usage.cache_write_input_tokens == 5


def test_stream_assembler_ignores_other_items_and_needs_terminal_event():
    assembler = ResponsesStreamAssembler(model="m")
    assert assembler.feed({"type": "response.output_item.added", "item": {"type": "message"}}) == []
    assert assembler.feed({"type": "response.output_text.delta", "delta": ""}) == []
    with pytest.raises(ModelUnavailable, match="ended"):
        assembler.response()


@pytest.mark.parametrize(
    ("code", "cls"),
    [
        ("rate_limit_exceeded", ModelRateLimited),
        ("server_error", ModelUnavailable),
        ("", ModelError),
    ],
)
def test_stream_error_event(code, cls):
    with pytest.raises(cls) as info:
        ResponsesStreamAssembler(model="m").feed({"type": "error", "code": code, "message": "boom"})
    assert type(info.value) is cls and "boom" in str(info.value)
