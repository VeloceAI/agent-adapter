import pytest

from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import (
    LLMRequest,
    Message,
    OutputSchema,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.core.types import ToolCall
from agent_adapter.models._openai_chat import (
    ChatStreamAssembler,
    build_chat_request,
    parse_chat_completion,
)
from tests.models.conftest import fixture

PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "units": {"type": "string", "default": "c"}},
    "required": ["city"],
}
WEATHER = ToolDef(name="get_weather", description="Weather.", parameters=PARAMS)


def _request(**fields):
    return LLMRequest(messages=(Message(role="user", content="hi"),), **fields)


def test_build_minimal_request_sends_only_set_fields():
    assert build_chat_request(_request(), model="gpt-4.1") == {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_build_tools_strict_and_downgraded():
    loose = ToolDef(name="loose", description="L.", parameters=PARAMS, strict=False)
    kwargs = build_chat_request(_request(tools=(WEATHER, loose)), model="m")
    strict, plain = kwargs["tools"]
    assert strict["type"] == "function"
    assert strict["function"]["strict"] is True
    assert strict["function"]["parameters"]["required"] == ["city", "units"]
    assert strict["function"]["parameters"]["additionalProperties"] is False
    assert plain["function"]["strict"] is False
    assert plain["function"]["parameters"]["required"] == ["city"]
    assert kwargs["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in kwargs
    off = build_chat_request(_request(tools=(WEATHER,)), model="m", strict_tools=False)
    assert off["tools"][0]["function"]["strict"] is False


@pytest.mark.parametrize(
    ("choice", "wire"),
    [
        ("none", "none"),
        ("required", "required"),
        ("get_weather", {"type": "function", "function": {"name": "get_weather"}}),
    ],
)
def test_build_tool_choice(choice, wire):
    kwargs = build_chat_request(
        _request(tools=(WEATHER,), tool_choice=choice, parallel_tool_calls=False), model="m"
    )
    assert kwargs["tool_choice"] == wire
    assert kwargs["parallel_tool_calls"] is False


def test_build_without_tools_omits_tool_choice():
    assert "tool_choice" not in build_chat_request(_request(tool_choice="required"), model="m")


def test_build_output_schema_sampling_reasoning_stop_seed():
    schema = OutputSchema(name="Answer", json_schema={"type": "object", "properties": {}})
    kwargs = build_chat_request(
        _request(
            output_schema=schema,
            temperature=0.1,
            top_p=0.9,
            max_output_tokens=64,
            seed=7,
            stop=("END",),
            reasoning="minimal",
        ),
        model="m",
    )
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["max_completion_tokens"]) == (
        0.1,
        0.9,
        64,
    )
    assert (kwargs["seed"], kwargs["stop"], kwargs["reasoning_effort"]) == (7, ["END"], "minimal")


def test_build_history_with_tool_round_trip():
    history = (
        Message(role="system", content="Be brief."),
        Message(role="user", content="Weather?"),
        Message(
            role="assistant",
            tool_calls=(ToolCall(id="call_a", name="get_weather", arguments={"city": "Paris"}),),
        ),
        Message(role="tool", tool_call_id="call_a", name="get_weather", content="21C"),
        Message(role="assistant", content="21C in Paris."),
    )
    messages = build_chat_request(LLMRequest(messages=history), model="m")["messages"]
    assert messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "21C"},
        {"role": "assistant", "content": "21C in Paris."},
    ]


def test_parse_tool_calls_usage_and_finish():
    response = parse_chat_completion(fixture("openai/chat_tool_calls"))
    assert response.provider == "openai"
    assert response.model == "gpt-4.1-2025-04-14" and response.id == "chatcmpl-1"
    assert response.tool_calls == (
        ToolCall(id="call_a", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="call_b", name="get_time", arguments={}),
    )
    assert (response.finish_reason, response.raw_finish_reason) == ("tool_calls", "tool_calls")
    assert response.usage.input_tokens == 76 and response.usage.reasoning_tokens == 4
    assert response.message.tool_calls == response.tool_calls
    assert response.text == ""


def test_parse_text():
    response = parse_chat_completion(fixture("openai/chat_text"))
    assert response.text == '{"answer": 42}'
    assert response.message == Message(role="assistant", content='{"answer": 42}')
    assert response.finish_reason == "stop"


@pytest.mark.parametrize(
    ("raw", "finish"),
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("function_call", "tool_calls"),
        ("content_filter", "content_filter"),
        ("weird", "other"),
    ],
)
def test_parse_finish_reason_map(raw, finish):
    data = {"model": "m", "choices": [{"finish_reason": raw, "message": {"content": "x"}}]}
    response = parse_chat_completion(data)
    assert (response.finish_reason, response.raw_finish_reason) == (finish, raw)


def test_parse_refusal_is_content_filter():
    data = {"model": "m", "choices": [{"finish_reason": "stop", "message": {"refusal": "No."}}]}
    response = parse_chat_completion(data)
    assert (response.finish_reason, response.text, response.raw_finish_reason) == (
        "content_filter",
        "No.",
        "stop",
    )


def test_parse_empty_and_custom_tool_calls():
    assert parse_chat_completion({"model": "m", "choices": []}).finish_reason == "other"
    data = {
        "model": "m",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{"id": "x", "type": "custom", "custom": {}}]},
            }
        ],
    }
    assert parse_chat_completion(data).tool_calls == ()


def test_parse_bad_arguments_raise_output_invalid():
    data = {
        "model": "m",
        "choices": [
            {"message": {"tool_calls": [{"id": "x", "function": {"name": "t", "arguments": "{"}}]}}
        ],
    }
    with pytest.raises(ModelOutputInvalid):
        parse_chat_completion(data)


def test_stream_assembler_keys_argument_deltas_by_index():
    assembler = ChatStreamAssembler()
    deltas = [d for chunk in fixture("openai/chat_stream") for d in assembler.feed(chunk)]
    assert deltas[0] == TextDelta(text="Checking")
    assert deltas[1] == ToolCallDelta(index=0, id="call_a", name="get_weather")
    assert ToolCallDelta(index=0, arguments_delta='{"ci') in deltas
    assert ToolCallDelta(index=1, id="call_b", name="get_time", arguments_delta="{}") in deltas
    response = assembler.response()
    assert response.text == "Checking"
    assert response.tool_calls == (
        ToolCall(id="call_a", name="get_weather", arguments={"city": "Paris"}),
        ToolCall(id="call_b", name="get_time", arguments={}),
    )
    assert response.finish_reason == "tool_calls"
    assert (response.usage.input_tokens, response.usage.cached_input_tokens) == (8, 2)
    assert (response.id, response.model) == ("c", "gpt-4.1")


def test_stream_assembler_refusal_other_choices_and_no_usage():
    assembler = ChatStreamAssembler()
    assembler.feed({"choices": [{"index": 1, "delta": {"content": "ignored"}}]})
    assembler.feed({"choices": [{"index": 0, "delta": {"refusal": "No"}}]})
    assembler.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    response = assembler.response()
    assert (response.finish_reason, response.text) == ("content_filter", "No")
    assert response.usage.prompt_tokens == 0
