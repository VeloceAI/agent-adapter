import pytest

from agent_adapter.core.errors import ConfigError
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
from agent_adapter.models._gemini_convert import (
    GeminiStreamAssembler,
    build_gemini_request,
    parse_gemini_response,
)
from tests.models.conftest import fixture

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={
        "type": "object",
        "title": "Args",
        "properties": {"city": {"type": "string", "default": "Rome"}},
    },
)


def _request(*messages, **fields):
    return LLMRequest(messages=messages or (Message(role="user", content="hi"),), **fields)


def test_build_minimal_disables_afc():
    assert build_gemini_request(_request(), model="gemini-2.5-flash") == {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "config": {"automatic_function_calling": {"disable": True}},
    }


def test_build_system_tools_schema_and_generation():
    schema = OutputSchema(name="A", json_schema={"type": "object", "title": "A", "properties": {}})
    request = _request(
        Message(role="system", content="One."),
        Message(role="system", content="Two."),
        Message(role="user", content="hi"),
        tools=(TOOL,),
        output_schema=schema,
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=50,
        seed=3,
        stop=("X",),
        reasoning="minimal",
    )
    config = build_gemini_request(request, model="m")["config"]
    assert config["system_instruction"] == "One.\n\nTwo."
    [declaration] = config["tools"][0]["function_declarations"]
    assert declaration == {
        "name": "get_weather",
        "description": "Weather.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": 'Default: "Rome".'}},
        },
    }
    assert config["tool_config"] == {"function_calling_config": {"mode": "AUTO"}}
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == {"type": "object", "properties": {}}
    assert (config["temperature"], config["top_p"], config["max_output_tokens"]) == (0.2, 0.8, 50)
    assert (config["seed"], config["stop_sequences"]) == (3, ["X"])
    assert config["thinking_config"] == {"thinking_level": "MINIMAL"}


@pytest.mark.parametrize(
    ("choice", "calling"),
    [
        ("none", {"mode": "NONE"}),
        ("required", {"mode": "ANY"}),
        ("get_weather", {"mode": "ANY", "allowed_function_names": ["get_weather"]}),
    ],
)
def test_build_tool_choice(choice, calling):
    config = build_gemini_request(_request(tools=(TOOL,), tool_choice=choice), model="m")["config"]
    assert config["tool_config"]["function_calling_config"] == calling


def test_build_echoes_raw_parts_and_answers_parallel_calls_in_one_turn():
    raw = ProviderRaw(
        provider="gemini",
        items=[
            {
                "function_call": {"id": "fc-1", "name": "get_weather", "args": {}},
                "thought_signature": "c2ln",
            },
            {"functionCall": {"name": "get_time", "args": {}}},
        ],
    )
    history = (
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            tool_calls=(ToolCall(id="fc-1", name="get_weather"), ToolCall(name="get_time")),
            raw=raw,
        ),
        Message(role="tool", tool_call_id="fc-1", content='{"temp": 20}'),
        Message(
            role="tool", tool_call_id="call_x", name="get_time", content='{"error": {"kind": "k"}}'
        ),
        Message(role="user", content="thanks"),
    )
    contents = build_gemini_request(_request(*history), model="m")["contents"]
    assert contents[1] == {"role": "model", "parts": raw.items}
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": "get_weather",
                    "response": {"output": {"temp": 20}},
                    "id": "fc-1",
                }
            },
            {"function_response": {"name": "get_time", "response": {"error": {"kind": "k"}}}},
        ],
    }
    assert contents[3] == {"role": "user", "parts": [{"text": "thanks"}]}


def test_build_renders_foreign_assistant_messages():
    history = (
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="Checking.",
            tool_calls=(ToolCall(id="call_1", name="t", arguments={"a": 1}),),
            raw=ProviderRaw(provider="openai", api="chat"),
        ),
        Message(role="tool", tool_call_id="call_1", content="plain"),
        Message(role="assistant"),  # empty turns are dropped
    )
    contents = build_gemini_request(_request(*history), model="m")["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"text": "Checking."},
            {"function_call": {"name": "t", "args": {"a": 1}, "id": "call_1"}},
        ],
    }
    assert contents[2]["parts"][0]["function_response"] == {
        "name": "t",
        "response": {"output": "plain"},
        "id": "call_1",
    }
    assert len(contents) == 3


def test_build_tool_message_without_name_fails():
    with pytest.raises(ConfigError, match="tool name"):
        build_gemini_request(
            _request(Message(role="tool", tool_call_id="x", content="1")), model="m"
        )


def test_parse_function_call_thoughts_and_raw():
    response = parse_gemini_response(fixture("gemini/function_call"))
    assert response.tool_calls == (
        ToolCall(id="fc-1", name="get_weather", arguments={"city": "Paris"}),
    )
    assert response.text == ""  # thought text is not reply text
    assert (response.finish_reason, response.raw_finish_reason) == ("tool_calls", "STOP")
    assert (response.model, response.id) == ("gemini-2.5-flash", "abc")
    raw = response.message.raw
    assert (
        raw is not None and raw.provider == "gemini" and raw.items[1]["thoughtSignature"] == "c2ln"
    )
    assert response.usage.reasoning_tokens == 40


def test_parse_text_snake_case():
    response = parse_gemini_response(fixture("gemini/text"))
    assert response.text == '{"answer": 42}'
    assert response.finish_reason == "stop"


@pytest.mark.parametrize(
    ("raw", "finish"),
    [
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("SPII", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "error"),
        ("UNEXPECTED_TOOL_CALL", "error"),
        ("TOO_MANY_TOOL_CALLS", "other"),
    ],
)
def test_parse_finish_reason_map(raw, finish):
    data = {"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": raw}]}
    response = parse_gemini_response(data)
    assert (response.finish_reason, response.raw_finish_reason) == (finish, raw)


def test_parse_blocked_prompt_and_empty():
    blocked = parse_gemini_response({"promptFeedback": {"blockReason": "SAFETY"}})
    assert (blocked.finish_reason, blocked.raw_finish_reason) == ("content_filter", "SAFETY")
    assert parse_gemini_response({}).finish_reason == "other"


def test_stream_assembler_whole_calls_per_chunk():
    assembler = GeminiStreamAssembler()
    deltas = [d for chunk in fixture("gemini/stream") for d in assembler.feed(chunk)]
    assert deltas == [
        TextDelta(text="Hel"),
        TextDelta(text="lo"),
        ToolCallDelta(index=0, name="get_weather", arguments_delta='{"city":"Rome"}'),
        ToolCallDelta(index=1, name="get_time", arguments_delta="{}"),
    ]
    response = assembler.response()
    assert response.text == "Hello"
    assert [c.name for c in response.tool_calls] == ["get_weather", "get_time"]
    assert response.finish_reason == "tool_calls"
    assert (response.model, response.id, response.usage.output_tokens) == (
        "gemini-2.5-flash",
        "s1",
        9,
    )
    assert len(response.message.raw.items) == 4


def test_stream_assembler_blocked_and_empty_chunks():
    assembler = GeminiStreamAssembler()
    assert assembler.feed({"promptFeedback": {"blockReason": "OTHER"}}) == []
    response = assembler.response()
    assert (response.finish_reason, response.raw_finish_reason) == ("content_filter", "OTHER")


def test_build_skips_empty_user_turns():
    request = _request(Message(role="user", content=""), Message(role="user", content="hi"))
    contents = build_gemini_request(request, model="m")["contents"]
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]
