import pytest
from pydantic import BaseModel, ValidationError

from agent_adapter.core.errors import ModelContentFiltered, ModelOutputInvalid
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    OutputSchema,
    ProviderRaw,
    StreamDone,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.core.types import ToolCall, Usage


class Answer(BaseModel):
    city: str
    temp: int


def response(**changes) -> LLMResponse:
    fields = {"provider": "openai", "model": "gpt", "message": Message(role="assistant")}
    return LLMResponse(**{**fields, **changes})


def test_message_defaults_and_frozen():
    msg = Message(role="assistant", tool_calls=[ToolCall(name="t")])
    assert msg.tool_calls == (ToolCall(name="t"),)
    assert (msg.content, msg.tool_call_id, msg.name, msg.raw, msg.cache_breakpoint) == (
        "",
        "",
        "",
        None,
        False,
    )
    with pytest.raises(ValidationError):
        Message(role="robot")
    with pytest.raises(ValidationError):
        msg.content = "x"
    raw = ProviderRaw(provider="gemini", items=[{"thought_signature": "x"}])
    assert Message(role="assistant", raw=raw).raw.api == ""


def test_request_defaults():
    req = LLMRequest(messages=[Message(role="user", content="hi")])
    assert req.model is None and req.tools == () and req.tool_choice == "auto"
    assert req.response_cache and not req.stream and req.extra == {} and req.metadata == {}
    assert LLMRequest(messages=(), tool_choice="search").tool_choice == "search"
    tool = ToolDef(name="t", description="d", parameters={"type": "object"})
    schema = OutputSchema(name="answer", json_schema=Answer.model_json_schema())
    req = LLMRequest(messages=(), tools=[tool], output_schema=schema, reasoning="low")
    assert req.tools[0].strict and req.output_schema.strict


def test_response_defaults():
    res = response()
    assert res.usage == Usage() and res.finish_reason == "stop" and not res.cached


def test_parse_prefers_parsed_then_text():
    parsed = response(parsed={"city": "Pune", "temp": 31}, text="ignored").parse(Answer)
    assert parsed == Answer(city="Pune", temp=31)
    assert response(text='{"city": "Pune", "temp": 30}').parse(Answer).temp == 30


def test_parse_raises_model_output_invalid():
    with pytest.raises(ModelOutputInvalid) as raised:
        response(text="not json", id="resp_1").parse(Answer)
    assert raised.value.provider == "openai" and raised.value.request_id == "resp_1"
    assert isinstance(raised.value.__cause__, ValidationError)
    with pytest.raises(ModelOutputInvalid):
        response(parsed={"city": "Pune"}).parse(Answer)


def test_parse_raises_content_filtered():
    with pytest.raises(ModelContentFiltered) as raised:
        response(finish_reason="content_filter", text="{}").parse(Answer)
    assert raised.value.model == "gpt"


def test_stream_events():
    assert TextDelta(text="a").type == "text"
    delta = ToolCallDelta(index=0, arguments_delta='{"a"')
    assert delta.type == "tool_call" and delta.id == ""
    assert StreamDone(response=response()).type == "done"
