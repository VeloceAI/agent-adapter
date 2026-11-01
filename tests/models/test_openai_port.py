"""OpenAIModel over the real SDK with an httpx2 MockTransport (transport path, R4 §11)."""

import json

import pytest

from agent_adapter.core.errors import ConfigError, MissingDependency, ModelOutputInvalid
from agent_adapter.core.messages import (
    LLMRequest,
    Message,
    OutputSchema,
    StreamDone,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import ToolCall
from agent_adapter.runtime.sync import run_sync
from tests.fakes.models import FakePromptCachePlugin
from tests.models.conftest import fixture
from tests.models.wire import RecordingWire, typed_sse

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

from agent_adapter.models.openai import OpenAIModel  # noqa: E402

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
SCHEMA = OutputSchema(name="Answer", json_schema={"type": "object", "properties": {}})


class Wire(RecordingWire):
    response_type = httpx2.Response


def sse_chat(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx2.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


def sse_responses(events):
    return typed_sse(httpx2.Response, events)


def client_for(wire):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(wire))
    return openai.AsyncOpenAI(api_key="test", http_client=http, max_retries=0)


def _request(**fields):
    return LLMRequest(messages=(Message(role="user", content="hi"),), **fields)


async def test_responses_generate_wire_shape_plugin_defaults_and_extra():
    wire = Wire(fixture("openai/responses_tool_call"))
    client = client_for(wire)
    plugin = FakePromptCachePlugin({"prompt_cache_key": "k1", "prompt_cache_retention": "24h"})
    model = OpenAIModel(
        "gpt-5",
        client=client,
        cache_plugin=plugin,
        defaults={"temperature": 0.5, "store": False},
    )
    assert isinstance(model, LLMPort)
    response = await model.generate(
        _request(tools=(TOOL,), temperature=0.1, extra={"prompt_cache_retention": "in_memory"})
    )
    [body] = wire.bodies
    assert wire.paths == ["/v1/responses"]
    assert body["model"] == "gpt-5"
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["tools"][0]["name"] == "get_weather" and body["tools"][0]["strict"] is True
    assert body["temperature"] == 0.1 and body["store"] is False
    assert body["prompt_cache_key"] == "k1"
    assert body["prompt_cache_retention"] == "in_memory"  # request.extra merges last
    assert plugin.calls[0]["client"] is client and plugin.calls[0]["model"] == "gpt-5"
    assert "store" in plugin.calls[0]["kwargs"]  # defaults are merged before the plug-in
    assert response.tool_calls == (
        ToolCall(id="call_b", name="get_weather", arguments={"city": "Oslo"}),
    )
    assert response.latency_ms > 0
    assert model.cache_key_fields() == {"api": "responses", "plugin": plugin.key_fields()}


async def test_plugin_for_other_provider_is_skipped():
    wire = Wire(fixture("openai/responses_text"))
    plugin = FakePromptCachePlugin({"prompt_cache_key": "k"}, providers=("gemini",))
    await OpenAIModel("gpt-5", client=client_for(wire), cache_plugin=plugin).generate(_request())
    assert "prompt_cache_key" not in wire.bodies[0] and plugin.calls == []


async def test_chat_generate_wire_shape_and_structured_output():
    wire = Wire(fixture("openai/chat_text"))
    model = OpenAIModel("gpt-4.1", api="chat", client=client_for(wire))
    response = await model.generate(_request(output_schema=SCHEMA, model="gpt-4.1-mini"))
    [body] = wire.bodies
    assert wire.paths == ["/v1/chat/completions"]
    assert body["model"] == "gpt-4.1-mini"
    assert body["response_format"]["type"] == "json_schema"
    assert response.parsed == {"answer": 42}
    assert response.usage.input_tokens == 50
    assert model.cache_key_fields() == {"api": "chat", "plugin": None}


async def test_chat_tools_rendered_nested_and_strict_tools_off():
    wire = Wire(fixture("openai/chat_tool_calls"))
    model = OpenAIModel("gpt-4.1", api="chat", client=client_for(wire), strict_tools=False)
    response = await model.generate(_request(tools=(TOOL,)))
    tool = wire.bodies[0]["tools"][0]
    assert tool["type"] == "function" and tool["function"]["strict"] is False
    assert response.finish_reason == "tool_calls"


async def test_structured_output_not_json_raises_output_invalid():
    reply = fixture("openai/chat_text")
    reply["choices"][0]["message"]["content"] = "not json"
    model = OpenAIModel("gpt-4.1", api="chat", client=client_for(Wire(reply)))
    with pytest.raises(ModelOutputInvalid):
        await model.generate(_request(output_schema=SCHEMA))


async def test_cache_write_ttl_filled_from_request_when_usage_has_no_split():
    reply = fixture("openai/responses_tool_call")
    reply["usage"]["input_tokens_details"]["cache_write_tokens"] = 70
    plugin = FakePromptCachePlugin({"prompt_cache_options": {"mode": "implicit", "ttl": "12h"}})
    model = OpenAIModel("gpt-5.6", client=client_for(Wire(reply)), cache_plugin=plugin)
    usage = (await model.generate(_request())).usage
    assert usage.cache_write_by_ttl == {"12h": 70}
    assert usage.input_tokens == 500 - 256 - 70


async def test_cache_write_split_from_usage_wins_over_request_ttl():
    plugin = FakePromptCachePlugin({"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}})
    wire = Wire(fixture("openai/responses_text"))
    model = OpenAIModel("gpt-5.6", client=client_for(wire), cache_plugin=plugin)
    usage = (await model.generate(_request())).usage
    assert wire.bodies[0]["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert usage.cache_write_by_ttl == {"30m": 3000, "12h": 4000}


async def test_no_ttl_without_cache_writes_or_without_options():
    model = OpenAIModel(
        "gpt-4.1", api="chat", client=client_for(Wire(fixture("openai/chat_tool_calls")))
    )
    assert (await model.generate(_request())).usage.cache_write_by_ttl == {}


async def test_chat_stream_deltas_wire_flags_and_done():
    wire = Wire(sse_chat(fixture("openai/chat_stream")))
    model = OpenAIModel("gpt-4.1", api="chat", client=client_for(wire))
    events = [e async for e in model.stream(_request(tools=(TOOL,)))]
    body = wire.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert events[0] == TextDelta(text="Checking")
    assert ToolCallDelta(index=0, arguments_delta='ty":"Paris"}') in events
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert [c.name for c in done.response.tool_calls] == ["get_weather", "get_time"]
    assert done.response.usage.cached_input_tokens == 2


async def test_generate_with_stream_true_returns_assembled_response():
    wire = Wire(sse_responses(fixture("openai/responses_stream")))
    model = OpenAIModel("gpt-5", client=client_for(wire))
    response = await model.generate(_request(stream=True))
    assert wire.bodies[0]["stream"] is True
    assert response.text == "Let me check."
    assert response.tool_calls[0].arguments == {"city": "Oslo"}
    assert response.message.raw is not None and response.message.raw.api == "responses"


async def test_responses_stream_keys_arguments_by_item_id():
    wire = Wire(sse_responses(fixture("openai/responses_stream")))
    events = [e async for e in OpenAIModel("gpt-5", client=client_for(wire)).stream(_request())]
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert [d.index for d in deltas] == [0, 1, 1, 0, 0]


async def test_responses_stream_echoes_raw_items_on_the_next_turn():
    wire = Wire(fixture("openai/responses_tool_call"), fixture("openai/responses_text"))
    model = OpenAIModel("gpt-5", client=client_for(wire))
    first = await model.generate(_request())
    history = (
        Message(role="user", content="hi"),
        first.message,
        Message(role="tool", tool_call_id="call_b", content="cold"),
    )
    await model.generate(LLMRequest(messages=history))
    second = wire.bodies[1]["input"]
    assert [item.get("type", item.get("role")) for item in second] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
    ]


def test_loop_local_clients_across_two_run_sync_calls(monkeypatch):
    made = []
    real = openai.AsyncOpenAI

    def factory(**kwargs):
        wire = Wire(fixture("openai/responses_text"))
        client = real(
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wire)), **kwargs
        )
        made.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    model = OpenAIModel("gpt-5", client_kwargs={"api_key": "test", "max_retries": 0})
    for _ in range(2):
        assert run_sync(model.generate(_request())).text == "Hello there"
    assert len(made) == 2 and made[0] is not made[1]
    assert all(client.is_closed() for client in made)


def test_missing_sdk_raises_missing_dependency(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[openai\]"):
        OpenAIModel("gpt-5")
    OpenAIModel("gpt-5", client=object())  # a host client needs no import


class _FakeStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def close(self):
        self.closed = True


async def test_early_stream_exit_closes_the_sdk_stream():
    stream = _FakeStream(fixture("openai/responses_stream"))

    class Responses:
        async def create(self, **kwargs):
            return stream

    class Client:
        responses = Responses()

    events = OpenAIModel("gpt-5", client=Client()).stream(_request())
    assert await anext(events) == TextDelta(text="Let me ")
    await events.aclose()
    assert stream.closed


def test_unknown_api_is_a_config_error():
    with pytest.raises(ConfigError, match="batch"):
        OpenAIModel("gpt-5", api="batch", client=object())  # type: ignore[arg-type]


def test_the_port_warms_the_platform_cache_on_windows(warm_calls):
    OpenAIModel("gpt-4.1", client=object())
    assert len(warm_calls) == 4  # every aliased / terse variant
