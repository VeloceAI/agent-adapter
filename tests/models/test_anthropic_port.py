"""AnthropicModel over the real SDK with an httpx2 MockTransport (transport path, R4 §11)."""

import json

import pytest

from agent_adapter.core.errors import (
    MissingDependency,
    ModelAuthError,
    ModelBadRequest,
    ModelError,
    ModelOutputInvalid,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import (
    LLMRequest,
    Message,
    OutputSchema,
    StreamDone,
    TextDelta,
    ToolCallDelta,
    ToolDef,
)
from agent_adapter.runtime.sync import run_sync
from tests.fakes.models import FakePromptCachePlugin
from tests.models.conftest import fixture
from tests.models.wire import RecordingWire, typed_sse

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from agent_adapter.models.anthropic import AnthropicModel, map_anthropic_error  # noqa: E402

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
TEXT = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": '{"answer": 42}'}],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


class Wire(RecordingWire):
    response_type = httpx2.Response


def sse(events):
    return typed_sse(httpx2.Response, events)


def client_for(wire):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(wire))
    return anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)


def _request(*messages, **fields):
    return LLMRequest(messages=messages or (Message(role="user", content="hi"),), **fields)


async def test_generate_wire_shape_plugin_and_parse():
    wire = Wire(fixture("anthropic/tool_use"))
    client = client_for(wire)

    def mark(kwargs):
        kwargs["system"][-1]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}

    plugin = FakePromptCachePlugin({}, edit=mark)
    model = AnthropicModel(
        "claude-test",
        client=client,
        cache_plugin=plugin,
        max_tokens=256,
        defaults={"service_tier": "auto"},
    )
    response = await model.generate(
        _request(
            Message(role="system", content="Be brief."),
            Message(role="user", content="hi"),
            tools=(TOOL,),
            reasoning="high",
            extra={"metadata": {"user_id": "u1"}},
        )
    )
    [body] = wire.bodies
    assert body["model"] == "claude-test" and body["max_tokens"] == 256
    assert body["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
    ]
    assert body["tools"][0]["input_schema"]["required"] == ["city"]
    assert body["tool_choice"] == {"type": "auto"}
    assert body["output_config"] == {"effort": "high"}
    assert body["service_tier"] == "auto" and body["metadata"] == {"user_id": "u1"}
    assert plugin.calls[0]["client"] is client
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert response.usage.cache_write_input_tokens == 1500
    assert model.cache_key_fields() == {"api": "", "plugin": plugin.key_fields()}


async def test_structured_output_on_the_wire_parsed_and_invalid():
    schema = OutputSchema(name="A", json_schema={"type": "object", "properties": {}})
    bad = {**TEXT, "content": [{"type": "text", "text": "nope"}]}
    wire = Wire(TEXT, bad)
    model = AnthropicModel("claude-test", client=client_for(wire))
    response = await model.generate(_request(output_schema=schema))
    assert wire.bodies[0]["output_config"]["format"]["type"] == "json_schema"
    assert response.parsed == {"answer": 42}
    with pytest.raises(ModelOutputInvalid):
        await model.generate(_request(output_schema=schema))


async def test_thinking_blocks_echo_on_the_next_turn():
    wire = Wire(fixture("anthropic/tool_use"), TEXT)
    model = AnthropicModel("claude-test", client=client_for(wire))
    first = await model.generate(_request(tools=(TOOL,)))
    history = (
        Message(role="user", content="hi"),
        first.message,
        Message(role="tool", tool_call_id="toolu_1", content="20C"),
    )
    await model.generate(LLMRequest(messages=history, tools=(TOOL,)))
    sent = wire.bodies[1]["messages"]
    assert sent[1]["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    assert sent[2]["content"][0]["tool_use_id"] == "toolu_1"


async def test_stream_input_json_deltas_and_done():
    wire = Wire(sse(fixture("anthropic/stream")))
    model = AnthropicModel("claude-test", client=client_for(wire))
    events = [e async for e in model.stream(_request(tools=(TOOL,)))]
    assert wire.bodies[0]["stream"] is True
    assert events[0] == TextDelta(text="Checking")
    assert ToolCallDelta(index=0, arguments_delta='"Paris"}') in events
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.response.tool_calls[0].arguments == {"city": "Paris"}
    assert done.response.usage.output_tokens == 42


async def test_generate_with_stream_true():
    wire = Wire(sse(fixture("anthropic/stream")))
    response = await AnthropicModel("claude", client=client_for(wire)).generate(
        _request(stream=True)
    )
    assert response.finish_reason == "tool_calls" and response.model == "claude-sonnet-4-5"


def _status(code, kind):
    def handler(request):
        body = {"type": "error", "error": {"type": kind, "message": "nope"}}
        headers = {"retry-after": "3", "request-id": "req_a"}
        return httpx2.Response(code, json=body, headers=headers)

    return handler


@pytest.mark.parametrize(
    ("code", "cls"),
    [
        (400, ModelBadRequest),
        (401, ModelAuthError),
        (403, ModelAuthError),
        (404, ModelBadRequest),
        (409, ModelBadRequest),
        (413, ModelBadRequest),
        (418, ModelBadRequest),
        (422, ModelBadRequest),
        (429, ModelRateLimited),
        (500, ModelUnavailable),
        (503, ModelUnavailable),
        (504, ModelUnavailable),  # anthropic 2.3 raises InternalServerError for 504
        (529, ModelUnavailable),
    ],
)
async def test_status_errors_map(code, cls):
    model = AnthropicModel("claude", client=client_for(_status(code, "x_error")))
    with pytest.raises(cls) as info:
        await model.generate(_request())
    err = info.value
    assert type(err) is cls
    assert (err.provider, err.status_code, err.retry_after, err.request_id) == (
        "anthropic",
        code,
        3.0,
        "req_a",
    )
    assert isinstance(err.__cause__, anthropic.APIStatusError)


async def test_connection_and_timeout_errors_map():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ModelUnavailable):
        await AnthropicModel("claude", client=client_for(refuse)).generate(_request())
    with pytest.raises(ModelTimeout):
        await AnthropicModel("claude", client=client_for(slow)).generate(_request())


async def test_stream_errors_are_mapped():
    model = AnthropicModel("claude", client=client_for(_status(529, "overloaded_error")))
    with pytest.raises(ModelUnavailable):
        _ = [e async for e in model.stream(_request())]


def _status_error(cls, code):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("x", response=httpx2.Response(code, request=request), body=None)


def test_dedicated_classes_without_a_status_route_map():
    deadline = map_anthropic_error(_status_error(anthropic.DeadlineExceededError, 504), model="m")
    busy = map_anthropic_error(_status_error(anthropic.ServiceUnavailableError, 503), model="m")
    assert type(deadline) is ModelTimeout and type(busy) is ModelUnavailable


def test_unmapped_and_missing_sdk(monkeypatch):
    import sys

    assert map_anthropic_error(KeyError("x"), model="m") is None
    assert type(map_anthropic_error(anthropic.AnthropicError("x"), model="m")) is ModelError
    assert map_anthropic_error(ModelError("x", provider="anthropic"), model="m") is None
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert map_anthropic_error(RuntimeError(), model="m") is None
    with pytest.raises(MissingDependency, match=r"agent-adapter\[anthropic\]"):
        AnthropicModel("claude")


def test_loop_local_clients_across_two_run_sync_calls(monkeypatch):
    made = []
    real = anthropic.AsyncAnthropic

    def factory(**kwargs):
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(Wire(TEXT)))
        client = real(http_client=http, **kwargs)
        made.append(client)
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", factory)
    model = AnthropicModel("claude", client_kwargs={"api_key": "test", "max_retries": 0})
    for _ in range(2):
        assert run_sync(model.generate(_request())).text == '{"answer": 42}'
    assert len(made) == 2 and made[0] is not made[1]
    assert all(client.is_closed() for client in made)


def _sse_error(kind):
    def handler(request):
        event = {"type": "error", "error": {"type": kind, "message": "mid-stream"}}
        body = f"event: error\ndata: {json.dumps(event)}\n\n"
        return httpx2.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    return handler


@pytest.mark.parametrize(
    ("kind", "cls"),
    [
        ("overloaded_error", ModelUnavailable),
        ("api_error", ModelUnavailable),
        ("rate_limit_error", ModelRateLimited),
        ("invalid_request_error", ModelBadRequest),
        ("authentication_error", ModelAuthError),
        ("permission_error", ModelAuthError),
        ("strange_error", ModelError),
    ],
)
async def test_mid_stream_error_events_are_typed_by_body(kind, cls):
    model = AnthropicModel("claude", client=client_for(_sse_error(kind)))
    with pytest.raises(cls) as info:
        _ = [e async for e in model.stream(_request())]
    assert type(info.value) is cls
    assert isinstance(info.value.__cause__, anthropic.APIStatusError)


async def test_fallback_hops_on_a_mid_stream_error_before_the_first_delta():
    from agent_adapter.models.fallback import FallbackModel
    from tests.fakes.models import ScriptedModel, text

    broken = AnthropicModel("claude", client=client_for(_sse_error("overloaded_error")))
    chain = FallbackModel([broken, ScriptedModel([text("ok")])])
    events = [e async for e in chain.stream(_request())]
    assert events[-1].response.text == "ok"


async def test_structured_output_schema_on_the_wire_matches_the_sdk_transform():
    from tests.models.test_anthropic_schema import NESTED

    wire = Wire(TEXT)
    model = AnthropicModel("claude-test", client=client_for(wire))
    await model.generate(_request(output_schema=OutputSchema(name="Report", json_schema=NESTED)))
    sent = wire.bodies[0]["output_config"]["format"]["schema"]
    assert sent["additionalProperties"] is False
    assert sent["properties"]["owner"]["additionalProperties"] is False
    assert sent == anthropic.transform_schema(NESTED)


async def test_missing_credentials_are_an_auth_error(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(Wire(TEXT)))
    client = anthropic.AsyncAnthropic(http_client=http, max_retries=0)
    with pytest.raises(ModelAuthError) as info:
        await AnthropicModel("claude", client=client).generate(_request())
    assert isinstance(info.value.__cause__, TypeError)
    assert map_anthropic_error(TypeError("unrelated"), model="m") is None


async def test_client_construction_errors_map_or_propagate(monkeypatch):
    def broken(**kwargs):
        raise anthropic.AnthropicError("bad credentials config")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", broken)
    with pytest.raises(ModelAuthError):
        await AnthropicModel("claude").generate(_request())

    def misuse(**kwargs):
        raise TypeError("unexpected keyword 'bogus'")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", misuse)
    with pytest.raises(TypeError, match="bogus"):
        await AnthropicModel("claude").generate(_request())


def test_the_port_warms_the_platform_cache_on_windows(warm_calls):
    AnthropicModel("claude-x", client=object())
    assert len(warm_calls) == 4  # every aliased / terse variant
