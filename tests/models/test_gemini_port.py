"""GeminiModel over google-genai with an httpx MockTransport (transport path, R4 §11)."""

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
from tests.models.wire import RecordingWire

genai = pytest.importorskip("google.genai")
httpx = pytest.importorskip("httpx")

from google.genai import errors, types  # noqa: E402

from agent_adapter.models.gemini import GeminiModel, map_gemini_error  # noqa: E402

TOOL = ToolDef(
    name="get_weather",
    description="Weather.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
SCHEMA = OutputSchema(
    name="A", json_schema={"type": "object", "properties": {"answer": {"type": "integer"}}}
)


class Wire(RecordingWire):
    response_type = httpx.Response


def sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def options(wire):
    return types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)),
        retry_options=types.HttpRetryOptions(attempts=1),
    )


def client_for(wire):
    return genai.Client(api_key="test", http_options=options(wire))


def _request(*messages, **fields):
    return LLMRequest(messages=messages or (Message(role="user", content="hi"),), **fields)


async def test_generate_wire_shape_tools_schema_thinking_and_plugin():
    wire = Wire(fixture("gemini/function_call"))
    client = client_for(wire)

    def cached(kwargs):
        kwargs["config"]["cached_content"] = "cachedContents/abc"

    plugin = FakePromptCachePlugin({}, edit=cached)
    model = GeminiModel(
        "gemini-2.5-flash",
        client=client,
        cache_plugin=plugin,
        thinking={"include_thoughts": True},
        defaults={"temperature": 0.9, "candidate_count": 1},
    )
    request = _request(
        Message(role="system", content="Be brief."),
        Message(role="user", content="hi"),
        tools=(TOOL,),
        temperature=0.1,
        reasoning="low",
        extra={"labels": {"run": "r1"}},
    )
    response = await model.generate(request)
    [body] = wire.bodies
    assert "/models/gemini-2.5-flash:generateContent" in wire.urls[0]
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    declaration = body["tools"][0]["functionDeclarations"][0]
    assert declaration["parameters_json_schema"]["required"] == ["city"]  # genai sends it as is
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    generation = body["generationConfig"]
    assert generation["temperature"] == 0.1 and generation["candidateCount"] == 1
    # genai forwards nested config dicts as given; proto JSON accepts either key style
    assert generation["thinkingConfig"] == {"include_thoughts": True, "thinking_level": "LOW"}
    assert body["cachedContent"] == "cachedContents/abc"
    assert body["labels"] == {"run": "r1"}
    assert plugin.calls[0]["client"] is client
    assert response.tool_calls[0].name == "get_weather" and response.tool_calls[0].id == "fc-1"
    assert response.finish_reason == "tool_calls"
    assert response.usage.cached_input_tokens == 2048
    assert model.cache_key_fields() == {"api": "", "plugin": plugin.key_fields()}


async def test_structured_output_on_the_wire_and_parsed():
    wire = Wire(fixture("gemini/text"))
    model = GeminiModel("gemini-2.5-flash", client=client_for(wire))
    response = await model.generate(_request(output_schema=SCHEMA))
    generation = wire.bodies[0]["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseJsonSchema"] == SCHEMA.json_schema
    assert response.parsed == {"answer": 42}


async def test_structured_output_without_json_is_invalid():
    reply = fixture("gemini/text")
    reply["candidates"][0]["content"]["parts"] = [{"text": "nope"}]
    model = GeminiModel("gemini-2.5-flash", client=client_for(Wire(reply)))
    with pytest.raises(ModelOutputInvalid):
        await model.generate(_request(output_schema=SCHEMA))


async def test_thought_signatures_round_trip_on_the_wire():
    wire = Wire(fixture("gemini/function_call"), fixture("gemini/text"))
    model = GeminiModel("gemini-2.5-flash", client=client_for(wire))
    first = await model.generate(_request(tools=(TOOL,)))
    history = (
        Message(role="user", content="hi"),
        first.message,
        Message(role="tool", tool_call_id="fc-1", content='{"temp": 21}'),
    )
    await model.generate(LLMRequest(messages=history, tools=(TOOL,)))
    contents = wire.bodies[1]["contents"]
    signed = contents[1]["parts"][1]
    assert signed["thoughtSignature"] == "c2ln"
    assert signed["functionCall"]["id"] == "fc-1"
    reply = contents[2]["parts"][0]["functionResponse"]
    assert reply == {"id": "fc-1", "name": "get_weather", "response": {"output": {"temp": 21}}}


async def test_stream_whole_calls_and_done():
    wire = Wire(sse(fixture("gemini/stream")))
    model = GeminiModel("gemini-2.5-flash", client=client_for(wire))
    events = [e async for e in model.stream(_request())]
    assert ":streamGenerateContent" in wire.urls[0]
    assert events[:2] == [TextDelta(text="Hel"), TextDelta(text="lo")]
    assert events[2] == ToolCallDelta(
        index=0, name="get_weather", arguments_delta='{"city":"Rome"}'
    )
    done = events[-1]
    assert isinstance(done, StreamDone) and done.response.text == "Hello"
    assert done.response.message.raw.items[2]["thought_signature"] == "c2ln"
    assert done.response.usage.output_tokens == 9


async def test_generate_with_stream_true():
    wire = Wire(sse(fixture("gemini/stream")))
    model = GeminiModel("gemini-2.5-flash", client=client_for(wire).aio)  # client.aio works too
    response = await model.generate(_request(stream=True))
    assert len(response.tool_calls) == 2 and response.latency_ms > 0


def _status(code, status):
    def handler(request):
        body = {"error": {"code": code, "message": "nope", "status": status}}
        return httpx.Response(code, json=body, headers={"retry-after": "4"})

    return handler


@pytest.mark.parametrize(
    ("code", "status", "cls"),
    [
        (429, "RESOURCE_EXHAUSTED", ModelRateLimited),
        (401, "UNAUTHENTICATED", ModelAuthError),
        (403, "PERMISSION_DENIED", ModelAuthError),
        (400, "INVALID_ARGUMENT", ModelBadRequest),
        (404, "NOT_FOUND", ModelBadRequest),
        (408, "DEADLINE_EXCEEDED", ModelTimeout),
        (500, "INTERNAL", ModelUnavailable),
        (503, "UNAVAILABLE", ModelUnavailable),
    ],
)
async def test_api_errors_map(code, status, cls):
    model = GeminiModel("gemini-2.5-flash", client=client_for(_status(code, status)))
    with pytest.raises(cls) as info:
        await model.generate(_request())
    err = info.value
    assert type(err) is cls
    assert (err.provider, err.status_code, err.retry_after) == ("gemini", code, 4.0)
    assert isinstance(err.__cause__, errors.APIError)


@pytest.mark.parametrize(
    ("exc", "cls"),
    [
        (httpx.ReadTimeout("slow"), ModelTimeout),
        (httpx.ConnectError("refused"), ModelUnavailable),
    ],
)
async def test_transport_errors_map(exc, cls):
    model = GeminiModel("gemini-2.5-flash", client=client_for(Wire(exc)))
    with pytest.raises(cls) as info:
        await model.generate(_request())
    assert info.value.status_code is None


async def test_stream_errors_are_mapped():
    model = GeminiModel("gemini-2.5-flash", client=client_for(_status(429, "RESOURCE_EXHAUSTED")))
    with pytest.raises(ModelRateLimited):
        _ = [e async for e in model.stream(_request())]


def test_unmapped_and_missing_sdk(monkeypatch):
    import sys

    assert map_gemini_error(KeyError("x"), model="m") is None
    assert map_gemini_error(ModelError("x", provider="gemini"), model="m") is None
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "httpx", None)
    monkeypatch.setitem(sys.modules, "google.auth", None)
    assert map_gemini_error(RuntimeError(), model="m") is None
    with pytest.raises(MissingDependency, match=r"agent-adapter\[gemini\]"):
        GeminiModel("gemini-2.5-flash")


def test_loop_local_clients_across_two_run_sync_calls(monkeypatch):
    made = []
    real = genai.Client

    def factory(**kwargs):
        client = real(http_options=options(Wire(fixture("gemini/text"))), **kwargs)
        made.append(client)
        return client

    monkeypatch.setattr(genai, "Client", factory)
    model = GeminiModel("gemini-2.5-flash", client_kwargs={"api_key": "test"})
    for _ in range(2):
        assert run_sync(model.generate(_request())).text == '{"answer": 42}'
    assert len(made) == 2 and made[0] is not made[1]


def test_other_genai_errors_become_model_errors():
    unparsable = map_gemini_error(errors.UnknownApiResponseError("not json"), model="m")
    assert type(unparsable) is ModelError
    redirect = map_gemini_error(errors.APIError(302, {"error": {"message": "moved"}}), model="m")
    assert type(redirect) is ModelError and redirect.status_code == 302


async def test_missing_api_key_is_an_auth_error(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelAuthError) as info:
        await GeminiModel("gemini-2.5-flash").generate(_request())
    assert isinstance(info.value.__cause__, ValueError)


def test_google_auth_errors_map():
    from google.auth import exceptions

    cases = [
        (exceptions.DefaultCredentialsError("no adc"), ModelAuthError),
        (exceptions.RefreshError("expired"), ModelAuthError),
        (exceptions.TransportError("down"), ModelUnavailable),
        (exceptions.TimeoutError("slow"), ModelTimeout),
    ]
    for exc, cls in cases:
        assert type(map_gemini_error(exc, model="m")) is cls


async def test_bad_client_kwargs_propagate_unchanged():
    model = GeminiModel("gemini-2.5-flash", client_kwargs={"api_key": "t", "bogus": 1})
    with pytest.raises(TypeError, match="bogus"):
        await model.generate(_request())
