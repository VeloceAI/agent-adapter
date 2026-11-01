"""Every OpenAI row of the error map (contract §11.2), through the SDK and a MockTransport."""

import json
import logging

import pytest

from agent_adapter.core.errors import (
    ModelAuthError,
    ModelBadRequest,
    ModelContentFiltered,
    ModelError,
    ModelOutputInvalid,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import LLMRequest, Message
from tests.fakes.models import FakePromptCachePlugin

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")

from agent_adapter.models.openai import OpenAIModel, map_openai_error  # noqa: E402

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))


def _model(handler, api="chat"):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = openai.AsyncOpenAI(api_key="t", http_client=http, max_retries=0)
    return OpenAIModel("gpt-4.1", api=api, client=client)


def _status(code, headers=None):
    def handler(request):
        body = {"error": {"message": f"status {code}", "type": "x"}}
        return httpx2.Response(code, json=body, headers=headers or {})

    return handler


@pytest.mark.parametrize(
    ("code", "cls"),
    [
        (400, ModelBadRequest),
        (401, ModelAuthError),
        (403, ModelAuthError),
        (404, ModelBadRequest),
        (409, ModelBadRequest),
        (418, ModelBadRequest),
        (422, ModelBadRequest),
        (429, ModelRateLimited),
        (500, ModelUnavailable),
        (503, ModelUnavailable),
    ],
)
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_status_errors_map(code, cls, api, caplog):
    headers = {"retry-after": "7", "x-request-id": "req_9"}
    with caplog.at_level(logging.WARNING), pytest.raises(cls) as info:
        await _model(_status(code, headers), api).generate(REQUEST)
    err = info.value
    assert type(err) is cls
    assert (err.provider, err.model, err.status_code) == ("openai", "gpt-4.1", code)
    assert (err.retry_after, err.request_id) == (7.0, "req_9")
    assert isinstance(err.__cause__, openai.APIStatusError)
    assert cls.kind in caplog.text


async def test_connection_error_is_unavailable_and_timeout_is_timeout():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ModelUnavailable) as info:
        await _model(refuse).generate(REQUEST)
    assert isinstance(info.value.__cause__, openai.APIConnectionError)
    with pytest.raises(ModelTimeout) as info:
        await _model(slow).generate(REQUEST)
    assert isinstance(info.value.__cause__, openai.APITimeoutError)


async def test_stream_errors_are_mapped_too():
    with pytest.raises(ModelRateLimited):
        _ = [e async for e in _model(_status(429)).stream(REQUEST)]


async def test_stream_error_event_raises_without_remapping():
    event = {"type": "error", "code": "server_error", "message": "x", "sequence_number": 0}
    body = f"event: error\ndata: {json.dumps(event)}\n\n"

    def handler(request):
        return httpx2.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    with pytest.raises(ModelUnavailable) as info:
        _ = [e async for e in _model(handler, api="responses").stream(REQUEST)]
    assert info.value.__cause__ is None


def _completion():
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {"id": "c", "object": "chat.completion", "created": 1, "model": "m", "choices": []}
    )


def test_parse_helper_errors_map():
    filtered = map_openai_error(openai.ContentFilterFinishReasonError(), model="m")
    length = map_openai_error(openai.LengthFinishReasonError(completion=_completion()), model="m")
    assert type(filtered) is ModelContentFiltered
    assert type(length) is ModelOutputInvalid


def test_response_validation_error_is_bad_request():
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(200, request=request)
    exc = openai.APIResponseValidationError(response=response, body={})
    assert type(map_openai_error(exc, model="m")) is ModelBadRequest


def test_unmapped_exceptions_return_none():
    assert map_openai_error(KeyError("x"), model="m") is None
    assert map_openai_error(ModelError("x", provider="openai"), model="m") is None


def test_missing_sdk_maps_nothing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openai", None)
    assert map_openai_error(RuntimeError(), model="m") is None


async def test_unmapped_errors_propagate_unchanged():
    class Boom(Exception):
        pass

    plugin = FakePromptCachePlugin({}, edit=lambda kwargs: (_ for _ in ()).throw(Boom()))
    model = _model(_status(200))
    model.cache_plugin = plugin
    with pytest.raises(Boom):
        await model.generate(REQUEST)

    def explode(request):
        raise Boom()

    with pytest.raises(Boom):
        await _model(explode).generate(REQUEST)


def _sse_error(body):
    def handler(request):
        payload = f"data: {json.dumps(body)}\n\n"
        return httpx2.Response(
            200, content=payload.encode(), headers={"content-type": "text/event-stream"}
        )

    return handler


@pytest.mark.parametrize(
    ("error", "cls"),
    [
        ({"type": "server_error", "message": "boom"}, ModelUnavailable),
        ({"type": "tokens", "code": "rate_limit_exceeded", "message": "slow"}, ModelRateLimited),
        ({"message": "odd"}, ModelError),
    ],
)
async def test_mid_stream_api_error_is_typed(error, cls):
    model = _model(_sse_error({"error": error}))
    with pytest.raises(cls) as info:
        _ = [e async for e in model.stream(REQUEST)]
    assert type(info.value) is cls
    assert isinstance(info.value.__cause__, openai.APIError)


async def test_fallback_hops_on_a_mid_stream_error_before_the_first_delta():
    from agent_adapter.models.fallback import FallbackModel
    from tests.fakes.models import ScriptedModel, text

    broken = _model(_sse_error({"error": {"type": "server_error", "message": "boom"}}))
    events = [e async for e in FallbackModel([broken, ScriptedModel([text("ok")])]).stream(REQUEST)]
    assert events[-1].response.text == "ok"


def test_every_other_sdk_error_becomes_a_model_error():
    err = map_openai_error(openai.OpenAIError("no api key"), model="m")
    assert type(err) is ModelError and err.provider == "openai"


async def test_missing_api_key_is_an_auth_error(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        monkeypatch.delenv(name, raising=False)
    model = OpenAIModel("gpt-5.6")
    with pytest.raises(ModelAuthError) as info:
        await model.generate(REQUEST)
    assert isinstance(info.value.__cause__, openai.OpenAIError)
    with pytest.raises(ModelAuthError):
        _ = [e async for e in model.stream(REQUEST)]


async def test_sdk_errors_raised_by_a_plugin_are_mapped():
    async def failing(request, kwargs, *, model, client):
        raise openai.OpenAIError("plug-in used the SDK badly")

    model = _model(_status(200))
    model.cache_plugin = FakePromptCachePlugin({})
    model.cache_plugin.apply = failing  # type: ignore[method-assign]
    with pytest.raises(ModelError) as info:
        await model.generate(REQUEST)
    assert type(info.value) is ModelError


async def test_bad_client_kwargs_propagate_unchanged():
    model = OpenAIModel("gpt-5.6", client_kwargs={"api_key": "t", "bogus": 1})
    with pytest.raises(TypeError, match="bogus"):
        await model.generate(REQUEST)
