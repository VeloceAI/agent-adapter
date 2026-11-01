import pytest

from agent_adapter.core.errors import ModelOutputInvalid, ToolUsageError
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    OutputSchema,
    StreamDone,
    TextDelta,
    ToolCallDelta,
)
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import ToolCall, Usage
from agent_adapter.models.callable import CallableModel
from tests.fakes.models import text

REQUEST = LLMRequest(messages=(Message(role="user", content="hi"),))
SCHEMA = OutputSchema(name="A", json_schema={"type": "object"})


async def test_str_result_becomes_response_with_port_identity():
    async def host(request):
        return "hello"

    model = CallableModel(host, provider="host", model="router")
    assert isinstance(model, LLMPort)
    response = await model.generate(REQUEST)
    assert (response.provider, response.model, response.text) == ("host", "router", "hello")
    assert response.message == Message(role="assistant", content="hello")
    assert response.latency_ms > 0
    other = await model.generate(REQUEST.model_copy(update={"model": "big"}))
    assert other.model == "big"


async def test_sync_function_is_accepted():
    model = CallableModel(lambda request: "sync")  # type: ignore[arg-type,return-value]
    assert (await model.generate(REQUEST)).text == "sync"


async def test_mapping_result_fills_message_and_finish():
    async def host(request):
        return {
            "text": "",
            "tool_calls": [{"name": "t", "arguments": {"a": 1}, "id": "c1"}, ToolCall(name="u")],
            "usage": {"input_tokens": 3},
        }

    response = await CallableModel(host).generate(REQUEST)
    assert response.finish_reason == "tool_calls"
    assert [c.name for c in response.tool_calls] == ["t", "u"]
    assert response.message.tool_calls == response.tool_calls
    assert response.usage == Usage(input_tokens=3)
    assert (response.provider, response.model) == ("callable", "host")


async def test_invalid_mapping_raises_output_invalid():
    async def host(request):
        return {"finish_reason": "exploded"}

    with pytest.raises(ModelOutputInvalid, match="invalid response mapping"):
        await CallableModel(host).generate(REQUEST)

    async def bad_calls(request):
        return {"tool_calls": 5}

    with pytest.raises(ModelOutputInvalid):
        await CallableModel(bad_calls).generate(REQUEST)


async def test_response_result_kept_and_latency_preserved():
    given = text("x").model_copy(update={"latency_ms": 12.0})

    async def host(request):
        return given

    assert await CallableModel(host).generate(REQUEST) == given


async def test_unsupported_result_type():
    async def host(request):
        return 42

    with pytest.raises(ToolUsageError, match="int"):
        await CallableModel(host).generate(REQUEST)


async def test_host_errors_propagate_unchanged():
    async def host(request):
        raise KeyError("boom")

    with pytest.raises(KeyError):
        await CallableModel(host).generate(REQUEST)


async def test_structured_output_parsed_or_invalid():
    request = REQUEST.model_copy(update={"output_schema": SCHEMA})

    async def good(request):
        return '{"a": 1}'

    assert (await CallableModel(good).generate(request)).parsed == {"a": 1}

    async def given(request):
        return text("ignored", parsed={"b": 2})

    assert (await CallableModel(given).generate(request)).parsed == {"b": 2}

    async def bad(request):
        return "nope"

    with pytest.raises(ModelOutputInvalid):
        await CallableModel(bad).generate(request)


async def test_stream_without_stream_fn_replays_reply():
    async def host(request):
        assert request.stream is False
        return {"text": "hi", "tool_calls": [{"name": "t", "arguments": {"a": 1}, "id": "c"}]}

    events = [
        e async for e in CallableModel(host).stream(REQUEST.model_copy(update={"stream": True}))
    ]
    assert events[0] == TextDelta(text="hi")
    assert events[1] == ToolCallDelta(index=0, id="c", name="t", arguments_delta='{"a":1}')
    assert isinstance(events[2], StreamDone) and events[2].response.text == "hi"


async def test_stream_fn_used_for_stream_and_streaming_generate():
    final = text("done")

    async def host(request):
        raise AssertionError("fn must not run")

    async def stream_fn(request):
        yield TextDelta(text="do")
        yield StreamDone(response=final)

    model = CallableModel(host, stream_fn=stream_fn)
    events = [e async for e in model.stream(REQUEST)]
    assert events == [TextDelta(text="do"), StreamDone(response=final)]
    assert await model.generate(REQUEST.model_copy(update={"stream": True})) == final


async def test_stream_fn_without_done_raises():
    async def stream_fn(request):
        yield TextDelta(text="x")

    async def host(request):
        return "unused"

    model = CallableModel(host, stream_fn=stream_fn)
    with pytest.raises(ModelOutputInvalid, match="StreamDone"):
        await model.generate(REQUEST.model_copy(update={"stream": True}))
    with pytest.raises(ModelOutputInvalid, match="StreamDone"):
        _ = [e async for e in model.stream(REQUEST)]


def test_cache_key_fields_name_the_function():
    async def host(request):
        return ""

    class Router:
        async def __call__(self, request):
            return ""

    assert CallableModel(host).cache_key_fields() == {
        "api": "callable",
        "fn": f"{__name__}.test_cache_key_fields_name_the_function.<locals>.host",
    }
    assert CallableModel(Router()).cache_key_fields()["fn"].endswith("<locals>.Router")


async def test_llm_response_parse_round_trip():
    async def host(request):
        return '{"a": 1}'

    response: LLMResponse = await CallableModel(host).generate(REQUEST)
    assert response.finish_reason == "stop"


async def test_closing_the_stream_early_closes_stream_fn():
    closed = []

    async def endless(request):
        try:
            while True:
                yield TextDelta(text="x")
        finally:
            closed.append(True)

    async def host(request):
        return ""

    events = CallableModel(host, stream_fn=endless).stream(REQUEST)
    assert await anext(events) == TextDelta(text="x")
    await events.aclose()
    assert closed == [True]
