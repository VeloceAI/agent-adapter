import asyncio
import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ConfigError,
    ModelAuthError,
    ModelBadRequest,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import LLMRequest, Message, StreamDone, TextDelta
from agent_adapter.core.ports import LLMPort
from agent_adapter.models.callable import CallableModel
from agent_adapter.models.fallback import DEFAULT_FALL_ON, FallbackModel
from agent_adapter.runtime.current import use_context
from agent_adapter.tracing.events import EventType
from tests.fakes.models import ScriptedModel, text
from tests.fakes.sinks import RecordingSink

REQUEST = LLMRequest(
    messages=(Message(role="user", content="hi"),), model="primary-big", metadata={"call_id": "c9"}
)


def _err(cls, provider="a"):
    return cls("down", provider=provider, model="m")


def test_constructor_validation_and_identity():
    with pytest.raises(ConfigError):
        FallbackModel([])
    with pytest.raises(ConfigError, match="ModelError"):
        FallbackModel([ScriptedModel([])], fall_on=(ValueError,))  # type: ignore[arg-type]
    model = FallbackModel([ScriptedModel([], model="a"), ScriptedModel([], model="b")])
    assert isinstance(model, LLMPort)
    assert (model.provider, model.model) == ("fallback", "a")
    assert model.fall_on == DEFAULT_FALL_ON


@pytest.mark.parametrize("cls", DEFAULT_FALL_ON)
async def test_falls_over_on_transient_errors(cls, caplog):
    first = ScriptedModel([_err(cls)], provider="a", model="a1")
    second = ScriptedModel([text("ok")], provider="b", model="b1")
    sink = RecordingSink()
    with caplog.at_level(logging.WARNING):
        response = await FallbackModel([first, second], sink=sink).generate(REQUEST)
    assert response.text == "ok"
    assert first.requests[0].model == "primary-big"
    assert second.requests[0].model is None
    assert "a/a1 failed with" in caplog.text and "c9" in caplog.text
    [event] = sink.events
    assert event.type == EventType.LLM_FALLBACK
    assert dict(event.data) == {
        "from_provider": "a",
        "from_model": "primary-big",
        "to_provider": "b",
        "to_model": "b1",
        "error_kind": cls.kind,
    }


@pytest.mark.parametrize("cls", [ModelAuthError, ModelBadRequest])
async def test_other_errors_propagate_without_fallback(cls):
    second = ScriptedModel([text("never")])
    with pytest.raises(cls):
        await FallbackModel([ScriptedModel([_err(cls)]), second]).generate(REQUEST)
    assert second.requests == []


async def test_non_model_errors_and_cancellation_propagate():
    second = ScriptedModel([text("never")])
    with pytest.raises(KeyError):
        await FallbackModel([ScriptedModel([KeyError("x")]), second]).generate(REQUEST)
    with pytest.raises(asyncio.CancelledError):
        await FallbackModel([ScriptedModel([asyncio.CancelledError()]), second]).generate(REQUEST)
    assert second.requests == []


async def test_all_fail_raises_last_error_and_emits_per_hop(caplog):
    sink = RecordingSink()
    chain = [
        ScriptedModel([_err(ModelRateLimited)], model="a"),
        ScriptedModel([_err(ModelTimeout)], model="b"),
        ScriptedModel([_err(ModelUnavailable)], model="c"),
    ]
    with caplog.at_level(logging.WARNING), pytest.raises(ModelUnavailable):
        await FallbackModel(chain, sink=sink).generate(REQUEST)
    assert [e.data["to_model"] for e in sink.events] == ["b", "c"]
    assert sink.events[1].data["from_model"] == "b"
    assert "last model" in caplog.text


async def test_custom_fall_on_and_context_sink():
    sink = RecordingSink()
    chain = [ScriptedModel([_err(ModelBadRequest)]), ScriptedModel([text("ok")])]
    model = FallbackModel(chain, fall_on=(ModelBadRequest,))
    ctx = ToolContext(run_id="run_1", call_id="call_1", sink=sink)
    with use_context(ctx):
        assert (await model.generate(REQUEST)).text == "ok"
    assert sink.events[0].run_id == "run_1"


async def test_stream_falls_over_before_first_event():
    final = text("streamed")

    async def failing(request):
        raise _err(ModelUnavailable)
        yield  # pragma: no cover

    async def working(request):
        yield TextDelta(text="s")
        yield StreamDone(response=final)

    async def unused(request):
        return ""

    chain = [CallableModel(unused, stream_fn=failing), CallableModel(unused, stream_fn=working)]
    events = [e async for e in FallbackModel(chain).stream(REQUEST)]
    assert events == [TextDelta(text="s"), StreamDone(response=final)]


async def test_stream_error_after_first_event_is_final():
    async def broken(request):
        yield TextDelta(text="partial")
        raise _err(ModelUnavailable)

    async def unused(request):
        return ""

    second = ScriptedModel([text("never")])
    model = FallbackModel([CallableModel(unused, stream_fn=broken), second])
    with pytest.raises(ModelUnavailable):
        _ = [e async for e in model.stream(REQUEST)]
    assert second.requests == []


async def test_stream_all_fail_raises_last():
    async def failing(request):
        raise _err(ModelTimeout)
        yield  # pragma: no cover

    async def unused(request):
        return ""

    model = FallbackModel([CallableModel(unused, stream_fn=failing)] * 2)
    with pytest.raises(ModelTimeout):
        _ = [e async for e in model.stream(REQUEST)]


def test_cache_key_fields_list_the_chain():
    model = FallbackModel([ScriptedModel([], provider="a", model="x")])
    [entry] = model.cache_key_fields()["models"]
    assert entry["provider"] == "a" and entry["model"] == "x" and entry["api"] == "callable"


async def test_closing_the_stream_early_closes_the_inner_stream():
    closed = []

    async def endless(request):
        try:
            while True:
                yield TextDelta(text="x")
        finally:
            closed.append(True)

    async def unused(request):
        return ""

    events = FallbackModel([CallableModel(unused, stream_fn=endless)]).stream(REQUEST)
    assert await anext(events) == TextDelta(text="x")
    await events.aclose()
    assert closed == [True]
