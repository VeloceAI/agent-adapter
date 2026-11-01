"""InstrumentedModel.stream: the same pipeline around a stream, without the response cache."""

import asyncio

import pytest

from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelOutputInvalid, ModelUnavailable
from agent_adapter.core.messages import LLMRequest, Message, StreamDone, TextDelta, ToolCallDelta
from agent_adapter.core.types import ToolCall, Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.hooks import Hook, ShortCircuit
from agent_adapter.models import CallableModel, FallbackModel, InstrumentedModel
from agent_adapter.runtime import current_context
from agent_adapter.tracing import EventLog
from agent_adapter.tracing.events import REQUIRED_DATA
from tests.fakes.hooks import RecordingHook
from tests.fakes.models import ScriptedModel, calls, text

PRICES = Prices.load(
    {"version": "t1", "models": {"scripted": {"provider": "openai", "input": 1.0, "output": 2.0}}}
)
USAGE = Usage(input_tokens=1000, output_tokens=500)
HI = LLMRequest(messages=(Message(role="user", content="hi"),), stage="s")


def instrumented(port, **kwargs):
    kwargs.setdefault("ledger", Ledger(prices=PRICES))
    kwargs.setdefault("sink", EventLog())
    return InstrumentedModel(port, **kwargs)


async def collect(model, request=HI):
    return [event async for event in model.stream(request)]


def streaming(*events, error=None, seen=None):
    async def stream_fn(request):
        if seen is not None:
            seen.append(current_context())
        for event in events:
            yield event
        if error is not None:
            raise error

    async def fn(request):
        raise AssertionError("not called")

    return CallableModel(fn, provider="scripted", model="scripted", stream_fn=stream_fn)


async def test_deltas_pass_through_and_the_done_response_is_recorded():
    final = text("hello", usage=USAGE)
    seen = []
    model = instrumented(
        streaming(
            TextDelta(text="hel"), TextDelta(text="lo"), StreamDone(response=final), seen=seen
        )
    )
    ctx = ToolContext(run_id="run_1", call_id="c")
    events = await collect(model.bind(ctx))
    assert [type(e).__name__ for e in events] == ["TextDelta", "TextDelta", "StreamDone"]
    assert events[-1].response.text == "hello"
    [entry] = model.ledger.entries()
    assert entry.entry_id == "c:llm:s:0" and entry.cost_usd == pytest.approx(0.002)
    types = [e.type for e in model.sink.events]
    assert types == ["llm.started", "llm.finished"]
    assert model.sink.events[0].data["stream"] is True
    for event in model.sink.events:
        assert REQUIRED_DATA[event.type] <= set(event.data)
    assert seen[0].run_id == "run_1" and seen[0].ledger is model.ledger
    assert current_context() is None  # the per-step context never leaks to the consumer


async def test_the_entry_is_recorded_before_the_final_event_is_yielded():
    model = instrumented(ScriptedModel([text("x")]))
    stream = model.stream(HI)
    async for event in stream:
        if isinstance(event, StreamDone):
            assert len(model.ledger.entries()) == 1
            break
    await stream.aclose()
    assert [e.type for e in model.sink.events] == ["llm.started", "llm.finished"]


async def test_streams_never_use_the_response_cache():
    port = ScriptedModel([text("a"), text("b")])
    model = instrumented(port, cache=MemoryCache())
    await collect(model)
    events = await collect(model)
    assert events[-1].response.text == "b" and len(port.requests) == 2


async def test_after_llm_hooks_replace_the_streamed_response():
    class Upper(Hook):
        async def after_llm(self, inv, response):
            return response.model_copy(update={"text": response.text.upper()})

    events = await collect(instrumented(ScriptedModel([text("ok")]), hooks=[Upper()]))
    assert events[-1].response.text == "OK"


async def test_a_before_llm_short_circuit_is_replayed_as_a_stream():
    canned = calls(("lookup", {"q": "x"}), content="thinking")
    port = ScriptedModel([])
    model = instrumented(port, hooks=[_Short(canned)])
    events = await collect(model)
    assert isinstance(events[0], TextDelta) and isinstance(events[1], ToolCallDelta)
    assert events[-1].response == canned and port.requests == []
    assert model.ledger.entries()[0].cost_usd == 0.0


async def test_an_error_before_the_first_event_can_be_recovered():
    class Recover(Hook):
        async def on_llm_error(self, inv, error):
            return ShortCircuit(text("fallback"))

    model = instrumented(streaming(error=ModelUnavailable("down", provider="p")), hooks=[Recover()])
    events = await collect(model)
    assert events[-1].response.text == "fallback"
    assert model.ledger.entries()[0].metadata["source"] == "hook"


async def test_an_error_after_the_first_event_is_final_and_recorded():
    hook = RecordingHook()
    model = instrumented(
        streaming(TextDelta(text="x"), error=ModelUnavailable("down", provider="p")), hooks=[hook]
    )
    with pytest.raises(ModelUnavailable):
        await collect(model)
    assert "on_llm_error" not in hook.phases()
    [entry] = model.ledger.entries()
    assert not entry.ok and entry.error_kind == "model-unavailable"
    assert [e.type for e in model.sink.events] == ["llm.started", "llm.failed"]


async def test_an_unrecovered_error_before_the_first_event_is_recorded():
    model = instrumented(streaming(error=ModelUnavailable("down", provider="p")))
    with pytest.raises(ModelUnavailable):
        await collect(model)
    assert model.sink.events[-1].type == "llm.failed"


async def test_a_stream_without_done_is_invalid_output():
    model = instrumented(streaming(TextDelta(text="x")))
    with pytest.raises(ModelOutputInvalid):
        await collect(model)
    assert model.ledger.entries()[0].error_kind == "model-output-invalid"


async def test_closing_early_records_nothing_but_closes_the_span():
    closed = []

    async def stream_fn(request):
        try:
            yield TextDelta(text="a")
            yield TextDelta(text="b")
        finally:
            closed.append(True)

    port = CallableModel(lambda r: None, stream_fn=stream_fn)
    model = instrumented(port)
    stream = model.stream(HI)
    assert isinstance(await anext(stream), TextDelta)
    await stream.aclose()
    assert closed == [True]
    assert model.ledger.entries() == []
    assert [e.type for e in model.sink.events] == ["llm.started", "llm.failed"]
    assert model.sink.events[-1].data["error_kind"] == "cancelled"


async def test_cancellation_is_not_recorded():
    model = instrumented(streaming(error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await collect(model)
    assert model.ledger.entries() == []


async def test_fallback_inside_instrumented_streams_with_one_entry_and_a_hop_event():
    first = streaming(error=ModelUnavailable("down", provider="a"))
    second = ScriptedModel([text("ok", usage=USAGE)])
    model = instrumented(FallbackModel([first, second]))
    events = await collect(model)
    assert events[-1].response.text == "ok"
    assert len(model.ledger.entries()) == 1
    assert [e.type for e in model.sink.events] == ["llm.started", "llm.fallback", "llm.finished"]


async def test_tool_call_deltas_pass_through():
    final = calls(("lookup", {"q": "x"}))
    delta = ToolCallDelta(index=0, id="call_0", name="lookup", arguments_delta='{"q":"x"}')
    events = await collect(instrumented(streaming(delta, StreamDone(response=final))))
    assert events[0] == delta
    assert events[-1].response.tool_calls == (
        ToolCall(name="lookup", arguments={"q": "x"}, id="call_0"),
    )


class _Short(Hook):
    def __init__(self, value):
        self.value = value

    async def before_llm(self, inv):
        return ShortCircuit(self.value)


async def test_a_port_stream_that_ends_without_done_is_invalid_output():
    class Bare(CallableModel):
        async def stream(self, request):
            yield TextDelta(text="x")

    model = instrumented(Bare(lambda r: None))
    with pytest.raises(ModelOutputInvalid, match="without StreamDone"):
        await collect(model)
    assert model.sink.events[-1].type == "llm.failed"


def test_replay_of_a_tool_call_only_response():
    from agent_adapter.models._port import replay

    events = replay(calls(("a", {}), ("b", {"x": 1})))
    assert [type(e).__name__ for e in events] == ["ToolCallDelta", "ToolCallDelta", "StreamDone"]
    assert events[1].arguments_delta == '{"x":1}'
