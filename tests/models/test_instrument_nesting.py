"""Nested instrumented calls: which call bills, which reports, and interrupted calls."""

import asyncio

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, ModelUnavailable
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.exec import Executor
from agent_adapter.hooks import Hook, ShortCircuit
from agent_adapter.models import CallableModel, FallbackModel, InstrumentedModel
from agent_adapter.tracing import EventLog
from tests.fakes.models import ScriptedModel, text
from tests.fakes.sinks import RecordingLedger

PRICES = Prices.load(
    {"version": "t1", "models": {"m": {"provider": "openai", "input": 1.0, "output": 2.0}}}
)
USAGE = Usage(input_tokens=1000, output_tokens=500)
COST = (1000 * 1.0 + 500 * 2.0) / 1e6
HI = LLMRequest(messages=(Message(role="user", content="hi"),), stage="s")


def ctx():
    """A fresh context per call: entry numbers live in its extras."""
    return ToolContext(run_id="run_1", call_id="c")


def answer(content="ok", **fields):
    return text(content, model="m", usage=USAGE).model_copy(update=fields)


class Router:
    """A host router around an instrumented model that copies the reply it returns."""

    provider = "router"
    model = "m"

    def __init__(self, inner):
        self.inner = inner

    async def generate(self, request):
        reply = await self.inner.generate(request)
        return reply.model_copy(update={"text": reply.text + " (routed)"})

    def stream(self, request):  # pragma: no cover - not used
        raise NotImplementedError

    def cache_key_fields(self):
        return {}


async def test_a_router_that_copies_the_inner_reply_bills_once():
    executor = Executor(ledger=Ledger(prices=PRICES), sink=EventLog())
    model = executor.model(Router(executor.model(ScriptedModel([answer()], model="m"))))
    response = await model.bind(ctx()).generate(HI)
    assert response.text == "ok (routed)"
    [entry] = executor.ledger.entries("run_1")
    assert entry.entry_id == "c:llm:s:0:llm:s:0"  # the inner call
    assert executor.ledger.total_usd("run_1") == pytest.approx(COST)


async def test_a_copy_with_the_same_provider_response_id_matches_before_usage():
    executor = Executor(ledger=Ledger(prices=PRICES))
    reply = answer(id="resp_1")

    class IdRouter(Router):
        async def generate(self, request):
            got = await self.inner.generate(request)
            return got.model_copy(update={"usage": Usage(input_tokens=1)})  # same id, other usage

    model = executor.model(IdRouter(executor.model(ScriptedModel([reply], model="m"))))
    await model.bind(ctx()).generate(HI)
    assert len(executor.ledger.entries("run_1")) == 1


async def test_a_different_reply_after_an_inner_call_is_billed_too():
    executor = Executor(ledger=Ledger(prices=PRICES))
    inner = executor.model(ScriptedModel([answer("draft")], model="m"))

    async def refine(request):
        await inner.generate(request)
        return answer("final", usage=Usage(input_tokens=10, output_tokens=5))

    await executor.model(CallableModel(refine, model="m")).bind(ctx()).generate(HI)
    assert len(executor.ledger.entries("run_1")) == 2


async def test_without_a_ledger_a_nested_call_is_still_reported_once():
    sink = EventLog()
    executor = Executor(sink=sink)
    model = executor.model(executor.model(ScriptedModel([answer()], model="m")))
    await model.bind(ctx()).generate(HI)
    finished = sink.of_type("llm.finished")
    assert [bool(e.data.get("nested")) for e in finished] == [False, True]


async def test_an_inner_call_with_its_own_sink_leaves_the_outer_event_not_nested():
    outer_sink, inner_sink, ledger = EventLog(), EventLog(), Ledger(prices=PRICES)
    inner = InstrumentedModel(ScriptedModel([answer()], model="m"), ledger=ledger, sink=inner_sink)
    outer = InstrumentedModel(inner, ledger=ledger, sink=outer_sink)
    await outer.bind(ctx()).generate(HI)
    [event] = outer_sink.of_type("llm.finished")
    assert "nested" not in event.data and event.data["usage"]["input_tokens"] == 1000
    assert len(ledger.entries("run_1")) == 1  # billed once, by the inner call


async def test_an_inner_call_with_its_own_ledger_bills_both_ledgers():
    own, outer_ledger, sink = Ledger(prices=PRICES), Ledger(prices=PRICES), EventLog()
    inner = InstrumentedModel(ScriptedModel([answer()], model="m"), ledger=own, sink=sink)
    outer = InstrumentedModel(inner, ledger=outer_ledger, sink=sink)
    await outer.bind(ctx()).generate(HI)
    assert len(own.entries()) == len(outer_ledger.entries()) == 1
    assert sink.of_type("llm.finished")[-1].data["nested"] is True


async def test_an_explicit_call_id_gives_each_fallback_hop_its_own_entry():
    executor = Executor(ledger=Ledger(prices=PRICES))
    a = ScriptedModel([ModelUnavailable("down", provider="a")], model="m")
    b = ScriptedModel([answer()], model="m")
    request = HI.model_copy(update={"metadata": {"call_id": "job-7"}})
    await FallbackModel([executor.model(a), executor.model(b)]).generate(request)
    entries = {e.entry_id: e.ok for e in executor.ledger.entries()}
    assert entries == {"job-7:hop0": False, "job-7:hop1": True}


async def test_an_explicit_call_id_on_an_instrumented_fallback_keeps_the_hop_entry():
    executor = Executor(ledger=Ledger(prices=PRICES))
    a = ScriptedModel([ModelUnavailable("down", provider="a")], model="m")
    b_raw = ScriptedModel([answer()], model="m")
    request = HI.model_copy(update={"metadata": {"call_id": "job-8"}})
    model = executor.model(FallbackModel([executor.model(a), b_raw]))
    await model.bind(ctx()).generate(request)
    entries = {e.entry_id: e.ok for e in executor.ledger.entries("run_1")}
    assert entries == {"job-8:hop0": False, "job-8": True}


# --- interrupted calls ------------------------------------------------------------------


class CancelAfter(Hook):
    async def after_llm(self, inv, response):
        raise asyncio.CancelledError


async def test_cancellation_after_a_paid_reply_records_the_reply():
    ledger, sink = Ledger(prices=PRICES), EventLog()
    model = InstrumentedModel(
        ScriptedModel([answer()], model="m"), hooks=[CancelAfter()], ledger=ledger, sink=sink
    )
    with pytest.raises(asyncio.CancelledError):
        await model.bind(ctx()).generate(HI)
    [entry] = ledger.entries("run_1")
    assert (entry.ok, entry.error_kind, entry.input_tokens) == (False, "cancelled", 1000)
    assert entry.cost_usd == pytest.approx(COST)
    assert sink.events[-1].type == "llm.failed"


async def test_cancellation_while_storing_the_reply_records_the_reply():
    class CancelOnSet:
        async def get(self, key):
            return None

        async def set(self, key, value, ttl=None):
            raise asyncio.CancelledError

        async def delete(self, key):
            return None

    ledger = RecordingLedger()
    model = InstrumentedModel(
        ScriptedModel([answer()], model="m"), ledger=ledger, cache=CancelOnSet()
    )
    with pytest.raises(asyncio.CancelledError):
        await model.bind(ctx()).generate(HI)
    [entry] = ledger.entries()
    assert (entry.ok, entry.error_kind, entry.output_tokens) == (False, "cancelled", 500)


class Recover(Hook):
    def __init__(self):
        self.offered = []

    async def on_llm_error(self, inv, error):
        self.offered.append(error)
        return ShortCircuit(answer("swallowed"))


async def test_approval_pending_from_a_port_is_never_offered_to_hooks():
    hook, ledger, sink = Recover(), Ledger(prices=PRICES), EventLog()
    pending = ApprovalPending("apr_1", tool="t")
    model = InstrumentedModel(
        ScriptedModel([pending], model="m"), hooks=[hook], ledger=ledger, sink=sink
    )
    with pytest.raises(ApprovalPending):
        await model.bind(ctx()).generate(HI)
    assert hook.offered == [] and ledger.entries() == []
    failed = sink.events[-1]
    assert failed.type == "llm.failed" and failed.data["error_kind"] == "approval-pending"


async def test_approval_pending_from_a_stream_is_never_offered_to_hooks():
    async def stream_fn(request):
        raise ApprovalPending("apr_2", tool="t")
        yield  # pragma: no cover

    hook, ledger = Recover(), Ledger(prices=PRICES)
    port = CallableModel(lambda r: None, model="m", stream_fn=stream_fn)
    model = InstrumentedModel(port, hooks=[hook], ledger=ledger)
    with pytest.raises(ApprovalPending):
        [event async for event in model.bind(ctx()).stream(HI)]
    assert hook.offered == [] and ledger.entries() == []


async def test_cancellation_after_a_nested_call_billed_adds_no_entry():
    ledger = Ledger(prices=PRICES)
    inner = InstrumentedModel(ScriptedModel([answer()], model="m"), ledger=ledger)
    outer = InstrumentedModel(inner, ledger=ledger, hooks=[CancelAfter()])
    with pytest.raises(asyncio.CancelledError):
        await outer.bind(ctx()).generate(HI)
    [entry] = ledger.entries("run_1")
    assert entry.ok and entry.entry_id == "c:llm:s:0:llm:s:0"


def test_each_inner_response_matches_once():
    from agent_adapter.models._llm_call import Inner, LLMCall, Settings

    call = LLMCall(Settings("p", "m", None, None, {}), HI, ctx(), stream=False)
    first, second = answer(), answer()
    call.inner = [Inner(first, None, None)]
    call.reply = second  # equal model and usage: matched by rule 2
    assert call._inside().response is first
    assert call._inside() is None


# --- round 3 ----------------------------------------------------------------------------


async def test_replies_with_different_provider_ids_are_different_calls():
    executor = Executor(ledger=Ledger(prices=PRICES))
    inner = executor.model(ScriptedModel([answer(id="resp_A")], model="m"))
    raw = ScriptedModel([answer(id="resp_B")], model="m")  # same model and usage

    async def route(request):
        await inner.generate(request)
        return await raw.generate(request)

    await executor.model(CallableModel(route, model="m")).bind(ctx()).generate(HI)
    assert len(executor.ledger.entries("run_1")) == 2
    assert executor.ledger.total_usd("run_1") == pytest.approx(2 * COST)


async def test_an_id_less_copy_still_matches_by_model_and_usage():
    executor = Executor(ledger=Ledger(prices=PRICES))
    inner = executor.model(ScriptedModel([answer(id="resp_A")], model="m"))

    class StripId(Router):
        async def generate(self, request):
            got = await self.inner.generate(request)
            return got.model_copy(update={"id": ""})

    await executor.model(StripId(inner)).bind(ctx()).generate(HI)
    assert len(executor.ledger.entries("run_1")) == 1


async def test_a_bound_inner_model_under_an_instrumented_fallback_bills_once():
    executor = Executor(ledger=Ledger(prices=PRICES))
    shared = ctx()
    inner = executor.model(ScriptedModel([answer()], model="m")).bind(shared)
    outer = executor.model(FallbackModel([inner])).bind(shared)
    await outer.generate(HI)
    [entry] = executor.ledger.entries("run_1")
    assert entry.ok and executor.ledger.total_usd("run_1") == pytest.approx(COST)


async def test_an_outer_failure_after_failed_nested_hops_is_marked_nested():
    sink = EventLog()
    executor = Executor(ledger=Ledger(prices=PRICES), sink=sink)
    a = ScriptedModel([ModelUnavailable("down", provider="a")], model="m")
    b = ScriptedModel([ModelUnavailable("down", provider="b")], model="m")
    model = executor.model(FallbackModel([executor.model(a), executor.model(b)]))
    with pytest.raises(ModelUnavailable):
        await model.bind(ctx()).generate(HI)
    failed = sink.of_type("llm.failed")
    assert [bool(e.data.get("nested")) for e in failed] == [False, False, True]


async def test_an_outer_failure_without_nested_calls_is_not_marked():
    sink = EventLog()
    model = InstrumentedModel(ScriptedModel([ModelUnavailable("x", provider="p")]), sink=sink)
    with pytest.raises(ModelUnavailable):
        await model.generate(HI)
    assert "nested" not in sink.of_type("llm.failed")[0].data
