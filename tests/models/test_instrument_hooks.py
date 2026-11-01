"""InstrumentedModel.generate: hooks, model errors, control flow and nested billing."""

import asyncio
import logging

import pytest

from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    GuardrailBlocked,
    ModelRateLimited,
    ModelUnavailable,
    ToolUsageError,
)
from agent_adapter.core.messages import LLMResponse
from agent_adapter.cost import Ledger
from agent_adapter.hooks import Hook, ShortCircuit
from agent_adapter.models import CallableModel, InstrumentedModel
from agent_adapter.runtime import current_context, use_context
from agent_adapter.tracing import EventLog
from tests.fakes.hooks import RecordingHook
from tests.fakes.models import text
from tests.fakes.sinks import RecordingLedger
from tests.models.instrument_support import (
    COST,
    HI,
    PRICES,
    USAGE,
    check_required,
    instrumented,
    scripted,
)

# --- hooks --------------------------------------------------------------------------------


class Short(Hook):
    def __init__(self, value):
        self.value = value

    async def before_llm(self, inv):
        return ShortCircuit(self.value)


async def test_hooks_see_the_request_provider_model_and_context():
    hook = RecordingHook()
    ledger = RecordingLedger()
    ctx = ToolContext(run_id="run_1", call_id="c")
    await InstrumentedModel(scripted(), hooks=[hook], ledger=ledger).bind(ctx).generate(HI)
    names = [name for name, *_ in hook.calls]
    assert names == ["before_llm", "after_llm"]
    inv = hook.calls[0][1][0]
    assert (inv.request, inv.provider, inv.model) == (HI, "scripted", "scripted")
    assert (inv.ctx.run_id, inv.ctx.ledger) == ("run_1", ledger)


async def test_a_before_llm_short_circuit_skips_the_model_and_costs_nothing():
    port = scripted()
    after = RecordingHook()
    model = instrumented(
        port, hooks=[Short(text("canned", usage=USAGE)), after], cache=MemoryCache()
    )
    response = await model.generate(HI)
    again = await model.generate(HI)
    assert (response.text, again.text, port.requests) == ("canned", "canned", [])
    assert [name for name, *_ in after.calls] == ["after_llm", "after_llm"]
    assert [e.cost_usd for e in model.ledger.entries()] == [0.0, 0.0]
    assert model.ledger.entries()[0].metadata["source"] == "hook"
    assert [e.type for e in model.sink.events].count("llm.finished") == 2


async def test_a_short_circuit_value_must_be_a_response():
    model = instrumented(scripted(), hooks=[Short("text")])
    with pytest.raises(ToolUsageError, match="LLMResponse"):
        await model.generate(HI)
    assert model.sink.events[-1].type == "llm.failed"
    assert model.ledger.entries()[0].error_kind == "tool-usage-error"


async def test_after_llm_hooks_replace_the_response_but_the_cache_keeps_the_providers():
    class Upper(Hook):
        async def after_llm(self, inv, response):
            return response.model_copy(update={"text": response.text.upper()})

    port = scripted(text("ok"))
    model = instrumented(port, hooks=[Upper()], cache=MemoryCache())
    assert (await model.generate(HI)).text == "OK"
    assert (await model.generate(HI)).text == "OK"  # not "OK" upper-cased twice from the cache


async def test_after_llm_must_return_a_response():
    class Bad(Hook):
        async def after_llm(self, inv, response):
            return "no"

    with pytest.raises(ToolUsageError, match="after_llm"):
        await instrumented(scripted(), hooks=[Bad()]).generate(HI)


async def test_a_model_error_recovered_by_on_llm_error_is_a_free_success():
    class Recover(Hook):
        async def on_llm_error(self, inv, error):
            return ShortCircuit(text(f"recovered from {error.kind}"))

    port = scripted(ModelUnavailable("down", provider="scripted"))
    model = instrumented(port, hooks=[Recover()], cache=MemoryCache())
    response = await model.generate(HI)
    assert response.text == "recovered from model-unavailable"
    [entry] = model.ledger.entries()
    assert entry.ok and entry.cost_usd == 0.0 and entry.metadata["source"] == "hook"
    assert len(port.requests) == 1
    port.responses.append(text("fresh"))
    assert (await model.generate(HI)).text == "fresh"  # the recovery was not cached


async def test_an_unrecovered_model_error_records_a_failed_entry_and_raises(caplog):
    error = ModelRateLimited("slow down", provider="scripted", model="scripted")
    model = instrumented(scripted(error), hooks=[RecordingHook()])
    with caplog.at_level(logging.WARNING), pytest.raises(ModelRateLimited):
        await model.generate(HI)
    [entry] = model.ledger.entries()
    assert (entry.ok, entry.error_kind, entry.cost_usd) == (False, "model-rate-limited", 0.0)
    failed = model.sink.events[-1]
    assert failed.type == "llm.failed" and failed.data["error_kind"] == "model-rate-limited"
    assert "latency_ms" in failed.data
    check_required(model.sink.events)
    assert "failed: model-rate-limited" in caplog.text
    assert [name for name, *_ in model.hooks[0].calls] == ["before_llm", "on_llm_error"]


async def test_a_failure_after_the_reply_still_records_its_usage():
    class Block(Hook):
        async def after_llm(self, inv, response):
            raise GuardrailBlocked("no", stage="llm_response")

    model = instrumented(scripted(text("ok", usage=USAGE)), hooks=[Block()])
    with pytest.raises(GuardrailBlocked):
        await model.generate(HI)
    [entry] = model.ledger.entries()
    assert not entry.ok and entry.error_kind == "guardrail-blocked"
    assert entry.input_tokens == 1000 and entry.cost_usd == pytest.approx(COST)


async def test_host_exceptions_propagate_unchanged_and_are_recorded():
    async def fn(request):
        raise KeyError("host bug")

    model = instrumented(CallableModel(fn))
    with pytest.raises(KeyError):
        await model.generate(HI)
    assert model.ledger.entries()[0].error_kind == "internal-error"


async def test_control_flow_propagates_without_ledger_or_hooks_and_closes_the_span():
    class GraphBubbleUp(Exception):
        pass

    for error in (asyncio.CancelledError(), GraphBubbleUp()):
        hook = RecordingHook()
        model = instrumented(scripted(error), hooks=[hook])
        with pytest.raises(type(error)):
            await model.generate(HI)
        assert model.ledger.entries() == []
        assert [e.type for e in model.sink.events] == ["llm.started", "llm.failed"]
        assert model.sink.events[-1].data["error_kind"] == "cancelled"
        assert [name for name, *_ in hook.calls] == ["before_llm"]


async def test_the_returned_response_is_an_llm_response():
    response = await InstrumentedModel(scripted()).generate(HI)
    assert isinstance(response, LLMResponse)


async def test_a_failing_price_lookup_leaves_the_event_cost_unknown(caplog):
    class Prices_:
        def cost(self, model, usage, *, at=None):
            raise RuntimeError("table broken")

    class PricingLedger(RecordingLedger):
        prices = Prices_()

    sink = EventLog()
    with caplog.at_level(logging.WARNING):
        await InstrumentedModel(scripted(), ledger=PricingLedger(), sink=sink).generate(HI)
    assert sink.of_type("llm.finished")[0].data["cost_usd"] is None
    assert "pricing failed" in caplog.text


async def test_a_model_call_inside_a_tool_the_port_ran_is_billed_separately():
    ledger = Ledger(prices=PRICES)
    inner = InstrumentedModel(scripted(text("inner", usage=USAGE)), ledger=ledger)

    async def host(request):
        tool_ctx = current_context().child(call_id="tool_call_1")  # e.g. the executor's child
        with use_context(tool_ctx):
            await inner.generate(HI)
        return text("outer", usage=USAGE)

    outer = InstrumentedModel(CallableModel(host, model="scripted"), ledger=ledger)
    await outer.bind(ToolContext(run_id="run_1", call_id="c")).generate(HI)
    entries = ledger.entries("run_1")
    assert [e.entry_id for e in entries] == ["tool_call_1:llm:answer:0", "c:llm:answer:0"]
    assert ledger.total_usd("run_1") == pytest.approx(2 * COST)


async def test_nested_instrumented_calls_on_the_same_ledger_bill_once():
    ledger = Ledger(prices=PRICES)
    inner = InstrumentedModel(scripted(text("inner", usage=USAGE)), ledger=ledger)
    sink = EventLog()
    outer = InstrumentedModel(inner, ledger=ledger, sink=sink)
    await outer.bind(ToolContext(run_id="run_1", call_id="c")).generate(HI)
    [entry] = ledger.entries("run_1")
    inner_id = "c:llm:answer:0:llm:answer:0"  # a child of the outer entry "c:llm:answer:0"
    assert entry.entry_id == inner_id and entry.metadata["parent_call_id"] == "c:llm:answer:0"
    finished = {e.call_id: e for e in sink.of_type("llm.finished")}
    assert finished[inner_id].data["cost_usd"] == pytest.approx(COST)
    assert finished[inner_id].parent_id == "c:llm:answer:0"
    outer = finished["c:llm:answer:0"]
    assert outer.data["cost_usd"] is None and outer.data["nested"] is True
    assert "nested" not in finished[inner_id].data


async def test_a_cache_hit_entry_costs_zero_even_without_a_pricing_ledger():
    ledger = RecordingLedger()
    model = InstrumentedModel(scripted(), ledger=ledger, cache=MemoryCache())
    await model.generate(HI)
    await model.generate(HI)
    assert [e.cost_usd for e in ledger.entries()] == [None, 0.0]
