"""InstrumentedModel.generate: context and ids, ledger and events, the response cache."""

import logging

import pytest

from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ToolUsageError,
)
from agent_adapter.core.ports import LLMPort
from agent_adapter.cost import Ledger
from agent_adapter.models import CallableModel, InstrumentedModel
from agent_adapter.runtime import current_context, use_context
from agent_adapter.tracing import EventLog
from tests.fakes.cache import BrokenCache
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

# --- shape --------------------------------------------------------------------------------


def test_is_an_llm_port_and_passes_provider_model_and_key_fields_through():
    port = scripted()
    model = InstrumentedModel(port)
    assert isinstance(model, LLMPort)
    assert (model.provider, model.model, model.wrapped) == ("scripted", "scripted", port)
    assert model.cache_key_fields() == port.cache_key_fields()


def test_bind_returns_a_new_model_sharing_the_configuration():
    ledger, ctx = RecordingLedger(), ToolContext(run_id="run_a")
    model = InstrumentedModel(scripted(), ledger=ledger, namespace="ns", cache_ttl=5)
    bound = model.bind(ctx)
    assert bound is not model and bound.ctx is ctx and model.ctx is None
    assert (bound.ledger, bound.namespace, bound.cache_ttl, bound.wrapped) == (
        ledger,
        "ns",
        5,
        model.wrapped,
    )


def test_hooks_must_be_hook_instances():
    with pytest.raises(ToolUsageError, match="Hook instances"):
        InstrumentedModel(scripted(), hooks=[object()])


# --- context and entry ids (I7) -----------------------------------------------------------


async def test_without_any_context_a_fresh_run_and_call_id_are_used():
    model = instrumented(scripted())
    await model.generate(HI)
    [entry] = model.ledger.entries()
    assert entry.run_id.startswith("run_") and len(entry.run_id) == 36
    call_id = entry.entry_id.split(":")[0]
    assert call_id.startswith("call_") and entry.entry_id == f"{call_id}:llm:answer:0"


async def test_the_current_context_numbers_calls_per_call_id_and_stage():
    model = instrumented(scripted(text("a"), text("b"), text("c")))
    ctx = ToolContext(run_id="run_1", call_id="call_x", thread_id="t")
    with use_context(ctx):
        await model.generate(HI)
        await model.generate(HI)
        await model.generate(HI.model_copy(update={"stage": ""}))
    ids = [e.entry_id for e in model.ledger.entries("run_1")]
    assert ids == [
        "call_x:llm:answer:0",
        "call_x:llm:answer:1",
        "call_x:llm:llm:0",
    ]  # empty stage -> "llm"
    assert {e.thread_id for e in model.ledger.entries()} == {"t"}
    assert model.ledger.entries()[2].stage == "llm"


async def test_a_bound_context_wins_and_a_re_run_reproduces_the_ids():
    model = instrumented(scripted(text("a"), text("b"), text("c")))
    first, rerun = ToolContext(run_id="run_1"), ToolContext(run_id="run_1")
    with use_context(ToolContext(run_id="other", call_id="c")):
        await model.bind(first).generate(HI)
        await model.bind(first).generate(HI)
        await model.bind(rerun).generate(HI)
    ids = [e.entry_id for e in model.ledger.entries("run_1")]
    assert ids == ["run_1:llm:answer:0", "run_1:llm:answer:1"]  # the re-run replaced entry 0
    assert model.ledger.entries("other") == []


async def test_the_counter_is_copied_on_write_so_children_do_not_share_it():
    model = instrumented(scripted(text("a"), text("b")))
    parent = ToolContext(run_id="run_1", call_id="p")
    await model.bind(parent).generate(HI)
    child = parent.child(call_id="p")  # a node re-run with the parent's extras
    snapshot = dict(parent.extras["aa.llm_seq"])
    await model.bind(child).generate(HI)
    assert parent.extras["aa.llm_seq"] == snapshot


async def test_a_metadata_call_id_is_the_entry_id_and_takes_no_number():
    model = instrumented(scripted(text("a"), text("b")))
    ctx = ToolContext(run_id="run_1", call_id="c")
    await model.bind(ctx).generate(HI.model_copy(update={"metadata": {"call_id": "mine"}}))
    await model.bind(ctx).generate(HI)
    assert [e.entry_id for e in model.ledger.entries()] == ["mine", "c:llm:answer:0"]


async def test_an_empty_run_id_gets_a_fresh_one():
    model = instrumented(scripted())
    await model.bind(ToolContext(call_id="c")).generate(HI)
    [entry] = model.ledger.entries()
    assert entry.run_id.startswith("run_") and entry.entry_id == "c:llm:answer:0"


async def test_the_port_runs_under_the_context_with_this_models_ports():
    seen = []

    async def fn(request):
        seen.append(current_context())
        return "ok"

    ledger, sink = Ledger(prices=PRICES), EventLog()
    ctx = ToolContext(run_id="run_1", call_id="c")
    await InstrumentedModel(CallableModel(fn), ledger=ledger, sink=sink).bind(ctx).generate(HI)
    [inner] = seen
    entry_id = "c:llm:answer:0"  # nested calls are children of this call's entry
    assert (inner.run_id, inner.call_id, inner.ledger, inner.sink) == (
        "run_1",
        entry_id,
        ledger,
        sink,
    )
    assert ledger.entries()[0].entry_id == entry_id


async def test_the_contexts_ledger_and_sink_serve_when_the_model_has_none():
    ledger, sink = Ledger(prices=PRICES), EventLog()
    ctx = ToolContext(run_id="run_1", ledger=ledger, sink=sink)
    await InstrumentedModel(scripted()).bind(ctx).generate(HI)
    assert len(ledger.entries()) == 1 and len(sink.events) == 2


async def test_the_models_own_ledger_wins_over_the_contexts():
    own, other = RecordingLedger(), RecordingLedger()
    ctx = ToolContext(run_id="run_1", ledger=other)
    await InstrumentedModel(scripted(), ledger=own).bind(ctx).generate(HI)
    assert len(own.entries()) == 1 and other.entries() == []


# --- ledger and events --------------------------------------------------------------------


async def test_a_call_records_one_priced_entry_and_two_events():
    model = instrumented(scripted(text("ok", usage=USAGE)))
    ctx = ToolContext(run_id="run_1", call_id="tool_1")
    response = await model.bind(ctx).generate(HI)
    assert response.text == "ok"
    [entry] = model.ledger.entries()
    assert entry.kind == "llm" and entry.stage == "answer" and entry.ok
    assert (entry.provider, entry.model, entry.call_id) == ("scripted", "scripted", entry.entry_id)
    assert (entry.input_tokens, entry.output_tokens) == (1000, 500)
    assert entry.cost_usd == pytest.approx(COST) and entry.price_version == "t1"
    assert entry.started_at <= entry.ended_at and entry.latency_ms >= 0
    assert entry.metadata == {"parent_call_id": "tool_1"} and not entry.cache_hit
    started, finished = model.sink.events
    check_required(model.sink.events)
    assert (started.type, finished.type) == ("llm.started", "llm.finished")
    assert started.data == {
        "provider": "scripted",
        "model": "scripted",
        "stage": "answer",
        "stream": False,
    }
    assert {e.call_id for e in model.sink.events} == {entry.entry_id}
    assert {e.parent_id for e in model.sink.events} == {"tool_1"}
    assert finished.data["cost_usd"] == pytest.approx(COST)
    assert finished.data["usage"]["input_tokens"] == 1000
    assert finished.data["finish_reason"] == "stop"
    assert finished.data["response_model"] == "scripted"


async def test_without_a_pricing_ledger_the_event_cost_is_unknown():
    sink = EventLog()
    model = InstrumentedModel(scripted(), ledger=RecordingLedger(), sink=sink)
    await model.generate(HI)
    assert sink.of_type("llm.finished")[0].data["cost_usd"] is None


async def test_the_request_model_is_reported_on_started_events():
    sink = EventLog()
    await InstrumentedModel(scripted(), sink=sink).generate(HI.model_copy(update={"model": "m2"}))
    assert sink.events[0].data["model"] == "m2"


async def test_a_raising_ledger_does_not_fail_the_call(caplog):
    class Boom(RecordingLedger):
        def record(self, entry):
            raise RuntimeError("down")

    with caplog.at_level(logging.WARNING):
        response = await InstrumentedModel(scripted(), ledger=Boom()).generate(HI)
    assert response.text == "ok" and "ledger.record failed" in caplog.text


# --- response cache -----------------------------------------------------------------------


async def test_a_repeated_request_is_served_from_the_response_cache_for_free():
    port = scripted(text("ok", usage=USAGE))
    model = instrumented(port, cache=MemoryCache(), cache_ttl=60)
    first = await model.bind(ToolContext(run_id="run_1")).generate(HI)
    second = await model.bind(ToolContext(run_id="run_2")).generate(HI)
    assert (first.cached, second.cached, second.text) == (False, True, "ok")
    assert len(port.requests) == 1
    [hit] = model.ledger.entries("run_2")
    assert hit.cache_hit and hit.cost_usd == 0.0 and hit.price_version == ""
    assert model.ledger.total_usd("run_2") == 0.0
    types = [e.type for e in model.sink.events]
    assert types == ["llm.started", "llm.finished", "llm.started", "llm.cached"]
    check_required(model.sink.events)


async def test_cache_ttl_reaches_the_backend():
    ttls = []

    class Recording(MemoryCache):
        async def set(self, key, value, ttl=None):
            ttls.append(ttl)
            await super().set(key, value, ttl)

    await InstrumentedModel(scripted(), cache=Recording(), cache_ttl=42).generate(HI)
    assert ttls == [42]


@pytest.mark.parametrize(
    "update", [{"response_cache": False}, {"stream": True}], ids=["opted-out", "stream"]
)
async def test_opted_out_and_streamed_requests_skip_the_response_cache(update):
    port = scripted(text("a"), text("b"))
    model = instrumented(port, cache=MemoryCache())
    request = HI.model_copy(update=update)
    await model.generate(request)
    second = await model.generate(request)
    assert second.text == "b" and len(port.requests) == 2


async def test_a_namespace_separates_response_caches():
    cache, port = MemoryCache(), scripted(text("a"), text("b"))
    await InstrumentedModel(port, cache=cache, namespace="one").generate(HI)
    second = await InstrumentedModel(port, cache=cache, namespace="two").generate(HI)
    assert second.text == "b"


async def test_a_broken_cache_is_a_miss():
    response = await InstrumentedModel(scripted(), cache=BrokenCache()).generate(HI)
    assert response.text == "ok" and not response.cached
