"""WP08 exit gate: ScriptedModel + Ledger + MemoryCache + EventLog through run_tool_loop."""

import pytest

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.cache import MemoryCache
from agent_adapter.core.errors import ModelUnavailable
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.types import Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.exec import Executor, run_tool_loop
from agent_adapter.models import FallbackModel
from agent_adapter.runtime import use_context
from agent_adapter.tracing import EventLog
from agent_adapter.tracing.events import REQUIRED_DATA
from tests.fakes.models import ScriptedModel, calls, text

PRICES = Prices.load(
    {
        "version": "t1",
        "models": {
            "primary": {"provider": "openai", "input": 1.0, "output": 2.0},
            "backup": {"provider": "openai", "input": 10.0, "output": 20.0},
        },
    }
)
USAGE = Usage(input_tokens=1000, output_tokens=100)
PRIMARY = (1000 * 1.0 + 100 * 2.0) / 1e6
BACKUP = (1000 * 10.0 + 100 * 20.0) / 1e6
USER = (Message(role="user", content="weather in Oslo?"),)
HI = LLMRequest(messages=USER, stage="answer")

registry = Registry()


@registry.tool
def weather(city: str) -> str:
    """Today's weather."""
    return f"sunny in {city}"


@registry.tool(confirm=True)
def book(city: str) -> str:
    """Book a trip (needs approval)."""
    return f"booked {city}"


def script(model="primary"):
    return ScriptedModel(
        [
            calls(("weather", {"city": "Oslo"}), model=model, usage=USAGE),
            text("It is sunny.", model=model, usage=USAGE),
        ],
        model=model,
    )


def pipeline(**options):
    ledger, sink = Ledger(prices=PRICES), EventLog()
    options.setdefault("cache", MemoryCache())
    return Executor(registry, ledger=ledger, sink=sink, **options), ledger, sink


def check_required(sink):
    for event in sink.events:
        assert REQUIRED_DATA[event.type] <= set(event.data), event.type


async def test_a_second_identical_loop_costs_nothing_and_records_cache_hits():
    executor, ledger, sink = pipeline()
    model = script()  # two replies: the second loop must not call the model
    first = await run_tool_loop(model, registry, USER, executor=executor, ctx=executor.context())
    second = await run_tool_loop(model, registry, USER, executor=executor, ctx=executor.context())
    assert first.final.text == second.final.text == "It is sunny."
    assert second.messages == first.messages and model.responses == []
    run_a, run_b = (
        {e.run_id for e in ledger.entries() if e.kind == "llm" and e.cache_hit == hit}
        for hit in (False, True)
    )
    [a], [b] = run_a, run_b
    assert ledger.total_usd(a) == pytest.approx(2 * PRIMARY)
    assert ledger.total_usd(b) == 0.0
    hits = [e for e in ledger.entries(b) if e.kind == "llm"]
    assert len(hits) == 2 and all(e.cache_hit and e.cost_usd == 0.0 for e in hits)
    assert ledger.summary(b).response_cache_hit_rate == 1.0
    assert [e.type for e in sink.events].count("llm.cached") == 2
    check_required(sink)


async def test_a_re_run_reproduces_the_entry_ids():
    executor, ledger, _ = pipeline(cache=None)
    model = ScriptedModel([*script().responses, *script().responses])
    first = await run_tool_loop(
        model, registry, USER, executor=executor, ctx=executor.context(run_id="run_same")
    )
    ids = [e.entry_id for e in ledger.entries("run_same")]
    await run_tool_loop(
        model, registry, USER, executor=executor, ctx=executor.context(run_id="run_same")
    )
    assert [e.entry_id for e in ledger.entries("run_same")] == ids
    assert ids == ["run_same:llm:tool_loop:0", "call_0", "run_same:llm:tool_loop:1"]
    assert first.steps == 2


async def test_fallback_of_instrumented_models_records_one_entry_per_hop():
    executor, ledger, sink = pipeline(cache=None)
    a = ScriptedModel([ModelUnavailable("down", provider="scripted")], model="primary")
    b = ScriptedModel([text("ok", model="backup", usage=USAGE)], model="backup")
    fallback = FallbackModel([executor.model(a), executor.model(b)])
    async with executor.run_scope() as ctx:
        with use_context(ctx):
            await fallback.generate(HI)
    failed, ok = ledger.entries(ctx.run_id)
    assert (failed.ok, failed.error_kind, failed.model) == (False, "model-unavailable", "primary")
    assert (ok.ok, ok.model, ok.cost_usd) == (True, "backup", pytest.approx(BACKUP))
    assert failed.entry_id != ok.entry_id
    types = [e.type for e in sink.events if e.type.startswith("llm.")]
    assert types == ["llm.started", "llm.failed", "llm.fallback", "llm.started", "llm.finished"]


async def test_instrumented_fallback_records_one_entry():
    executor, ledger, sink = pipeline(cache=None)
    a = ScriptedModel([ModelUnavailable("down", provider="scripted")], model="primary")
    b = ScriptedModel([text("ok", model="backup", usage=USAGE)], model="backup")
    model = executor.model(FallbackModel([a, b]))
    async with executor.run_scope() as ctx:
        await model.bind(ctx).generate(HI)
    [entry] = ledger.entries(ctx.run_id)
    assert (entry.ok, entry.provider, entry.model) == (True, "scripted", "backup")
    assert entry.cost_usd == pytest.approx(BACKUP)
    types = [e.type for e in sink.events if e.type.startswith("llm.")]
    assert types == ["llm.started", "llm.fallback", "llm.finished"]
    assert sink.of_type("llm.started")[0].data["provider"] == "fallback"
    check_required(sink)


async def test_a_loop_over_a_fallback_of_instrumented_models_bills_each_hop_once():
    executor, ledger, _ = pipeline(cache=None)
    a = ScriptedModel([ModelUnavailable("down", provider="scripted")], model="primary")
    b = ScriptedModel([text("ok", model="backup", usage=USAGE)], model="backup")
    fallback = FallbackModel([executor.model(a), executor.model(b)])
    result = await run_tool_loop(fallback, registry, USER, executor=executor)
    entries = ledger.entries()
    assert result.final.text == "ok"
    assert [(e.ok, e.model) for e in entries] == [(False, "primary"), (True, "backup")]
    assert ledger.total_usd() == pytest.approx(BACKUP)


async def test_a_pending_approval_stops_the_loop():
    executor, ledger, sink = pipeline(approval=DeferredApproval(MemoryApprovalStore()))
    model = ScriptedModel([calls(("book", {"city": "Oslo"}), model="primary", usage=USAGE)])
    result = await run_tool_loop(model, registry, USER, executor=executor)
    assert (result.finish, result.steps) == ("pending", 1)
    assert result.tool_results[0].status == "pending"
    assert [e.kind for e in ledger.entries()] == ["llm"]  # pending tool calls are not recorded
    assert sink.of_type("tool.pending") and sink.of_type("approval.requested")
    check_required(sink)


async def test_instrumented_fallback_over_a_failing_hop_and_a_raw_port_bills_both():
    executor, ledger, _ = pipeline(cache=None)
    a = ScriptedModel([ModelUnavailable("down", provider="scripted")], model="primary")
    b_raw = ScriptedModel([text("ok", model="backup", usage=USAGE)], model="backup")
    model = executor.model(FallbackModel([executor.model(a), b_raw]))
    async with executor.run_scope() as ctx:
        await model.bind(ctx).generate(HI)
    failed, ok = ledger.entries(ctx.run_id)
    assert (failed.ok, failed.model, failed.cost_usd) == (False, "primary", 0.0)
    assert (ok.ok, ok.model) == (True, "backup")
    assert ok.cost_usd == pytest.approx(0.012) and ledger.total_usd(ctx.run_id) == pytest.approx(
        0.012
    )
    assert failed.metadata["parent_call_id"] == ok.entry_id  # the hop is a child of the call


async def test_streamed_fallback_of_instrumented_hops_bills_each_hop_once():
    executor, ledger, sink = pipeline(cache=None)
    a = ScriptedModel([ModelUnavailable("down", provider="scripted")], model="primary")
    b = ScriptedModel([text("ok", model="backup", usage=USAGE)], model="backup")
    model = executor.model(FallbackModel([executor.model(a), executor.model(b)]))
    async with executor.run_scope() as ctx:
        events = [event async for event in model.bind(ctx).stream(HI)]
    assert events[-1].response.text == "ok"
    entries = ledger.entries(ctx.run_id)
    assert [(e.ok, e.model) for e in entries] == [(False, "primary"), (True, "backup")]
    assert ledger.total_usd(ctx.run_id) == pytest.approx(BACKUP)
    outer = [e for e in sink.of_type("llm.finished") if e.data.get("nested")]
    assert len(outer) == 1 and outer[0].data["cost_usd"] is None
