"""Per-call, per-second and free prices on instrumented calls: one duration, one cost."""

import time
from types import SimpleNamespace

import pytest

from agent_adapter.cost import Ledger, Prices
from agent_adapter.hooks import Hook
from agent_adapter.models import CallableModel, InstrumentedModel, _llm_call
from agent_adapter.tracing import EventLog
from tests.fakes.sinks import RecordingLedger
from tests.models.instrument_support import HI, instrumented, scripted

PRICES = Prices.load(
    {
        "version": "own-1",
        "models": {"host": {"provider": "local", "per_call": 0.25, "per_second": 1.0}},
    }
)


class Clock:
    """A fake `perf_counter` that moves only when a test advances it."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(_llm_call, "time", SimpleNamespace(perf_counter=fake, time=time.time))
    return fake


def _model(clock, port_s, **kwargs):
    async def fn(request):
        clock.advance(port_s)  # the provider's own time
        return "ok"

    kwargs.setdefault("sink", EventLog())
    return InstrumentedModel(CallableModel(fn), ledger=Ledger(prices=PRICES), **kwargs)


class SlowHooks(Hook):
    def __init__(self, clock):
        self.clock = clock

    async def before_llm(self, inv):
        self.clock.advance(3.0)

    async def after_llm(self, inv, response):
        self.clock.advance(5.0)
        return response


async def test_event_and_ledger_cost_come_from_one_duration(clock):
    model = _model(clock, 2.0)
    await model.generate(HI)
    [entry] = model.ledger.entries()
    [finished] = model.sink.of_type("llm.finished")
    assert entry.cost_usd == 0.25 + 2.0
    assert entry.price_version == "own-1"
    assert finished.data["cost_usd"] == entry.cost_usd


async def test_per_second_does_not_bill_hooks(clock):
    model = _model(clock, 2.0, hooks=[SlowHooks(clock)])
    await model.generate(HI)
    [entry] = model.ledger.entries()
    assert entry.latency_ms == 10_000.0  # the wall time includes both hooks
    assert entry.cost_usd == 0.25 + 2.0
    assert model.sink.of_type("llm.finished")[0].data["cost_usd"] == entry.cost_usd


async def test_per_second_does_not_bill_a_slow_stream_consumer(clock):
    model = _model(clock, 1.5)
    async for _ in model.stream(HI.model_copy(update={"stream": True})):
        clock.advance(4.0)  # the consumer's pause between chunks
    [entry] = model.ledger.entries()
    assert entry.latency_ms > 4_000.0
    assert entry.cost_usd == 0.25 + 1.5


async def test_entries_without_a_pricing_ledger_reach_the_ledger_unpriced():
    ledger = RecordingLedger()
    model = InstrumentedModel(CallableModel(lambda request: "ok"), ledger=ledger)
    await model.generate(HI)
    assert ledger.entries()[0].cost_usd is None


async def test_event_cost_of_a_free_model_is_zero():
    free = Prices.load({"version": "v", "models": {"*": {"provider": "local", "free": True}}})
    sink = EventLog()
    model = instrumented(scripted(), ledger=Ledger(prices=free), sink=sink)
    await model.generate(HI)
    assert sink.of_type("llm.finished")[0].data["cost_usd"] == 0.0


async def test_model_entries_record_one_attempt(clock):
    model = _model(clock, 0.0)
    await model.generate(HI)
    assert model.ledger.entries()[0].attempts == 1  # SDK-internal retries are not seen
