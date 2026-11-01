import logging
import threading
from datetime import UTC, date, datetime

import pytest

import agent_adapter.cost.prices as prices_module
from agent_adapter.core.ports import LedgerPort
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost import CallbackSink, Ledger, MemorySink, Prices

TABLE = {
    "version": "t1",
    "models": {
        "m": {"provider": "openai", "input": 1.0, "cached_input": 0.5, "output": 2.0},
        "ttl": {
            "provider": "anthropic",
            "input": 3.0,
            "cache_write_by_ttl": {"5m": 3.75, "1h": 6.0},
            "output": 15.0,
        },
        "promo": {
            "provider": "gemini",
            "periods": [
                {"until": "2026-06-30", "input": 1.0, "output": 1.0, "cache_storage_per_hour": 1.0},
                {"from": "2026-07-01", "input": 2.0, "output": 2.0, "cache_storage_per_hour": 4.0},
            ],
        },
    },
    "units": {"voyage:rerank-2.5": {"provider": "voyage", "per_unit": 0.05}},
}


def ts(day):
    return datetime(day.year, day.month, day.day, 12, tzinfo=UTC).timestamp()


@pytest.fixture(autouse=True)
def fresh_warnings():
    prices_module._WARNED.clear()


@pytest.fixture
def ledger():
    return Ledger(prices=Prices.load(TABLE))


def llm(call_id="c1", run_id="r1", **fields):
    fields.setdefault("model", "m")
    return LedgerEntry(run_id=run_id, call_id=call_id, kind="llm", **fields)


def test_ledger_implements_the_port(ledger):
    assert isinstance(ledger, LedgerPort)


def test_record_prices_llm_entries_and_sets_the_version(ledger):
    ledger.record(llm(input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=500_000))
    [entry] = ledger.entries()
    assert entry.cost_usd == pytest.approx(1.0 + 0.5 + 1.0)
    assert entry.price_version == "t1"


def test_record_prices_cache_writes_by_ttl(ledger):
    usage = Usage(
        cache_write_input_tokens=2_000_000, cache_write_by_ttl={"5m": 1_000_000, "1h": 1_000_000}
    )
    ledger.record(LedgerEntry.from_usage(usage, run_id="r1", call_id="c", kind="llm", model="ttl"))
    assert ledger.entries()[0].cost_usd == pytest.approx(9.75)


def test_record_prices_by_the_entry_start_date(ledger):
    ledger.record(llm("a", model="promo", input_tokens=1_000_000, started_at=ts(date(2026, 6, 30))))
    ledger.record(llm("b", model="promo", input_tokens=1_000_000, started_at=ts(date(2026, 7, 1))))
    assert [e.cost_usd for e in ledger.entries()] == [1.0, 2.0]


def test_record_without_start_time_prices_for_today(ledger, monkeypatch):
    monkeypatch.setattr(prices_module, "_today", lambda: date(2026, 8, 1))
    ledger.record(llm(model="promo", input_tokens=1_000_000))
    assert ledger.entries()[0].cost_usd == 2.0


def test_record_keeps_a_given_cost(ledger):
    ledger.record(llm(input_tokens=1_000_000, cost_usd=9.0, price_version="host"))
    [entry] = ledger.entries()
    assert (entry.cost_usd, entry.price_version) == (9.0, "host")


def test_record_unknown_model_stays_unknown_and_warns_once(ledger, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        ledger.record(llm("a", model="zzz-unknown", input_tokens=10))
        ledger.record(llm("b", model="zzz-unknown", input_tokens=10))
    assert [e.cost_usd for e in ledger.entries()] == [None, None]
    assert [e.price_version for e in ledger.entries()] == ["", ""]
    assert caplog.text.count("zzz-unknown") == 1
    assert ledger.summary().unknown_cost_entries == 2


def test_record_unrowed_variant_is_an_unknown_cost(ledger):
    ledger.record(llm(model="m-luna", input_tokens=10))
    assert ledger.entries()[0].cost_usd is None
    assert ledger.summary().unknown_cost_entries == 1


def test_record_llm_without_model_is_not_priced(ledger, caplog):
    with caplog.at_level(logging.WARNING):
        ledger.record(llm(model="", input_tokens=10))
    assert ledger.entries()[0].cost_usd is None
    assert caplog.text == ""


def test_record_response_cache_hit_costs_nothing(ledger):
    ledger.record(llm(input_tokens=1_000_000, cache_hit=True))
    [entry] = ledger.entries()
    assert (entry.cost_usd, entry.price_version) == (0.0, "")  # no price looked up


def test_record_prices_api_entries_by_provider_and_model(ledger):
    ledger.record(
        LedgerEntry(
            run_id="r1", call_id="a", kind="api", provider="voyage", model="rerank-2.5", units=4.0
        )
    )
    assert ledger.entries()[0].cost_usd == pytest.approx(0.2)


def test_record_prices_gemini_cache_storage(ledger):
    entry = LedgerEntry(
        run_id="r1",
        call_id="s",
        kind="api",
        provider="gemini",
        model="promo:cache_storage",
        units=0.5,
        started_at=ts(date(2026, 7, 2)),
    )
    ledger.record(entry)
    assert ledger.entries()[0].cost_usd == pytest.approx(2.0)


def test_record_leaves_unpriced_api_and_tool_entries_alone(ledger):
    ledger.record(
        LedgerEntry(run_id="r1", call_id="a", kind="api", provider="x", model="y", units=1.0)
    )
    ledger.record(LedgerEntry(run_id="r1", call_id="t", kind="tool", model="search"))
    assert [e.cost_usd for e in ledger.entries()] == [None, None]


def test_record_is_idempotent_and_replaces_in_place(ledger):
    ledger.record(llm("a", input_tokens=1))
    ledger.record(llm("b"))
    ledger.record(llm("a", input_tokens=2_000_000))
    assert [e.call_id for e in ledger.entries()] == ["a", "b"]
    assert ledger.entries()[0].input_tokens == 2_000_000
    assert ledger.total_usd() == pytest.approx(2.0)


def test_entries_are_grouped_by_run_least_recent_first(ledger):
    ledger.record(llm("a", run_id="r1"))
    ledger.record(llm("b", run_id="r2"))
    ledger.record(llm("c", run_id="r1"))
    assert [e.call_id for e in ledger.entries()] == ["b", "a", "c"]


def test_ledger_settings_are_read_only(ledger):
    assert ledger.max_runs == 1000 and ledger.sinks == ()
    assert ledger.prices.version == "t1"
    with pytest.raises(AttributeError):
        ledger.prices = Prices.default()  # type: ignore[misc]


def test_entry_id_separates_entries_of_one_call(ledger):
    ledger.record(llm("c", entry_id="c:llm:plan:0", input_tokens=1_000_000))
    ledger.record(llm("c", entry_id="c:llm:plan:1", input_tokens=1_000_000))
    assert ledger.total_usd() == pytest.approx(2.0)


def test_filters_by_run_and_thread(ledger):
    ledger.record(llm("a", run_id="r1", thread_id="t1", cost_usd=1.0))
    ledger.record(llm("b", run_id="r1", thread_id="t2", cost_usd=2.0))
    ledger.record(llm("c", run_id="r2", thread_id="t1", cost_usd=4.0))
    assert ledger.total_usd() == 7.0
    assert ledger.total_usd("r1") == 3.0
    assert ledger.total_usd(thread_id="t1") == 5.0
    assert ledger.total_usd("r1", thread_id="t1") == 1.0
    assert [e.call_id for e in ledger.entries("r2")] == ["c"]
    assert ledger.entries("nope") == []


def test_oldest_runs_are_evicted_beyond_max_runs():
    ledger = Ledger(prices=Prices.load(TABLE), max_runs=2)
    ledger.record(llm(run_id="r1", cost_usd=1.0))
    ledger.record(llm(run_id="r2", cost_usd=1.0))
    ledger.record(llm("again", run_id="r1", cost_usd=1.0))  # r1 is now the most recent
    ledger.record(llm(run_id="r3", cost_usd=1.0))
    assert {e.run_id for e in ledger.entries()} == {"r1", "r3"}


def test_max_runs_must_be_positive():
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError, match="max_runs"):
        Ledger(max_runs=0)


def test_default_prices_are_the_packaged_table():
    ledger = Ledger()
    ledger.record(llm(model="gpt-4.1", input_tokens=1_000_000))
    assert ledger.entries()[0].cost_usd == pytest.approx(2.0)
    assert ledger.entries()[0].price_version == Prices.default().version


def test_sinks_get_the_priced_entry(ledger):
    memory = MemorySink()
    ledger = Ledger(prices=Prices.load(TABLE), sinks=[memory])
    ledger.record(llm(input_tokens=1_000_000))
    ledger.record(llm(input_tokens=1_000_000))
    assert [e.cost_usd for e in memory.entries] == [1.0, 1.0]


def test_failing_sink_is_logged_and_others_still_run(caplog):
    def boom(entry):
        raise RuntimeError("db down")

    memory = MemorySink()
    ledger = Ledger(prices=Prices.load(TABLE), sinks=[CallbackSink(boom), memory])
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.ledger"):
        ledger.record(llm())
    assert len(memory.entries) == 1
    assert len(ledger.entries()) == 1
    assert "ledger sink CallbackSink failed" in caplog.text


class BrokenPrices:
    version = "broken"

    def cost(self, *args, **kwargs):
        raise RuntimeError("bad table")

    def unit_cost(self, *args, **kwargs):
        raise RuntimeError("bad table")


def test_pricing_failure_is_logged_and_the_entry_still_recorded(caplog):
    ledger = Ledger(prices=BrokenPrices())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.ledger"):
        ledger.record(llm(input_tokens=1))
        ledger.record(LedgerEntry(run_id="r1", call_id="a", kind="api", provider="p", model="m"))
    assert [e.cost_usd for e in ledger.entries()] == [None, None]
    assert caplog.text.count("pricing failed") == 2


def test_record_never_raises(ledger, caplog, monkeypatch):
    def explode(entry):
        raise RuntimeError("index broken")

    monkeypatch.setattr(ledger, "_store", explode)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.ledger"):
        ledger.record(llm())
    assert "ledger failed to record" in caplog.text


def test_summary_for_a_run_and_for_everything(ledger):
    ledger.record(llm("a", run_id="r1", stage="plan", cost_usd=1.0))
    ledger.record(llm("b", run_id="r2", stage="plan", cost_usd=2.0))
    assert ledger.summary("r1").total_usd == 1.0
    assert ledger.summary("r1").run_id == "r1"
    assert ledger.summary().total_usd == 3.0
    assert ledger.summary().by_stage["plan"].calls == 2


def test_clear_one_run_or_everything(ledger):
    ledger.record(llm(run_id="r1"))
    ledger.record(llm(run_id="r2"))
    ledger.clear("r1")
    assert {e.run_id for e in ledger.entries()} == {"r2"}
    ledger.clear("missing")
    ledger.clear()
    assert ledger.entries() == []


def test_concurrent_records_from_threads(ledger):
    def work(n):
        for i in range(100):
            ledger.record(llm(f"{n}-{i}", run_id=f"r{n % 3}", input_tokens=1))

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ledger.entries()) == 600
