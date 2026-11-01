"""The ledger prices user-defined models: per-call, per-second (from `latency_ms`) and free."""

import logging

import pytest

import agent_adapter.cost.prices as prices_module
from agent_adapter.core.types import LedgerEntry
from agent_adapter.cost import Ledger, Prices

OWN = {
    "version": "own-1",
    "models": {
        "gpu-*": {"provider": "runpod", "per_second": 0.002, "per_call": 0.0005},
        "llama*": {"provider": "ollama", "free": True},
    },
}


@pytest.fixture(autouse=True)
def fresh_warnings():
    prices_module._WARNED.clear()
    yield
    prices_module._WARNED.clear()


@pytest.fixture
def ledger():
    return Ledger(prices=Prices.load(OWN))


def llm(model, call_id="c1", **fields):
    return LedgerEntry(run_id="r1", call_id=call_id, kind="llm", model=model, **fields)


def test_per_second_price_uses_the_entry_latency(ledger):
    ledger.record(llm("gpu-a100", latency_ms=2_500.0, output_tokens=400))
    [entry] = ledger.entries()
    assert entry.cost_usd == pytest.approx(0.0005 + 2.5 * 0.002)
    assert entry.price_version == "own-1"


def test_failed_call_with_a_reply_is_still_billed_for_its_time(ledger):
    ledger.record(llm("gpu-a100", latency_ms=1_000.0, ok=False, error_kind="model_timeout"))
    assert ledger.total_usd() == pytest.approx(0.0025)


def test_cache_hit_of_a_per_second_model_costs_nothing(ledger):
    ledger.record(llm("gpu-a100", latency_ms=5_000.0, cache_hit=True))
    assert ledger.entries()[0].cost_usd == 0.0


def test_free_model_costs_zero_without_a_warning(ledger, caplog):
    with caplog.at_level(logging.WARNING):
        ledger.record(llm("llama3.1:8b", latency_ms=9_000.0, input_tokens=10**6))
    assert ledger.entries()[0].cost_usd == 0.0
    assert ledger.summary().unknown_cost_entries == 0
    assert caplog.text == ""


def test_bad_latency_on_a_per_second_model_leaves_the_cost_unknown(ledger, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.ledger"):
        ledger.record(llm("gpu-a100", latency_ms=-1.0))
    assert ledger.entries()[0].cost_usd is None
    assert "pricing failed" in caplog.text


def test_token_only_models_ignore_latency():
    ledger = Ledger()
    ledger.record(llm("gpt-4.1", latency_ms=-5.0, input_tokens=1_000_000))
    assert ledger.entries()[0].cost_usd == 2.0
