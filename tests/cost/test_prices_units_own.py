"""Unit prices for user-defined endpoints: pattern keys and free entries; per-call per attempt."""

import logging
from datetime import date

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost import Ledger, Prices
from agent_adapter.cost import prices as prices_module

UNITS = {
    "version": "u1",
    "models": {
        "acme-*": {"provider": "local", "input": 1.0, "output": 1.0},
        "llama*": {"provider": "ollama", "free": True},
        "per-req": {"provider": "acme", "per_call": 0.01},
        "pro": {"provider": "gemini", "input": 1.0, "output": 1.0, "cache_storage_per_hour": 4.5},
    },
    "units": {
        "voyage:rerank-2.5": {"provider": "voyage", "per_unit": 0.05},
        "voyage:*": {"provider": "voyage", "per_unit": 0.10},
        "voyage:voyage-3*": {"provider": "voyage", "per_unit": 0.06},
        "local:*": {"provider": "local", "free": True},
        "local:acme-13b": {"provider": "local", "per_unit": 2.0},
    },
}


@pytest.fixture
def prices():
    return Prices.load(UNITS)


@pytest.fixture(autouse=True)
def fresh_warnings():
    prices_module._WARNED.clear()
    yield
    prices_module._WARNED.clear()


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("voyage:rerank-2.5", 0.05),  # exact beats the patterns
        ("voyage:voyage-3-large", 0.06),  # longest pattern first
        ("voyage:rerank-3", 0.10),
        ("local:embedder", 0.0),  # free
    ],
)
def test_unit_patterns_exact_then_longest(prices, item, expected):
    assert prices.unit_cost(item, 1.0) == pytest.approx(expected)


def test_free_unit_never_warns_and_unmatched_still_does(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.unit_cost("local:embedder", 9.0) == 0.0
        assert prices.unit_cost("cohere:embed-4", 1.0) is None
    assert "local:embedder" not in caplog.text
    assert caplog.text.count("no unit price for 'cohere:embed-4'") == 1


def test_unit_catch_all_matches_any_provider():
    prices = Prices.default().merged({"units": {"*": {"provider": "any", "free": True}}})
    assert prices.unit_cost("cohere:embed-4", 5.0) == 0.0


def test_model_units_use_the_requested_name_for_a_pattern_model(prices):
    usage = Usage(input_tokens=1_000_000, units=1.0)
    assert prices.cost("acme-13b", usage) == pytest.approx(1.0 + 2.0)  # "local:acme-13b"
    assert prices.cost("acme-7b", usage) == pytest.approx(1.0)  # "local:*" is free


def test_storage_of_a_free_model_is_free_and_silent(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.unit_cost("ollama:llama3:cache_storage", 2.0) == 0.0
    assert caplog.text == ""
    assert prices.unit_cost("gemini:pro:cache_storage", 2.0) == 9.0


def test_storage_without_a_model_rate_falls_back_to_the_units_table():
    prices = Prices.load(
        {
            "version": "v",
            "models": {"m": {"provider": "p", "input": 1.0, "output": 1.0}},
            "units": {"p:*": {"provider": "p", "per_unit": 0.5}},
        }
    )
    assert prices.unit_cost("p:m:cache_storage", 2.0) == 1.0


@pytest.mark.parametrize(
    ("key", "entry", "fragment"),
    [
        ("voyage:*", {"provider": "voyage", "free": True, "per_unit": 0.1}, "free"),
        ("voyage:x", {"provider": "voyage"}, "per_unit"),
        ("voyage:a*b", {"provider": "voyage", "per_unit": 0.1}, "'\\*' may only end"),
        ("voyage*", {"provider": "voyage", "per_unit": 0.1}, "provider:model"),
        ("voyage:", {"provider": "voyage", "per_unit": 0.1}, "provider:model"),
    ],
)
def test_invalid_unit_entries_are_rejected(key, entry, fragment):
    with pytest.raises(ConfigError, match=fragment):
        Prices.load({"version": "v", "units": {key: entry}})


# --- per-call price per attempt -------------------------------------------------------------


def test_per_call_is_charged_per_attempt(prices):
    assert prices.cost("per-req", Usage(), attempts=3) == pytest.approx(0.03)


@pytest.mark.parametrize("attempts", [0, -1])
def test_bad_attempts_for_a_per_call_price_is_a_config_error(prices, attempts):
    with pytest.raises(ConfigError, match="attempts"):
        prices.cost("per-req", Usage(), attempts=attempts)


def test_token_only_models_ignore_attempts():
    assert Prices.default().cost("gpt-4.1", Usage(input_tokens=1_000_000), attempts=0) == 2.0


def test_ledger_charges_per_call_for_each_attempt_it_sees(prices):
    ledger = Ledger(prices=prices)
    for call_id, attempts in (("a", 1), ("b", 2), ("c", 0)):
        ledger.record(
            LedgerEntry(run_id="r", call_id=call_id, kind="llm", model="per-req", attempts=attempts)
        )
    assert [e.cost_usd for e in ledger.entries()] == pytest.approx([0.01, 0.02, 0.01])


def test_ledger_prices_free_api_units_without_a_warning(prices, caplog):
    ledger = Ledger(prices=prices)
    entry = LedgerEntry(
        run_id="r", call_id="e", kind="api", provider="local", model="embedder", units=4.0
    )
    with caplog.at_level(logging.WARNING):
        ledger.record(entry)
    assert ledger.entries()[0].cost_usd == 0.0
    assert caplog.text == ""


def test_storage_in_a_free_period_is_free():
    entry = {
        "provider": "gemini",
        "cache_storage_per_hour": 1.0,
        "periods": [
            {"until": "2026-12-31", "free": True},
            {"from": "2027-01-01", "input": 1.0, "output": 1.0},
        ],
    }
    prices = Prices.load({"version": "v", "models": {"promo": entry}})
    assert prices.unit_cost("gemini:promo:cache_storage", 2.0, at=date(2026, 6, 1)) == 0.0
    assert prices.unit_cost("gemini:promo:cache_storage", 2.0, at=date(2027, 6, 1)) == 2.0
