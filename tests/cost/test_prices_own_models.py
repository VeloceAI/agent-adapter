"""Prices for user-defined models: pattern keys, free entries, per-call and per-second prices."""

import logging
import math
from datetime import date

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ModelCaps, Usage
from agent_adapter.cost import Prices
from agent_adapter.cost import prices as prices_module

OWN = {
    "version": "own-1",
    "aliases": {"house": "acme-7b"},
    "models": {
        "acme-7b": {"provider": "local", "input": 1.0, "output": 2.0},
        "acme-*": {"provider": "local", "input": 0.5, "output": 1.0, "caps": {"reasoning": True}},
        "acme-big-*": {"provider": "local", "input": 3.0, "output": 6.0},
        "llama*": {"provider": "ollama", "free": True},
        "gpu-box": {"provider": "runpod", "per_second": 0.002},
        "per-req": {"provider": "acme", "per_call": 0.01},
        "mixed": {
            "provider": "acme",
            "input": 1.0,
            "output": 1.0,
            "per_call": 0.5,
            "per_second": 0.1,
        },
        "promo-box": {
            "provider": "runpod",
            "periods": [
                {"until": "2026-12-31", "free": True},
                {"from": "2027-01-01", "per_second": 0.004, "per_call": 0.001},
            ],
        },
    },
}


@pytest.fixture
def prices():
    return Prices.load(OWN)


@pytest.fixture(autouse=True)
def fresh_warnings():
    prices_module._WARNED.clear()
    yield
    prices_module._WARNED.clear()


# --- pattern keys ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("acme-7b", "acme-7b"),  # exact beats the pattern
        ("house", "acme-7b"),  # alias beats the pattern
        ("acme-13b", "acme-*"),
        ("acme-", "acme-*"),  # the empty rest matches too
        ("acme-big-2", "acme-big-*"),  # longest pattern first
        ("llama3.1:8b", "llama*"),
        ("acme-*", "acme-*"),  # the key itself is an exact name
        ("other", None),
        ("acm", None),
    ],
)
def test_resolve_patterns_after_exact_and_alias_longest_first(prices, model, expected):
    assert prices.resolve(model) == expected


def test_pattern_prices_and_caps_apply_to_every_match(prices):
    assert prices.cost("acme-13b", Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == 1.5
    assert prices.price("acme-13b").model == "acme-*"
    assert prices.caps("acme-13b") == ModelCaps(reasoning=True)
    assert prices.caps("acme-7b") == ModelCaps()


def test_catch_all_matches_only_what_nothing_else_matches():
    house = Prices.default().merged({"models": {"*": {"provider": "local", "free": True}}})
    assert house.resolve("my-own-model") == "*"
    assert house.resolve("gpt-4.1") == "gpt-4.1"
    assert house.resolve("gpt-4.1-2025-04-14") == "gpt-4.1"  # alias
    assert house.resolve("gpt-4.1-20250414") == "gpt-4.1"  # snapshot of a priced row
    assert house.cost("my-own-model", Usage(input_tokens=10**9)) == 0.0


def test_a_caps_only_family_gives_caps_but_never_blocks_a_price(caplog):
    house = Prices.default().merged({"models": {"*": {"provider": "local", "free": True}}})
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert house.cost("claude-opus-4-5", Usage(input_tokens=10**6)) == 0.0
        assert house.price("claude-opus-4-5").model == "*"
    assert caplog.text == ""
    assert house.resolve("claude-opus-4-5") == "claude-"  # the caps key
    assert house.caps("claude-opus-4-5").cache_control is True


def test_a_shorter_pattern_prices_ids_under_a_caps_only_family():
    house = Prices.default().merged(
        {"models": {"claude*": {"provider": "anthropic", "input": 1.0, "output": 2.0}}}
    )
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert house.cost("claude-house-x", usage) == 3.0
    assert house.caps("claude-house-x").cache_control is True  # caps still from "claude-"


def test_an_exact_caps_only_row_falls_through_to_a_priced_pattern():
    prices = Prices.load(
        {
            "version": "v",
            "models": {
                "m1": {"provider": "p", "caps": {"reasoning": True}},
                "m*": {"provider": "p", "per_call": 0.5},
            },
        }
    )
    assert prices.cost("m1", Usage()) == 0.5
    assert prices.caps("m1").reasoning is True


def test_caps_only_without_any_price_still_warns_about_the_caps_key(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert Prices.default().cost("claude-opus-4-5", Usage(input_tokens=1)) is None
    assert "model 'claude-' has capabilities only" in caplog.text


def test_snapshot_row_beats_a_shorter_pattern_but_not_a_longer_one():
    house = Prices.default().merged(
        {
            "models": {
                "gpt-*": {"provider": "proxy", "input": 9.0, "output": 9.0},
                "gpt-4.1-2025*": {"provider": "proxy", "input": 7.0, "output": 7.0},
            }
        }
    )
    assert house.resolve("gpt-4.1-0613") == "gpt-4.1"
    assert house.resolve("gpt-4.1-nano") == "gpt-*"  # not a snapshot: falls to the pattern
    assert house.resolve("gpt-4.1-20250414") == "gpt-4.1-2025*"


def test_same_length_pattern_beats_a_caps_only_family():
    house = Prices.default().merged(
        {"models": {"claude-*": {"provider": "anthropic", "input": 3.0, "output": 15.0}}}
    )
    assert house.resolve("claude-opus-4-5") == "claude-*"
    assert house.resolve("claude-") == "claude-"  # exact name


def test_default_table_has_no_patterns_and_unknown_models_still_warn(caplog):
    prices = Prices.default()
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.cost("my-own-model", Usage(input_tokens=1)) is None
    assert "no price for model 'my-own-model'" in caplog.text


def test_unmatched_model_with_patterns_still_warns(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.price("other") is None
    assert "no price for model 'other'" in caplog.text


@pytest.mark.parametrize("key", ["a*b", "**", "*x", "acme-**", "a*-*"])
def test_star_anywhere_but_the_end_is_rejected(key):
    with pytest.raises(ConfigError, match="'\\*' may only end"):
        Prices.load({"version": "v", "models": {key: {"provider": "p", "free": True}}})
    with pytest.raises(ConfigError, match="'\\*' may only end"):
        Prices.default().merged({"models": {key: {"provider": "p", "free": True}}})


def test_alias_to_a_pattern_or_with_a_star_is_rejected():
    models = {"acme-*": {"provider": "p", "free": True}}
    with pytest.raises(ConfigError, match="pattern"):
        Prices.load({"version": "v", "aliases": {"a": "acme-*"}, "models": models})
    with pytest.raises(ConfigError, match="pattern"):
        Prices.load({"version": "v", "aliases": {"a*": "x"}, "models": {"x": models["acme-*"]}})


def test_caps_only_pattern_gives_caps_but_no_price(caplog):
    prices = Prices.load(
        {"version": "v", "models": {"*": {"provider": "p", "caps": {"reasoning": True}}}}
    )
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.price("anything") is None
    assert "capabilities only" in caplog.text
    assert prices.caps("anything").reasoning is True


# --- free entries ---------------------------------------------------------------------------


def test_free_entry_costs_zero_and_never_warns(prices, caplog):
    usage = Usage(input_tokens=5_000, cached_input_tokens=10, output_tokens=900, units=3.0)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.cost("llama3.1:8b", usage, seconds=12.0) == 0.0
        resolved = prices.price("llama3.1:8b")
    assert caplog.text == ""
    assert resolved is not None and resolved.free is True
    assert (resolved.input, resolved.output, resolved.per_call, resolved.per_second) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "extra",
    [
        {"input": 1.0, "output": 1.0},
        {"per_call": 0.1},
        {"per_second": 0.1},
        {"cache_storage_per_hour": 1.0},
        {"tiers": [{"max_prompt_tokens": None, "input": 1.0, "output": 1.0}]},
    ],
)
def test_free_entry_with_prices_is_rejected(extra):
    with pytest.raises(ConfigError, match="free"):
        Prices.load({"version": "v", "models": {"m": {"provider": "p", "free": True, **extra}}})


def test_free_family_key_is_rejected():
    with pytest.raises(ConfigError, match="caps-only"):
        Prices.load({"version": "v", "models": {"acme-": {"provider": "p", "free": True}}})


def test_free_with_periods_at_the_top_level_is_rejected():
    entry = {"provider": "p", "free": True, "periods": [{"per_call": 0.1}]}
    with pytest.raises(ConfigError, match="periods"):
        Prices.load({"version": "v", "models": {"m": entry}})


def test_free_period_then_a_priced_one(prices):
    usage = Usage(input_tokens=10)
    assert prices.cost("promo-box", usage, at=date(2026, 12, 31), seconds=100.0) == 0.0
    assert prices.cost("promo-box", usage, at=date(2027, 1, 1), seconds=100.0) == pytest.approx(
        0.401
    )


def test_free_false_without_prices_is_caps_only(caplog):
    prices = Prices.load({"version": "v", "models": {"m": {"provider": "p", "free": False}}})
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.price("m") is None
    assert "capabilities only" in caplog.text


# --- per-call and per-second prices ---------------------------------------------------------


def test_per_call_price_is_charged_once_per_cost(prices):
    assert prices.cost("per-req", Usage()) == 0.01
    assert prices.cost("per-req", Usage(input_tokens=10**6, output_tokens=10**6)) == 0.01


def test_per_second_price_uses_the_call_duration(prices):
    assert prices.cost("gpu-box", Usage(), seconds=2.5) == pytest.approx(0.005)
    assert prices.cost("gpu-box", Usage()) == 0.0  # no duration given


def test_per_call_and_per_second_add_to_token_prices(prices):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert prices.cost("mixed", usage, seconds=3.0) == pytest.approx(2.0 + 0.5 + 0.3)


def test_resolved_price_carries_call_and_second_rates(prices):
    resolved = prices.price("mixed")
    assert (resolved.per_call, resolved.per_second, resolved.free) == (0.5, 0.1, False)
    assert prices.price("gpu-box").input == 0.0  # no token prices: tokens cost nothing


def test_token_only_models_keep_their_cost_and_ignore_seconds():
    prices = Prices.default()
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert prices.cost("gpt-4.1", usage, seconds=99.0) == prices.cost("gpt-4.1", usage) == 10.0
    assert prices.cost("gpt-4.1", usage, seconds=-1.0) == 10.0  # duration unused, not checked


@pytest.mark.parametrize("seconds", [-0.5, math.nan, math.inf])
def test_bad_duration_for_a_per_second_price_is_a_config_error(prices, seconds):
    with pytest.raises(ConfigError, match="seconds"):
        prices.cost("gpu-box", Usage(), seconds=seconds)


@pytest.mark.parametrize("field", ["per_call", "per_second"])
def test_negative_call_or_second_price_is_rejected(field):
    with pytest.raises(ConfigError, match=field):
        Prices.load({"version": "v", "models": {"m": {"provider": "p", field: -0.1}}})


def test_per_call_with_tiers_is_allowed():
    tiers = [{"max_prompt_tokens": None, "input": 1.0, "output": 1.0}]
    prices = Prices.load(
        {"version": "v", "models": {"m": {"provider": "p", "tiers": tiers, "per_call": 0.2}}}
    )
    assert prices.cost("m", Usage(input_tokens=1_000_000)) == pytest.approx(1.2)


def test_per_call_on_a_family_key_is_rejected():
    with pytest.raises(ConfigError, match="caps-only"):
        Prices.load({"version": "v", "models": {"acme-": {"provider": "p", "per_call": 0.1}}})


# --- an empty table -------------------------------------------------------------------------


def test_empty_table_prices_nothing_and_merges_your_models(caplog):
    empty = Prices.empty()
    assert empty.version == "empty"
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert empty.cost("gpt-4.1", Usage(input_tokens=1)) is None
    assert "no price for model 'gpt-4.1'" in caplog.text
    mine = empty.merged({"version": "mine", "models": OWN["models"]})
    assert mine.version == "empty+mine"
    assert mine.cost("per-req", Usage()) == 0.01


def test_empty_table_takes_a_version():
    assert Prices.empty("acme-1").version == "acme-1"
    with pytest.raises(ConfigError, match="version"):
        Prices.empty("")
