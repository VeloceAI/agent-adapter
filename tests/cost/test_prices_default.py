import json
import logging
from datetime import date
from pathlib import Path

import pytest

import agent_adapter.cost.prices as prices_module
from agent_adapter.core.types import Usage
from agent_adapter.cost import Prices

DEFAULT = Path(prices_module.__file__).with_name("prices_default.json")


@pytest.fixture
def prices():
    return Prices.default()


def test_default_is_cached_and_versioned(prices):
    assert Prices.default() is prices
    assert prices.version == "2026-10-29"


def test_default_file_is_package_data():
    data = json.loads(DEFAULT.read_text(encoding="utf-8"))
    assert data["currency"] == "USD" and data["unit"] == "per_1M_tokens"


@pytest.mark.parametrize(
    ("model", "inp", "cached", "out"),
    [
        ("gpt-4.1", 2.00, 0.50, 8.00),
        ("gpt-4.1-mini", 0.40, 0.10, 1.60),
        ("gpt-5", 1.25, 0.125, 10.00),
        ("gpt-5-mini", 0.25, 0.025, 2.00),
        ("gpt-5.1", 1.25, 0.125, 10.00),
        ("gpt-5.2", 1.75, 0.175, 14.00),
        ("gpt-5.4", 2.50, 0.25, 15.00),
        ("gpt-5.6", 4.00, 0.40, 20.00),
        ("gemini-2.5-flash", 0.30, 0.03, 2.50),
        ("gemini-2.5-flash-lite", 0.10, 0.01, 0.40),
        ("gemini-3-flash-preview", 0.50, 0.05, 3.00),
    ],
)
def test_r4_flat_rows(prices, model, inp, cached, out):
    price = prices.price(model)
    assert (price.input, price.cached_input, price.output) == (inp, cached, out)


def test_gpt_4_1_dated_id_is_an_alias(prices):
    assert prices.resolve("gpt-4.1-2025-04-14") == "gpt-4.1"


def test_gpt_5_6_has_no_cache_write_prices(prices):
    price = prices.price("gpt-5.6")
    assert price.cache_write == price.input
    assert dict(price.cache_write_by_ttl) == {}
    assert prices.caps("gpt-5.6").prompt_cache_options is True


def test_openai_cache_writes_are_billed_once(prices):
    # Normalised OpenAI usage: prompt 12000 = 4000 cached + 7000 written + 1000 uncached.
    usage = Usage(
        input_tokens=1_000,
        cached_input_tokens=4_000,
        cache_write_input_tokens=7_000,
        output_tokens=500,
    )
    expected = ((12_000 - 4_000) * 2.00 + 4_000 * 0.50 + 500 * 8.00) / 1e6
    assert prices.cost("gpt-4.1", usage) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4.1-2025-04-14", "gpt-4.1"),
        ("gpt-5-2025-08-07", "gpt-5"),
        ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"),
        ("gemini-2.5-flash-001", "gemini-2.5-flash"),
        ("gemini-2.5-flash-latest", "gemini-2.5-flash"),
    ],
)
def test_snapshot_suffixes_resolve(prices, model, expected):
    assert prices.resolve(model) == expected


@pytest.mark.parametrize(
    "model", ["gpt-5.6-luna", "gpt-5-pro", "gpt-5-nano", "gpt-4.1-nano", "gpt-5-1", "gpt-5-12"]
)
def test_variants_without_a_row_are_unknown(prices, model, caplog):
    prices_module._WARNED.discard(f"model:{model}")
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.resolve(model) is None
        assert prices.cost(model, Usage(input_tokens=1)) is None
        assert prices.cost(model, Usage(input_tokens=1)) is None
    assert caplog.text.count(f"no price for model {model!r}") == 1
    assert prices.caps(model) == prices_module.ModelCaps()


@pytest.mark.parametrize(
    ("model", "small", "large"),
    [
        ("gemini-2.5-pro", (1.25, 0.125, 10.00), (2.50, 0.25, 15.00)),
        ("gemini-3.1-pro-preview", (2.00, 0.20, 12.00), (4.00, 0.40, 18.00)),
    ],
)
def test_gemini_pro_tiers(prices, model, small, large):
    lo = prices.price(model, prompt_tokens=200_000)
    hi = prices.price(model, prompt_tokens=200_001)
    assert (lo.input, lo.cached_input, lo.output) == small
    assert (hi.input, hi.cached_input, hi.output) == large


@pytest.mark.parametrize(
    ("model", "day", "expected"),
    [
        ("gemini-3.8-flash", date(2026, 12, 31), (0.75, 0.075, 3.75)),
        ("gemini-3.8-flash", date(2027, 1, 1), (1.50, 0.15, 7.50)),
        ("gemini-4-argon", date(2026, 11, 30), (2.00, 0.10, 10.00)),
        ("gemini-4-argon", date(2026, 12, 1), (4.00, 0.20, 20.00)),
    ],
)
def test_gemini_promo_periods(prices, model, day, expected):
    price = prices.price(model, at=day)
    assert (price.input, price.cached_input, price.output) == expected


@pytest.mark.parametrize(
    ("item", "day", "rate"),
    [
        ("gemini:gemini-2.5-pro:cache_storage", None, 4.50),
        ("gemini:gemini-2.5-flash:cache_storage", None, 1.00),
        ("gemini:gemini-3.1-pro-preview:cache_storage", None, 4.50),
        ("gemini:gemini-3.8-flash:cache_storage", date(2026, 12, 1), 0.50),
        ("gemini:gemini-3.8-flash:cache_storage", date(2027, 1, 1), 1.00),
    ],
)
def test_gemini_cache_storage(prices, item, day, rate):
    assert prices.unit_cost(item, 1.0, at=day) == pytest.approx(rate)


def test_gemini_caps(prices):
    assert prices.caps("gemini-2.5-pro").explicit_cache is True
    assert prices.caps("gemini-2.5-pro").min_cache_tokens == 2048
    assert prices.caps("gemini-4-argon").explicit_cache is False
    assert prices.caps("gemini-4-argon").min_cache_tokens == 4096


def test_anthropic_is_caps_only(prices):
    assert prices.price("claude-sonnet-4-5") is None
    assert prices.cost("claude-sonnet-4-5", Usage(input_tokens=1)) is None
    caps = prices.caps("claude-sonnet-4-5")
    assert caps.cache_control is True and caps.min_cache_tokens == 1024


def test_no_uncited_rows(prices):
    data = json.loads(DEFAULT.read_text(encoding="utf-8"))
    assert "gemini-3-pro-preview" not in data["models"]
    assert prices.resolve("gemini-3-pro-preview") is None
    assert data["units"] == {}
    assert not any(key.startswith("gpt-5.6-") for key in data["models"])
    assert prices.resolve("gpt-5.6-sol") is None  # not priced at gpt-5.6 rates
    providers = {entry["provider"] for entry in data["models"].values()}
    assert providers == {"openai", "gemini", "anthropic"}
    priced_anthropic = [
        key
        for key, entry in data["models"].items()
        if entry["provider"] == "anthropic" and ({"input", "tiers", "periods"} & set(entry))
    ]
    assert priced_anthropic == []


def test_gemini_2_5_pro_cost_example(prices):
    usage = Usage(input_tokens=100_000, cached_input_tokens=50_000, output_tokens=10_000)
    expected = (100_000 * 1.25 + 50_000 * 0.125 + 10_000 * 10.00) / 1e6
    assert prices.cost("gemini-2.5-pro", usage) == pytest.approx(expected)
