"""Context windows and reply ceilings in the packaged price table, and the rows added with them."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

import agent_adapter.cost.prices as prices_module
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ModelCaps, Usage
from agent_adapter.cost import Prices

DEFAULT = Path(prices_module.__file__).with_name("prices_default.json")
VENDOR_PAGES = (
    "https://developers.openai.com/api/docs/models/",
    "https://aistudio.google.com/docs/models/",
    "https://platform.claude.com/docs/en/about-claude/models/overview",
)
NO_WINDOW = {"gpt-5.6", "claude-"}  # no vendor page for the id; a caps-only family


@pytest.fixture
def prices():
    return Prices.default()


def _rows() -> dict[str, dict]:
    return json.loads(DEFAULT.read_text(encoding="utf-8"))["models"]


def test_every_cited_row_has_a_window_a_ceiling_and_its_vendor_page():
    for key, row in _rows().items():
        if key in NO_WINDOW:
            assert "context_window" not in row.get("caps", {}), key
            continue
        caps = row["caps"]
        assert caps["max_output_tokens"] < caps["context_window"], key
        assert row["source"].startswith(VENDOR_PAGES), key
        if row["source"].startswith(VENDOR_PAGES[:2]):
            assert row["source"].endswith(f"/{key}"), key  # the model's own page


@pytest.mark.parametrize(
    ("model", "window", "ceiling"),
    [
        ("gpt-4.1", 1_047_576, 32_768),
        ("gpt-4.1-2025-04-14", 1_047_576, 32_768),  # an alias reads its row's caps
        ("gpt-5", 400_000, 128_000),
        ("gpt-5-2025-08-07", 400_000, 128_000),  # a snapshot id
        ("gpt-5.4", 1_050_000, 128_000),
        ("gpt-5.4-mini", 400_000, 128_000),
        ("gpt-6-astra", 1_050_000, 128_000),
        ("gemini-2.5-pro", 1_048_576, 65_536),
        ("gemini-4-argon", 1_048_576, 65_536),
        ("gemini-3.1-flash-lite", 1_048_576, 65_536),
        ("claude-opus-5-5", 1_000_000, 128_000),
        ("claude-haiku-5-5-20261007", 1_000_000, 128_000),
    ],
)
def test_known_models_carry_their_vendor_window(prices, model, window, ceiling):
    caps = prices.caps(model)
    assert (caps.context_window, caps.max_output_tokens) == (window, ceiling)
    assert caps.prompt_window() == window - ceiling


@pytest.mark.parametrize("model", ["gpt-5.6", "claude-sonnet-4-5", "my-house-model"])
def test_models_without_a_cited_window_have_none(prices, model):
    caps = prices.caps(model)
    assert caps.context_window is None and caps.prompt_window() is None


@pytest.mark.parametrize(
    ("model", "inp", "cached", "out"),
    [
        ("gpt-5.6-sol", 4.00, 0.40, 20.00),
        ("gpt-5.6-terra", 2.00, 0.20, 12.00),
        ("gpt-5.6-luna", 0.20, 0.02, 1.20),
        ("gpt-5.5", 5.00, 0.50, 30.00),
        ("gpt-5.4-mini", 0.75, 0.075, 4.50),
        ("gpt-6-astra", 10.00, 1.00, 50.00),
        ("gpt-6.1-sol", 2.00, 0.10, 10.00),
        ("gpt-6.1-terra", 1.00, 0.05, 5.00),
        ("gpt-6-luna", 0.10, 0.01, 0.50),
        ("gemini-3.5-flash-lite", 0.30, 0.03, 2.50),
        ("gemini-3.1-flash-lite", 0.25, 0.025, 1.50),
        ("claude-fable-5-1", 10.00, 0.25, 50.00),
        ("claude-opus-5-5", 4.00, 0.20, 20.00),
        ("claude-sonnet-5-5", 2.00, 0.10, 10.00),
    ],
)
def test_added_rows_have_their_list_prices(prices, model, inp, cached, out):
    price = prices.price(model)
    assert (price.input, price.cached_input, price.output) == (inp, cached, out)


def test_gpt_5_6_variants_are_their_own_rows(prices):
    assert prices.resolve("gpt-5.6-sol") == "gpt-5.6-sol"
    assert prices.resolve("gpt-5.6-luna") == "gpt-5.6-luna"
    assert prices.resolve("gpt-5.6") == "gpt-5.6"


def test_gemini_3_6_flash_promo_periods(prices):
    early = prices.price("gemini-3.6-flash", at=date(2026, 12, 31))
    late = prices.price("gemini-3.6-flash", at=date(2027, 1, 1))
    assert (early.input, early.output) == (0.75, 3.75)
    assert (late.input, late.output) == (1.50, 7.50)
    storage = "gemini:gemini-3.6-flash:cache_storage"
    assert prices.unit_cost(storage, 1.0, at=date(2026, 12, 1)) == pytest.approx(0.50)
    assert prices.unit_cost(storage, 1.0, at=date(2027, 1, 1)) == pytest.approx(1.00)


def test_claude_rows_keep_the_family_cache_caps(prices):
    for model in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        caps = prices.caps(model)
        assert caps.cache_control is True and caps.min_cache_tokens == 1024, model


def test_claude_cache_writes_are_priced_by_ttl(prices):
    usage = Usage(
        input_tokens=1_000,
        cached_input_tokens=2_000,
        cache_write_input_tokens=3_000,
        cache_write_by_ttl={"5m": 1_000, "1h": 1_000},
        output_tokens=500,
    )
    # 1,000 written without a TTL label are billed at the 5m rate.
    expected = (1_000 * 4.00 + 2_000 * 0.20 + 2_000 * 5.00 + 1_000 * 8.00 + 500 * 20.00) / 1e6
    assert prices.cost("claude-opus-5-5", usage) == pytest.approx(expected)


def test_claude_haiku_tiers_switch_above_100k_prompt_tokens(prices):
    small = prices.price("claude-haiku-5-5", prompt_tokens=100_000)
    large = prices.price("claude-haiku-5-5", prompt_tokens=100_001)
    assert (small.input, small.cached_input, small.output) == (0.10, 0.01, 0.50)
    assert dict(small.cache_write_by_ttl) == {"5m": 0.125, "1h": 0.20}
    assert (large.input, large.cached_input, large.output) == (0.50, 0.05, 2.50)
    assert dict(large.cache_write_by_ttl) == {"5m": 0.625, "1h": 1.00}


def test_model_caps_rejects_a_ceiling_at_or_over_the_window():
    with pytest.raises(ValidationError, match="less than context_window"):
        ModelCaps(context_window=1_000, max_output_tokens=1_000)
    assert ModelCaps(context_window=1_000, max_output_tokens=999).prompt_window() == 1
    assert ModelCaps(context_window=1_000).prompt_window() == 1_000  # no ceiling: all of it
    assert ModelCaps(max_output_tokens=500).prompt_window() is None


def test_a_user_table_with_a_bad_ceiling_is_a_config_error():
    row = {"provider": "house", "input": 1.0, "output": 2.0}
    bad = {
        "version": "mine",
        "models": {"m": {**row, "caps": {"context_window": 10, "max_output_tokens": 10}}},
    }
    with pytest.raises(ConfigError, match="less than context_window"):
        Prices.load(bad)
    good = {"version": "mine", "models": {"m": {**row, "source": "https://example.com/m"}}}
    assert Prices.load(good).caps("m").context_window is None
