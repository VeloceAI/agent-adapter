import logging
from datetime import date

import pytest

from agent_adapter.core.types import ModelCaps, Usage
from agent_adapter.cost import Prices, ResolvedPrice
from agent_adapter.cost import prices as prices_module

TABLE = {
    "version": "t1",
    "aliases": {"flat-2025-01-01": "flat"},
    "models": {
        "flat": {
            "provider": "openai",
            "input": 2.0,
            "cached_input": 0.5,
            "output": 8.0,
            "caps": {"prompt_cache_key": True},
        },
        "flat-mini": {"provider": "openai", "input": 0.4, "output": 1.6},
        "ttl": {
            "provider": "openai",
            "input": 4.0,
            "cached_input": 0.4,
            "cache_write": 5.0,
            "cache_write_by_ttl": {"30m": 5.0, "12h": 10.0},
            "output": 20.0,
        },
        "tiered": {
            "provider": "gemini",
            "cache_storage_per_hour": 4.5,
            "tiers": [
                {
                    "max_prompt_tokens": 200_000,
                    "input": 1.25,
                    "cached_input": 0.125,
                    "output": 10.0,
                },
                {"max_prompt_tokens": None, "input": 2.5, "cached_input": 0.25, "output": 15.0},
            ],
            "caps": {"explicit_cache": True, "min_cache_tokens": 2048},
        },
        "promo": {
            "provider": "gemini",
            "cache_storage_per_hour": 1.0,
            "periods": [
                {
                    "until": "2026-12-31",
                    "input": 0.75,
                    "output": 3.75,
                    "cache_storage_per_hour": 0.5,
                },
                {"from": "2027-01-01", "input": 1.5, "cached_input": 0.15, "output": 7.5},
            ],
        },
        "window": {
            "provider": "gemini",
            "periods": [
                {
                    "from": "2026-01-01",
                    "until": "2026-06-30",
                    "tiers": [
                        {"max_prompt_tokens": 1000, "input": 1.0, "output": 1.0},
                        {"max_prompt_tokens": None, "input": 2.0, "output": 2.0},
                    ],
                }
            ],
        },
        "claude-opus-4": {"provider": "anthropic", "input": 15.0, "output": 75.0},
        "claude-": {
            "provider": "anthropic",
            "caps": {"cache_control": True, "min_cache_tokens": 1024},
        },
    },
    "units": {"voyage:rerank-2.5": {"provider": "voyage", "per_unit": 0.05, "unit": "1M_tokens"}},
}


@pytest.fixture
def prices():
    return Prices.load(TABLE)


@pytest.fixture(autouse=True)
def fresh_warnings():
    prices_module._WARNED.clear()
    yield
    prices_module._WARNED.clear()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("flat", "flat"),
        ("flat-2025-01-01", "flat"),
        ("flat-mini", "flat-mini"),
        ("flat-mini-2025-04-14", "flat-mini"),
        ("flat-2025-04-14", "flat"),
        ("flat-20250414", "flat"),
        ("flat-001", "flat"),
        ("flat-0613", "flat"),
        ("flat-20250514", "flat"),
        ("flat-1", None),
        ("flat-12", None),
        ("flat-latest", "flat"),
        ("flat-mini-latest", "flat-mini"),
        ("flat-preview", None),
        ("flat-luna", None),
        ("flat-2025-4-14", None),
        ("flat-mini-luna", None),
        ("flat2025", None),
        ("claude-sonnet-4-5", "claude-"),  # a key ending in "-" names a whole family
        ("fla", None),
        ("unknown", None),
        ("", None),
    ],
)
def test_resolve_exact_alias_then_longest_prefix(prices, model, expected):
    assert prices.resolve(model) == expected


def test_price_flat_defaults_cached_and_cache_write_to_input(prices):
    price = prices.price("flat-mini")
    assert price == ResolvedPrice(
        model="flat-mini",
        provider="openai",
        input=0.4,
        cached_input=0.4,
        cache_write=0.4,
        cache_write_by_ttl={},
        output=1.6,
        version="t1",
    )


def test_resolved_price_ttl_map_is_read_only(prices):
    price = prices.price("ttl")
    assert price is not None
    assert dict(price.cache_write_by_ttl) == {"30m": 5.0, "12h": 10.0}
    with pytest.raises(TypeError):
        price.cache_write_by_ttl["5m"] = 1.0  # type: ignore[index]


def test_price_tier_by_prompt_tokens(prices):
    assert prices.price("tiered", prompt_tokens=200_000).input == 1.25
    assert prices.price("tiered", prompt_tokens=200_001).input == 2.5
    assert prices.price("tiered").output == 10.0


@pytest.mark.parametrize(
    ("day", "expected"),
    [(date(2026, 12, 31), 0.75), (date(2027, 1, 1), 1.5), (date(2020, 1, 1), 0.75)],
)
def test_price_period_by_date_inclusive(prices, day, expected):
    assert prices.price("promo", at=day).input == expected


def test_price_period_cached_input_defaults_within_the_period(prices):
    assert prices.price("promo", at=date(2026, 11, 1)).cached_input == 0.75
    assert prices.price("promo", at=date(2027, 2, 1)).cached_input == 0.15


def test_price_period_with_tiers(prices):
    assert prices.price("window", at=date(2026, 3, 1), prompt_tokens=5000).input == 2.0
    assert prices.price("window", at=date(2026, 3, 1), prompt_tokens=10).input == 1.0


def test_price_outside_every_period_is_none_and_warns_once(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.price("window", at=date(2027, 1, 1)) is None
        assert prices.price("window", at=date(2027, 1, 2)) is None
    assert caplog.text.count("no price period") == 1


def test_price_defaults_to_today_utc(prices, monkeypatch):
    monkeypatch.setattr(prices_module, "_today", lambda: date(2027, 6, 1))
    assert prices.price("promo").input == 1.5


def test_unknown_model_is_none_and_warns_once_per_model(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.price("mystery") is None
        assert prices.cost("mystery", Usage(input_tokens=10)) is None
        assert prices.price("other") is None
    assert caplog.text.count("no price for model 'mystery'") == 1
    assert caplog.text.count("no price for model 'other'") == 1


def test_caps_only_entry_has_no_price_but_caps(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.price("claude-opus-5") is None
        assert prices.cost("claude-opus-5", Usage(input_tokens=1)) is None
    assert caplog.text.count("capabilities only") == 1
    assert prices.caps("claude-opus-5") == ModelCaps(cache_control=True, min_cache_tokens=1024)


def test_caps_unknown_model_is_default(prices):
    assert prices.caps("mystery") == ModelCaps()
    assert prices.caps("flat-2025-01-01").prompt_cache_key is True
    assert prices.caps("tiered").min_cache_tokens == 2048


def test_cost_formula_flat(prices):
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=2_000_000, output_tokens=500_000)
    assert prices.cost("flat", usage) == pytest.approx(2.0 + 1.0 + 4.0)


def test_cost_cache_writes_by_ttl_with_remainder_at_the_default_rate(prices):
    usage = Usage(
        cache_write_input_tokens=3_000_000,
        cache_write_by_ttl={"30m": 1_000_000, "12h": 1_000_000},
    )
    # 1M at 5.00 (30m) + 1M at 10.00 (12h) + 1M unlabelled at cache_write 5.00
    assert prices.cost("ttl", usage) == pytest.approx(20.0)


def test_cost_unpriced_ttl_falls_back_to_cache_write(prices):
    usage = Usage(cache_write_input_tokens=1_000_000, cache_write_by_ttl={"5m": 1_000_000})
    assert prices.cost("ttl", usage) == pytest.approx(5.0)


def test_cost_cache_writes_without_a_write_price_are_billed_at_input(prices):
    usage = Usage(input_tokens=1_000_000, cache_write_input_tokens=1_000_000)
    assert prices.cost("flat", usage) == pytest.approx(4.0)


def test_version_like_suffix_does_not_reach_a_shorter_priced_row(prices):
    # claude-opus-4-5 is a different model, not a snapshot of claude-opus-4.
    assert prices.resolve("claude-opus-4") == "claude-opus-4"
    assert prices.resolve("claude-opus-4-5") == "claude-"
    assert prices.price("claude-opus-4-5") is None
    assert prices.resolve("claude-opus-4-20250514") == "claude-opus-4"


def test_unknown_suffix_is_an_unknown_model_with_one_warning(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.cost("flat-luna", Usage(input_tokens=1)) is None
        assert prices.price("flat-luna") is None
    assert caplog.text.count("no price for model 'flat-luna'") == 1
    assert prices.caps("flat-luna") == ModelCaps()


def test_cost_tier_chosen_by_prompt_tokens_including_cached_and_writes(prices):
    small = Usage(input_tokens=100_000, cached_input_tokens=100_000)
    large = Usage(input_tokens=100_000, cached_input_tokens=100_000, cache_write_input_tokens=1)
    assert prices.cost("tiered", small) == pytest.approx((100_000 * 1.25 + 100_000 * 0.125) / 1e6)
    assert prices.cost("tiered", large) == pytest.approx(
        (100_000 * 2.5 + 100_000 * 0.25 + 1 * 2.5) / 1e6
    )


def test_cost_period_by_date(prices):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert prices.cost("promo", usage, at=date(2026, 12, 1)) == pytest.approx(4.5)
    assert prices.cost("promo", usage, at=date(2027, 1, 1)) == pytest.approx(9.0)


def test_cost_adds_units_priced_by_provider_and_model():
    data = {
        "version": "u",
        "models": {"rerank-2.5": {"provider": "voyage", "input": 1.0, "output": 0.0}},
        "units": {"voyage:rerank-2.5": {"provider": "voyage", "per_unit": 0.05}},
    }
    usage = Usage(input_tokens=1_000_000, units=2.0)
    assert Prices.load(data).cost("rerank-2.5", usage) == pytest.approx(1.1)


def test_cost_units_without_a_unit_price_count_zero_and_warn(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.cost("flat", Usage(units=3.0)) == 0.0
    assert "no unit price for 'openai:flat'" in caplog.text


def test_unit_cost_by_item(prices):
    assert prices.unit_cost("voyage:rerank-2.5", 2.0) == pytest.approx(0.1)


def test_unit_cost_unknown_item_is_none_and_warns_once(prices, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cost.prices"):
        assert prices.unit_cost("voyage:rerank-9", 1.0) is None
        assert prices.unit_cost("voyage:rerank-9", 1.0) is None
    assert caplog.text.count("no unit price for 'voyage:rerank-9'") == 1


def test_unit_cost_gemini_cache_storage(prices):
    assert prices.unit_cost("gemini:tiered:cache_storage", 2.0) == pytest.approx(9.0)
    assert prices.unit_cost("gemini:tiered-001:cache_storage", 1.0) == pytest.approx(4.5)


def test_unit_cost_cache_storage_period_override_and_default(prices):
    item = "gemini:promo:cache_storage"
    assert prices.unit_cost(item, 1.0, at=date(2026, 12, 31)) == pytest.approx(0.5)
    assert prices.unit_cost(item, 1.0, at=date(2027, 1, 1)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "item",
    [
        "gemini:flat:cache_storage",  # provider mismatch
        "gemini:mystery:cache_storage",  # unknown model
        "openai:flat:cache_storage",  # no storage price
        "gemini:window:cache_storage",  # no storage price in its period or entry
        "cache_storage",
    ],
)
def test_unit_cost_cache_storage_unpriced_is_none(prices, item):
    assert prices.unit_cost(item, 1.0) is None


def test_merged_replaces_per_model_and_adds_new(prices):
    merged = prices.merged(
        {
            "version": "acme-1",
            "aliases": {"house": "flat-mini"},
            "models": {"flat": {"provider": "openai", "input": 1.0, "output": 2.0}},
            "units": {"cohere:rerank": {"provider": "cohere", "per_unit": 2.0}},
        }
    )
    assert merged.version == "t1+acme-1"
    assert merged.price("flat").cached_input == 1.0  # whole entry replaced, not field-merged
    assert merged.caps("flat") == ModelCaps()
    assert merged.price("tiered") is not None
    assert merged.resolve("house") == "flat-mini"
    assert merged.unit_cost("cohere:rerank", 1.0) == 2.0
    assert merged.unit_cost("voyage:rerank-2.5", 1.0) == 0.05
    assert prices.price("flat").input == 2.0  # original unchanged


def test_version_is_read_only(prices):
    with pytest.raises(AttributeError):
        prices.version = "x"  # type: ignore[misc]


def test_merged_without_version_marks_custom(prices):
    assert prices.merged({"models": {}}).version == "t1+custom"


def test_merged_with_prices(prices):
    other = Prices.load(
        {"version": "o", "models": {"new": {"provider": "x", "input": 1, "output": 1}}}
    )
    merged = prices.merged(other)
    assert merged.version == "t1+o"
    assert merged.resolve("new") == "new"


def test_merged_validates_overrides(prices):
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError):
        prices.merged({"models": {"bad": {"provider": "x", "input": 1}}})
    with pytest.raises(ConfigError, match="alias 'a'"):
        prices.merged({"aliases": {"a": "nowhere"}})
