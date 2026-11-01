import json

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.cost import Prices


def table(**models):
    return {"version": "t1", "models": models}


FLAT = {"provider": "openai", "input": 2.0, "output": 8.0}


def test_load_from_mapping_keeps_the_version():
    prices = Prices.load(table(m=FLAT))
    assert prices.version == "t1"
    assert prices.resolve("m") == "m"


def test_load_from_a_path_and_a_str(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(table(m=FLAT)), encoding="utf-8")
    assert Prices.load(path).version == "t1"
    assert Prices.load(str(path)).version == "t1"


def test_load_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read price table"):
        Prices.load(tmp_path / "absent.json")


def test_load_invalid_json_is_a_config_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        Prices.load(path)


def test_load_non_object_json_is_a_config_error(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a JSON object"):
        Prices.load(path)


def test_load_rejects_a_non_mapping_source():
    with pytest.raises(ConfigError, match="must be a mapping or a path"):
        Prices.load(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"models": {}}, "version"),
        ({"version": "", "models": {}}, "version"),
        ({"version": "v", "currency": "EUR"}, "currency"),
        ({"version": "v", "unit": "per_1K_tokens"}, "unit"),
        ({"version": "v", "colour": "blue"}, "colour"),
        (table(m={"input": 1.0, "output": 1.0}), "provider"),
        (table(m={"provider": "openai", "input": 1.0}), "input and output"),
        (table(m={"provider": "openai", "cached_input": 1.0}), "input and output"),
        (table(m={**FLAT, "inptu": 1.0}), "inptu"),
        (table(m={**FLAT, "input": -1.0}), "input"),
        (table(m={**FLAT, "caps": {"explicit_cache": True, "warp_drive": True}}), "warp_drive"),
        (table(m={**FLAT, "caps": "yes"}), "caps"),
        (
            table(m={**FLAT, "tiers": [{"max_prompt_tokens": None, "input": 1, "output": 1}]}),
            "either flat prices or tiers",
        ),
        (table(m={"provider": "gemini", "tiers": []}), "at least one tier"),
        (
            table(
                m={
                    "provider": "gemini",
                    "tiers": [{"max_prompt_tokens": 10, "input": 1, "output": 1}],
                }
            ),
            "last tier",
        ),
        (
            table(
                m={
                    "provider": "gemini",
                    "tiers": [
                        {"max_prompt_tokens": 20, "input": 1, "output": 1},
                        {"max_prompt_tokens": 10, "input": 1, "output": 1},
                        {"max_prompt_tokens": None, "input": 1, "output": 1},
                    ],
                }
            ),
            "ascending",
        ),
        (
            table(
                m={
                    "provider": "gemini",
                    "tiers": [
                        {"max_prompt_tokens": None, "input": 1, "output": 1},
                        {"max_prompt_tokens": None, "input": 1, "output": 1},
                    ],
                }
            ),
            "last tier",
        ),
        (
            table(m={**FLAT, "periods": [{"from": "2026-01-01", "input": 1, "output": 1}]}),
            "periods or top-level prices",
        ),
        (table(m={"provider": "gemini", "periods": []}), "at least one period"),
        (table(m={"provider": "gemini", "periods": [{"from": "2026-01-01"}]}), "prices or tiers"),
        (
            table(
                m={
                    "provider": "g",
                    "periods": [
                        {"from": "2026-02-01", "until": "2026-01-01", "input": 1, "output": 1}
                    ],
                }
            ),
            "is before",
        ),
        (
            table(
                m={
                    "provider": "g",
                    "periods": [
                        {"until": "2026-06-30", "input": 1, "output": 1},
                        {"from": "2026-06-30", "input": 2, "output": 2},
                    ],
                }
            ),
            "overlap",
        ),
        (
            table(m={"provider": "g", "periods": [{"from": "soon", "input": 1, "output": 1}]}),
            "from",
        ),
        ({"version": "v", "aliases": {"x": "missing"}, "models": {}}, "alias 'x'"),
        ({"version": "v", "units": {"voyage:r": {"provider": "voyage"}}}, "per_unit"),
        (
            {"version": "v", "units": {"rerank": {"provider": "voyage", "per_unit": 1}}},
            "provider:model",
        ),
    ],
)
def test_invalid_tables_raise_config_error(data, fragment):
    with pytest.raises(ConfigError, match="invalid price table") as info:
        Prices.load(data)
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    "entry",
    [
        {"provider": "openai", "input": 1.0, "output": 1.0},
        {"provider": "g", "tiers": [{"max_prompt_tokens": None, "input": 1, "output": 1}]},
        {"provider": "g", "periods": [{"from": "2026-01-01", "input": 1, "output": 1}]},
    ],
)
def test_priced_family_key_is_rejected(entry):
    with pytest.raises(ConfigError, match="family key 'gpt-' must be caps-only"):
        Prices.load(table(**{"gpt-": entry}))
    with pytest.raises(ConfigError, match="family key 'gpt-' must be caps-only"):
        Prices.default().merged({"models": {"gpt-": entry}})


def test_caps_only_family_key_still_resolves_caps():
    prices = Prices.default().merged(
        {"models": {"mistral-": {"provider": "mistral", "caps": {"reasoning": True}}}}
    )
    assert prices.resolve("mistral-large-3") == "mistral-"
    assert prices.caps("mistral-large-3").reasoning is True
    assert prices.caps("claude-sonnet-4-5").cache_control is True


def test_caps_only_entry_loads():
    prices = Prices.load(
        table(**{"claude-": {"provider": "anthropic", "caps": {"cache_control": True}}})
    )
    assert prices.resolve("claude-x") == "claude-"


def test_note_is_allowed_on_tables_models_and_units():
    data = {
        "version": "v",
        "note": "n",
        "models": {"m": {**FLAT, "note": "n"}},
        "units": {
            "voyage:r": {"provider": "voyage", "per_unit": 1, "unit": "1M_tokens", "note": "n"}
        },
    }
    assert Prices.load(data).version == "v"


def test_load_logs_before_raising(caplog):
    with caplog.at_level("ERROR", logger="agent_adapter.cost.prices"), pytest.raises(ConfigError):
        Prices.load({"models": {}})
    assert "invalid price table" in caplog.text


def test_negative_ttl_price_is_rejected():
    data = table(m={**FLAT, "cache_write_by_ttl": {"5m": -1.0}})
    with pytest.raises(ConfigError, match=r"cache_write_by_ttl\.5m: Input should be greater"):
        Prices.load(data)
