"""load_policy: files, dicts and specs; precise, fail-closed errors with file, line and key."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.policy import Policy, PolicyError, PolicySpec, load_policy

YAML = """\
version: 1
environments: [dev, prod]
default:
  dev: allow
  prod: deny
rules:
  - tools: ["refund_*"]
    allow: true
    confirm: "amount > 100"
  - tools: wipe_all
    allow: false
budgets:
  - max_usd: 5
    scope: user
    per: day
"""

TOML = """\
environments = ["dev", "prod"]
default = "deny"

[[rules]]
tools = ["lookup_*"]
allow = true

[[rules]]
tools = ["refund"]
confirm = "amount > 100"
colour = "red"
"""


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_policy_reads_yaml(tmp_path: Path) -> None:
    policy = load_policy(write(tmp_path, "policy.yaml", YAML), env="prod")
    assert isinstance(policy, Policy)
    assert policy.env == "prod"
    assert policy.default == "deny"
    assert [where for where, _ in policy.rules] == ["rules[0]", "rules[1]"]
    assert policy.source.endswith("policy.yaml")
    assert "prod" in repr(policy)


def test_load_policy_reads_toml_and_json(tmp_path: Path) -> None:
    toml = TOML.replace('colour = "red"\n', "")
    assert load_policy(write(tmp_path, "p.toml", toml), env="dev").default == "deny"
    data = {"rules": [{"tools": ["a"], "allow": False}]}
    assert load_policy(write(tmp_path, "p.json", json.dumps(data))).rules[0][1].allow is False


def test_load_policy_takes_a_dict_or_a_spec() -> None:
    spec = PolicySpec(default="deny")
    assert load_policy(spec).spec is spec
    assert load_policy({"default": "deny"}).default == "deny"


def test_unknown_key_names_line_and_key_in_toml(tmp_path: Path) -> None:
    path = write(tmp_path, "p.toml", TOML)
    with pytest.raises(PolicyError) as caught:
        load_policy(path, env="dev")
    error = caught.value
    assert error.key == "rules[1].colour"
    assert error.line == 11
    assert f"{path}:11" in str(error)
    assert isinstance(error, ConfigError)


def test_bad_value_names_the_yaml_line(tmp_path: Path) -> None:
    text = YAML.replace('confirm: "amount > 100"', 'confirm: "amount > (100"')
    with pytest.raises(PolicyError) as caught:
        load_policy(write(tmp_path, "p.yml", text), env="prod")
    assert caught.value.key == "rules[0].confirm"
    assert caught.value.line == 9


def test_error_never_echoes_the_value(tmp_path: Path) -> None:
    text = YAML.replace("max_usd: 5", "max_usd: secret-value-xyz")
    with pytest.raises(PolicyError) as caught:
        load_policy(write(tmp_path, "p.yaml", text), env="prod")
    assert "secret-value-xyz" not in str(caught.value)
    assert caught.value.key == "budgets[0].max_usd"


def test_several_errors_are_counted() -> None:
    with pytest.raises(PolicyError, match=r"and 1 more"):
        load_policy({"bogus": 1, "other": 2})


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("rules: [\n", "not valid YAML"),
        ("default: allow\ndefault: deny\n", "appears twice"),
        ("- a\n- b\n", "must be a mapping"),
    ],
)
def test_bad_yaml_fails(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(PolicyError, match=match):
        load_policy(write(tmp_path, "p.yaml", text))


def test_bad_toml_and_json_fail(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="not valid TOML") as caught:
        load_policy(write(tmp_path, "p.toml", "a = \n"))
    assert caught.value.line == 1
    with pytest.raises(PolicyError, match="not valid JSON"):
        load_policy(write(tmp_path, "p.json", "{"))
    with pytest.raises(PolicyError, match="appears twice"):
        load_policy(write(tmp_path, "p.json", '{"default": "allow", "default": "deny"}'))


def test_file_problems_fail(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match=r"\.yaml, \.yml, \.toml or \.json"):
        load_policy(write(tmp_path, "p.ini", "x"))
    with pytest.raises(PolicyError, match="cannot read"):
        load_policy(tmp_path / "missing.yaml")
    big = tmp_path / "big.json"
    big.write_bytes(b" " * 1_000_001)
    with pytest.raises(PolicyError, match="at most"):
        load_policy(big)
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe")
    with pytest.raises(PolicyError, match="UTF-8"):
        load_policy(bad)
    with pytest.raises(PolicyError, match="mapping"):
        load_policy(write(tmp_path, "list.json", "[1]"))


def test_not_a_mapping_fails() -> None:
    with pytest.raises(PolicyError, match="mapping"):
        load_policy([1, 2])  # type: ignore[arg-type]


def test_empty_yaml_is_not_a_policy(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="mapping"):
        load_policy(write(tmp_path, "p.yaml", ""))


def test_yaml_without_pyyaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def no_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("no yaml")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    with pytest.raises(MissingDependency, match="pyyaml"):
        load_policy(write(tmp_path, "p.yaml", "default: deny\n"))


@pytest.mark.parametrize(
    ("data", "key"),
    [
        ({"rules": [{"tools": "a", "confirm": False}]}, "rules[0].confirm"),
        ({"rules": [{"tools": "a"}]}, "rules[0]"),
        ({"rules": [{"tools": []}]}, "rules[0].tools"),
        ({"rules": [{"tools": [""], "allow": True}]}, "rules[0].tools"),
        ({"rules": [{"allow": "yes"}]}, "rules[0].allow"),
        ({"version": 2}, "version"),
        ({"default": "maybe"}, "default"),
        ({"budgets": [{"max_usd": -1}]}, "budgets[0].max_usd"),
        ({"budgets": [{"max_usd": float("inf")}]}, "budgets[0].max_usd"),
        ({"rate_limits": [{"kind": "tool", "calls": 0, "per": 1}]}, "rate_limits[0].calls"),
        ({"environments": ["a", "a"]}, "environments"),
        ({"memory": {"save": "sometimes"}}, "memory.save"),
        ({"relax": ["everything"]}, "relax[0]"),
    ],
)
def test_bad_settings_fail_at_load(data: dict[str, Any], key: str) -> None:
    with pytest.raises(PolicyError) as caught:
        load_policy(data)
    assert caught.value.key == key


def test_environments_must_be_declared_and_known() -> None:
    with pytest.raises(PolicyError, match="list them in `environments`"):
        load_policy({"rules": [{"allow": True, "envs": "prod"}]})
    with pytest.raises(PolicyError, match="unknown environments"):
        load_policy({"environments": ["prod"], "rules": [{"allow": True, "envs": "prd"}]})
    with pytest.raises(PolicyError, match="unknown environments"):
        load_policy({"environments": ["prod"], "default": {"staging": "deny"}})


def test_env_comes_from_aa_env_and_must_be_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"environments": ["dev", "prod"]}
    monkeypatch.setenv("AA_ENV", "prod")
    assert load_policy(data).env == "prod"
    assert load_policy(data, env="dev").env == "dev"
    monkeypatch.delenv("AA_ENV")
    with pytest.raises(PolicyError, match="AA_ENV is not set") as caught:
        load_policy(data)
    assert caught.value.key == "environments"
    with pytest.raises(PolicyError, match="'qa'"):
        load_policy(data, env="qa")


def test_env_missing_from_a_default_map_is_deny() -> None:
    policy = load_policy({"environments": ["dev", "prod"], "default": {"dev": "allow"}}, env="prod")
    assert policy.default == "deny"


def test_relaxations_need_allow_relax() -> None:
    data = {"relax": ["limits.max_calls"], "limits": {"max_calls": 50}}
    with pytest.raises(PolicyError, match="allow_relax=True") as caught:
        load_policy(data)
    assert caught.value.key == "relax"
    assert load_policy(data, allow_relax=True).relax == {"limits.max_calls"}


def test_policy_error_without_a_place() -> None:
    assert str(PolicyError("plain")) == "plain"
    assert str(PolicyError("m", key="rules[0]")) == "at rules[0]: m"
