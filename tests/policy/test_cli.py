"""`agent-adapter policy check`: valid files print their rules, invalid ones exit 2."""

from __future__ import annotations

import argparse
import builtins
import io
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.cli.policy import add_policy_parser, run_policy
from agent_adapter.policy import PolicyError

POLICY = """\
environments: [dev, prod]
default: {dev: allow, prod: deny}
rules:
  - tools: lookup_*
    allow: true
  - tools: refund
    allow: true
    teams: support
    users: [ann@example.com]
    confirm: "amount > 100"
  - tools: wipe_orders
    allow: false
guards:
  - guard: pii
    tools: send_note
budgets:
  - max_usd: 5
    scope: user
    per: day
    store: sqlite:spend.db
rate_limits:
  - kind: llm
    calls: 5
    per: 1
limits: {max_calls: 40}
memory: {retention_days: 30, save: confirm}
sandbox: {require_isolated: true, deny_network: true}
relax: [memory.retention_days]
"""


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-adapter")
    add_policy_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(argv)


def test_check_prints_the_effective_rules(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY, encoding="utf-8")
    args = parse("policy", "check", str(path), "--env", "prod", "--allow-relax", "--tool", "refund")
    out = io.StringIO()
    assert run_policy(args, out) == 0
    text = out.getvalue()
    assert "valid (environment prod, default deny)" in text
    assert "rules[1]: tools refund; allow; confirm when amount > 100; teams support" in text
    assert "1 listed user(s)" in text and "ann@example.com" not in text
    assert "rules[2]: tools wipe_orders; deny" in text
    assert "guards[0]: guard pii (block on args, result)" in text
    assert "budgets[0]: budget budget 5 USD per user per day, stop, store sqlite" in text
    assert "spend.db" not in text
    assert "rate limit llm: 5 per 1 s" in text
    assert "limits: 40 calls per run" in text
    assert "memory: save confirm, retention 30 days" in text
    assert "sandbox: isolated required, network denied" in text
    assert "relaxations: memory.retention_days" in text
    assert "tool refund:\n  allowed for some callers" in text


def test_check_a_plain_policy(tmp_path: Path) -> None:
    path = tmp_path / "p.toml"
    path.write_text(
        '[[rules]]\ntools = "a"\nconfirm = true\n[memory]\nsave = "deny"\n[sandbox]\n',
        encoding="utf-8",
    )
    out = io.StringIO()
    assert run_policy(parse("policy", "check", str(path)), out) == 0
    text = out.getvalue()
    assert "environment (none), default allow" in text
    assert "rules[0]: tools a; confirm always" in text
    assert "memory: save deny, retention unchanged" in text
    assert "sandbox: isolated optional, network as configured" in text


def test_check_refuses_an_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("rules:\n  - tools: a\n    confirm: false\n", encoding="utf-8")
    with pytest.raises(PolicyError) as caught:
        run_policy(parse("policy", "check", str(path)), io.StringIO())
    assert caught.value.line == 3
    assert caught.value.key == "rules[0].confirm"


def test_check_yaml_without_pyyaml_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("default: deny\n", encoding="utf-8")
    real = builtins.__import__

    def no_yaml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ImportError("no yaml")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    assert run_policy(parse("policy", "check", str(path)), io.StringIO()) == 2
    assert "pyyaml" in capsys.readouterr().err


def test_check_needs_an_action() -> None:
    with pytest.raises(SystemExit):
        parse("policy")
