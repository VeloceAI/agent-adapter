"""explain and describe: the rules for one tool, and the viewer's Tools tab catalogue."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec import Executor
from agent_adapter.exec.defaults import reset_default_executor, set_default_executor
from agent_adapter.policy import Explanation, load_policy
from agent_adapter.viewer import record
from agent_adapter.viewer._scrub import catalogue
from tests.policy.helpers import caller, shop
from tests.viewer.helpers import lines

FULL = {
    "default": "deny",
    "rules": [
        {"tools": "lookup_*", "allow": True},
        {"tools": "refund", "allow": True, "teams": "support", "confirm": "amount > 100"},
        {"tools": "refund", "confirm": True, "users": ["ann@example.com"]},
        {"tools": "wipe_orders", "allow": False, "users": ["bob@example.com"]},
    ],
    "guards": [{"guard": "pii", "tools": "send_note", "action": "approve"}],
    "budgets": [{"max_usd": 5, "scope": "user", "per": "day", "name": "daily"}],
    "rate_limits": [{"kind": "tool", "calls": 10, "per": 1}, {"kind": "llm", "calls": 5, "per": 1}],
    "limits": {"max_calls": 40},
    "memory": {"retention_days": 30, "save": "confirm"},
}


def test_explain_without_a_caller_says_some() -> None:
    policy = load_policy(FULL)
    found = policy.explain("refund")
    assert isinstance(found, Explanation)
    assert found.allowed is None
    text = str(found)
    assert "allowed for some callers (rules[1] allows some callers)" in text
    assert "approval: when amount > 100 (rules[1]) [tools refund; teams support]" in text
    assert "approval: always (rules[2]) [tools refund; 1 listed user(s)]" in text
    assert "ann@example.com" not in text
    assert "budget daily: 5 USD per user per day, stop (budgets[0])" in text
    assert "rate limit llm: 5 per 1 s" in text
    assert "run limit: 40 calls (limits)" in text


def test_explain_for_a_caller_is_decided() -> None:
    policy = load_policy(FULL)
    support = policy.explain("refund", caller(team="support"))
    assert support.allowed is True
    assert "approval: when amount > 100 (rules[1])" in support.lines
    sales = policy.explain("refund", caller(team="sales"))
    assert sales.allowed is False
    assert not [line for line in sales.lines if line.startswith("approval")]
    assert policy.explain("wipe_orders", caller(user="bob@example.com")).allowed is False
    assert policy.explain("lookup_order").allowed is True


def test_explain_a_spec_shows_code_approval_guards_and_memory() -> None:
    policy = load_policy(FULL)
    reg = shop()
    close = policy.explain(reg.get("close_shop"))
    assert "approval: always (set in code; a policy cannot remove it)" in close.lines
    note = policy.explain(reg.get("send_note"))
    assert "approval: when guard pii_guard flags the arguments (guards[0])" in note.lines
    assert "guard pii: approve on args (guards[0])" in note.lines
    assert "rate limit tool: 10 per 1 s" in note.lines
    memory = policy.explain("remember")
    assert "memory: save confirm; kept 30 days (memory)" in memory.lines
    assert "approval: always (memory.save)" in memory.lines
    assert load_policy({}).explain("recall").lines == ("allowed (default allow)",)


def test_explain_a_code_rule_and_a_maybe_deny() -> None:
    from agent_adapter.core.tool import build_spec

    def pick(n: int) -> int:
        """Pick."""
        return n

    spec = build_spec(pick, confirm=lambda call, ctx: True)
    policy = load_policy({"rules": [{"tools": "pick", "allow": False, "teams": "x"}]})
    found = policy.explain(spec)
    assert found.allowed is None
    assert "approval: by the code's own rule" in found.lines
    kinds = load_policy({"rules": [{"kinds": "code", "allow": False}]})
    assert kinds.explain("pick").allowed is None
    assert kinds.explain(spec).allowed is True


def test_catalogue_shows_the_effective_policy() -> None:
    policy = load_policy({"rules": [{"tools": "refund", "confirm": "amount > 100"}]})
    tools = {tool["name"]: tool for tool in catalogue(shop(), policy)}
    assert tools["refund"]["confirm"] == "rule"
    assert "approval: when amount > 100 (rules[0])" in tools["refund"]["policy"]
    assert tools["lookup_order"]["confirm"] == "never"
    assert "policy" not in catalogue(shop())[0]


class Broken:
    """A policy whose lines fail: the catalogue falls back to the plain tool."""

    def hooks(self) -> tuple[Any, ...]:
        return ()

    def limits(self, current: Any) -> Any:
        return current

    def effective(self, spec: ToolSpec) -> ToolSpec:
        return spec

    def describe(self, spec: ToolSpec) -> list[str]:
        raise RuntimeError("boom")


def test_catalogue_survives_a_failing_policy(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        shown = catalogue(shop(), Broken())
    assert all("policy" not in tool for tool in shown)
    assert "not written" in caplog.text


def _tools_line(folder: Path) -> dict[str, Any]:
    (path,) = [p for p in folder.rglob("*.jsonl")]
    return next(line for line in lines(path) if line.get("kind") == "tools")


async def test_recorded_runs_carry_the_policy(tmp_path: Path) -> None:
    policy = load_policy({"rules": [{"tools": "wipe_orders", "allow": False}]})
    executor = policy.apply(Executor(shop()))
    recorder = record(executor, folder=tmp_path / "runs")
    try:
        async with executor.run_scope():
            pass
        assert recorder.flush()
    finally:
        recorder.stop()
    tools = {tool["name"]: tool for tool in _tools_line(tmp_path / "runs")["tools"]}
    assert "denied (rules[0] denies it)" in tools["wipe_orders"]["policy"]


async def test_recording_the_default_executor_carries_its_policy(tmp_path: Path) -> None:
    policy = load_policy({"rules": [{"tools": "wipe_orders", "allow": False}]})
    executor = policy.apply(Executor(shop()))
    set_default_executor(executor)
    recorder = record(folder=tmp_path / "runs")
    try:
        async with executor.run_scope():
            pass
        assert recorder.flush()
    finally:
        recorder.stop()
        reset_default_executor()
    tools = {tool["name"]: tool for tool in _tools_line(tmp_path / "runs")["tools"]}
    assert tools["wipe_orders"]["policy"][0].startswith("denied")
