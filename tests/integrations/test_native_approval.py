"""Shared steps of the native approval bridges, without any framework installed."""

import logging

import pytest

from agent_adapter.core.errors import ToolArgsInvalid
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec.args import dump_args
from agent_adapter.integrations import _native_approval as native_approval
from agent_adapter.integrations._native_approval import Grant, issue_tokens, validated_call
from agent_adapter.integrations._native_gate import TokenTable

registry = Registry()


@registry.tool
def pay(amount: int, to: str = "acme") -> str:
    """Pay an amount."""
    return f"paid {amount} to {to}"


def test_validated_call_dumps_validated_arguments_in_json_mode():
    call = validated_call(registry.get("pay"), {"amount": "7"}, "c1")
    assert (call.name, call.arguments, call.id) == ("pay", {"amount": 7, "to": "acme"}, "c1")


def test_validated_call_raises_tool_args_invalid():
    with pytest.raises(ToolArgsInvalid):
        validated_call(registry.get("pay"), {"amount": "lots"}, "c1")


def test_dump_args_is_the_executor_view_of_the_arguments():
    validated = registry.get("pay").validate_args({"amount": 3})
    assert dump_args(validated) == {"amount": 3, "to": "acme"}


def _grants(count: int) -> list[Grant]:
    approve = ApprovalDecision(approved=True)
    return [Grant("pay", {"amount": n}, approve, f"c{n}", "run_1") for n in range(count)]


def test_issue_tokens_warns_with_a_count_when_a_resolve_evicts_tokens(caplog):
    tokens = TokenTable(max_tokens=2)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.integrations._native_approval"):
        issue_tokens(tokens, _grants(3), integration="demo")
    assert len(tokens) == 2 and tokens.evicted == 1
    assert "demo: this resolve evicted 1 older approval token(s)" in caplog.text


def test_issue_tokens_caps_one_resolve(monkeypatch, caplog):
    monkeypatch.setattr(native_approval, "MAX_TOKENS_PER_RESOLVE", 2)
    tokens = TokenTable()
    with caplog.at_level(logging.WARNING, logger="agent_adapter.integrations._native_approval"):
        issue_tokens(tokens, _grants(5), integration="demo")
    assert len(tokens) == 2 and tokens.evicted == 0
    assert "5 approved calls in one resolve; only the first 2 get a token" in caplog.text
    assert tokens.take("pay", {"amount": 4}, call_id="c4", run_id="run_1") is None


def test_issue_tokens_is_quiet_when_nothing_is_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        issue_tokens(TokenTable(), _grants(2), integration="demo")
    assert caplog.text == ""
