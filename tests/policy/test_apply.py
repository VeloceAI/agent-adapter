"""Policy.apply on the plain executor: access, approvals, limits and stacking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ApprovalDenied, RunLimitExceeded, ToolUsageError
from agent_adapter.core.types import ToolCall
from agent_adapter.exec import Executor, RunLimits
from agent_adapter.hooks import LoggingHook
from agent_adapter.policy import PolicyDenied, load_policy
from agent_adapter.policy._hooks import PolicyConfirm
from tests.policy.helpers import Approvals, caller, shop

REFUND = {"order_id": "o-1", "amount": 50.0}
BIG_REFUND = {"order_id": "o-1", "amount": 500.0}


def ex(data: dict[str, Any], *, approval: Any = None, env: str = "", **options: Any) -> Executor:
    return load_policy(data, env=env).apply(Executor(shop(), approval=approval, **options))


async def test_default_allow_with_a_deny_rule() -> None:
    executor = ex({"rules": [{"tools": "wipe_*", "allow": False, "reason": "not from agents"}]})
    assert await executor.run("lookup_order", {"order_id": "o-1"}) == {
        "order_id": "o-1",
        "status": "shipped",
    }
    with pytest.raises(PolicyDenied, match="policy: not from agents") as caught:
        await executor.run("wipe_orders")
    assert isinstance(caught.value, ApprovalDenied)


async def test_default_deny_allows_only_listed_tools() -> None:
    executor = ex({"default": "deny", "rules": [{"tools": "lookup_*", "allow": True}]})
    assert (await executor.run("lookup_order", {"order_id": "o"}))["status"] == "shipped"
    result = await executor.run_call(ToolCall(name="refund", arguments=REFUND))
    assert result.error is not None
    assert result.error.kind == "approval-denied"
    assert "default deny" in result.error.message


async def test_deny_wins_over_allow() -> None:
    executor = ex({"rules": [{"tools": "*", "allow": True}, {"tools": "refund", "allow": False}]})
    with pytest.raises(PolicyDenied):
        await executor.run("refund", REFUND)


async def test_rules_by_environment() -> None:
    data = {
        "environments": ["dev", "prod"],
        "default": {"dev": "allow", "prod": "deny"},
        "rules": [{"tools": "lookup_*", "allow": True, "envs": "prod"}],
    }
    dev, prod = ex(data, env="dev"), ex(data, env="prod")
    assert await dev.run("wipe_orders") == "wiped"
    with pytest.raises(PolicyDenied):
        await prod.run("wipe_orders")
    assert (await prod.run("lookup_order", {"order_id": "o"}))["status"] == "shipped"


async def test_rules_by_team_and_user() -> None:
    data = {
        "default": "deny",
        "rules": [
            {"tools": "refund", "allow": True, "teams": "support"},
            {"tools": "wipe_orders", "allow": True, "users": ["ann@example.com"]},
        ],
    }
    executor = ex(data)
    assert await executor.run("refund", REFUND, caller(team="support")) == "refunded"
    for ctx in (caller(team="sales"), caller()):
        with pytest.raises(PolicyDenied):
            await executor.run("refund", REFUND, ctx)
    assert await executor.run("wipe_orders", None, caller(user="ann@example.com")) == "wiped"
    with pytest.raises(PolicyDenied):
        await executor.run("wipe_orders", None, caller(user="bob@example.com"))


async def test_team_from_principal_claims_and_custom_team_of() -> None:
    from types import SimpleNamespace

    from agent_adapter.core.context import ToolContext

    data = {"default": "deny", "rules": [{"tools": "refund", "allow": True, "teams": "support"}]}
    executor = ex(data)
    principal = SimpleNamespace(id="p1", claims={"team": "support"})
    assert await executor.run("refund", REFUND, ToolContext(extras={"principal": principal})) == (
        "refunded"
    )
    mapping = {"id": "p1", "claims": {"team": "support"}}
    assert await executor.run("refund", REFUND, ToolContext(extras={"principal": mapping})) == (
        "refunded"
    )
    custom = load_policy(data, team_of=lambda ctx: "support").apply(Executor(shop()))
    assert await custom.run("refund", REFUND) == "refunded"


async def test_kinds_and_tags_select_tools() -> None:
    executor = ex({"rules": [{"tags": "money", "allow": False}]})
    with pytest.raises(PolicyDenied):
        await executor.run("refund", REFUND)
    assert await executor.run("wipe_orders") == "wiped"
    by_kind = ex({"rules": [{"kinds": "tool", "tools": "wipe_*", "allow": False}]})
    with pytest.raises(PolicyDenied):
        await by_kind.run("wipe_orders")


async def test_conditional_confirm_asks_only_over_the_limit() -> None:
    port = Approvals()
    executor = ex({"rules": [{"tools": "refund", "confirm": "amount > 100"}]}, approval=port)
    assert await executor.run("refund", REFUND) == "refunded"
    assert port.asked == []
    assert await executor.run("refund", BIG_REFUND) == "refunded"
    assert [r.tool for r in port.asked] == ["refund"]


async def test_conditional_confirm_without_a_port_is_denied() -> None:
    executor = ex({"rules": [{"tools": "refund", "confirm": "amount > 100"}]})
    with pytest.raises(ApprovalDenied, match="no approval port"):
        await executor.run("refund", BIG_REFUND)


async def test_a_refusal_from_the_reviewer_stops_the_call() -> None:
    log: list[str] = []
    policy = load_policy({"rules": [{"tools": "refund", "confirm": True}]})
    executor = policy.apply(Executor(shop(log), approval=Approvals(answer=False)))
    with pytest.raises(ApprovalDenied):
        await executor.run("refund", REFUND)
    assert log == []


async def test_undecidable_condition_asks_for_approval(caplog: pytest.LogCaptureFixture) -> None:
    port = Approvals()
    executor = ex({"rules": [{"tools": "send_note", "confirm": "length > 3"}]}, approval=port)
    with caplog.at_level(logging.WARNING):
        assert await executor.run("send_note", {"text": "hello secret"}) == "hello secret"
    assert len(port.asked) == 1
    assert "undecidable" in caplog.text
    assert "hello secret" not in caplog.text


async def test_confirm_scoped_to_a_team() -> None:
    port = Approvals()
    data = {"rules": [{"tools": "refund", "confirm": True, "teams": "trainees"}]}
    executor = ex(data, approval=port)
    await executor.run("refund", REFUND, caller(team="support"))
    assert port.asked == []
    await executor.run("refund", REFUND, caller(team="trainees"))
    assert len(port.asked) == 1


async def test_code_confirm_true_is_never_removed() -> None:
    policy = load_policy({"rules": [{"tools": "*", "allow": True, "confirm": "1 > 2"}]})
    executor = policy.apply(Executor(shop()))
    spec = executor.registry.get("close_shop")
    assert policy.effective(spec) is spec
    with pytest.raises(ApprovalDenied, match="no approval port"):
        await executor.run("close_shop")


async def test_code_confirm_rule_is_kept_beside_the_policy() -> None:
    from agent_adapter.core.registry import Registry

    reg = Registry()

    @reg.tool(confirm=lambda call, ctx: call.arguments["n"] == 7)
    async def pick(n: int) -> int:
        """Pick a number."""
        return n

    port = Approvals()
    policy = load_policy({"rules": [{"tools": "pick", "confirm": "n > 100"}]})
    executor = policy.apply(Executor(reg, approval=port))
    assert await executor.run("pick", {"n": 1}) == 1
    assert await executor.run("pick", {"n": 7}) == 7
    assert await executor.run("pick", {"n": 101}) == 101
    assert len(port.asked) == 2
    effective = policy.effective(reg.get("pick"))
    assert isinstance(effective.confirm, PolicyConfirm)
    assert "rules[0]" in repr(effective.confirm)
    assert policy.effective(reg.get("pick")) is effective  # cached per spec


async def test_run_spec_of_an_unregistered_tool_is_covered() -> None:
    from agent_adapter.core.tool import build_spec

    async def refund(order_id: str, amount: float) -> str:
        """Refund elsewhere."""
        return "x"

    executor = ex({"rules": [{"tools": "refund", "allow": False}]})
    with pytest.raises(PolicyDenied):
        await executor.run_spec(build_spec(refund), REFUND)


async def test_logging_hook_withholds_arguments_of_policy_confirmed_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = ex(
        {"rules": [{"tools": "refund", "confirm": "amount > 100"}]},
        approval=Approvals(),
        hooks=[LoggingHook(logger=logging.getLogger("tests.policy.shop"))],
    )
    with caplog.at_level(logging.INFO, logger="tests.policy.shop"):
        await executor.run("refund", {"order_id": "o-SECRET", "amount": 5.0})
    assert "o-SECRET" not in caplog.text


async def test_memory_save_rules(tmp_path: Path) -> None:
    from agent_adapter.core.registry import Registry
    from agent_adapter.memory import SQLiteMemory, memory_tools

    reg = Registry()
    reg.add_many(memory_tools(SQLiteMemory(tmp_path / "memory.db")))
    port = Approvals()
    asks = load_policy({"memory": {"save": "confirm"}}).apply(Executor(reg, approval=port))
    ctx = caller(user="ann")
    assert (await asks.run("remember", {"text": "likes tea"}, ctx)).startswith("Saved")
    assert [r.tool for r in port.asked] == ["remember"]
    denies = load_policy({"memory": {"save": "deny"}}).apply(Executor(reg))
    with pytest.raises(PolicyDenied, match=r"memory.save"):
        await denies.run("remember", {"text": "x"}, ctx)
    assert await denies.run("recall", {"query": ""}, ctx) is not None


async def test_max_calls_only_lowers_unless_relaxed() -> None:
    lower = load_policy({"limits": {"max_calls": 1}})
    executor = lower.apply(Executor(shop(), limits=RunLimits(max_calls=5)))
    assert executor.limits.max_calls == 1
    ctx = caller()
    await executor.run("lookup_order", {"order_id": "a"}, ctx)
    with pytest.raises(RunLimitExceeded):
        await executor.run("lookup_order", {"order_id": "b"}, ctx)
    higher = load_policy({"limits": {"max_calls": 9}})
    assert higher.apply(Executor(shop(), limits=RunLimits(max_calls=5))).limits.max_calls == 5
    assert higher.apply(Executor(shop())).limits.max_calls == 9
    relaxed = load_policy(
        {"limits": {"max_calls": 9}, "relax": ["limits.max_calls"]}, allow_relax=True
    )
    assert relaxed.apply(Executor(shop(), limits=RunLimits(max_calls=5))).limits.max_calls == 9
    same = Executor(shop(), limits=RunLimits(max_calls=9))
    assert higher.limits(same.limits) is same.limits


async def test_two_policies_both_apply_and_reapplying_is_a_no_op() -> None:
    first = load_policy({"rules": [{"tools": "wipe_orders", "allow": False}]})
    second = load_policy({"rules": [{"tools": "refund", "confirm": True}]})
    base = Executor(shop())
    once = first.apply(base)
    assert first.apply(once) is once
    both = second.apply(once)
    assert second.apply(both) is both
    assert first.apply(both) is both
    with pytest.raises(PolicyDenied):
        await both.run("wipe_orders")
    with pytest.raises(ApprovalDenied, match="no approval port"):
        await both.run("refund", REFUND)
    assert base.policy is None
    assert len(both.policy.describe(both.registry.get("refund"))) >= 2  # type: ignore[union-attr]


async def test_replace_keeps_the_policy() -> None:
    executor = ex({"rules": [{"tools": "wipe_orders", "allow": False}]})
    with pytest.raises(PolicyDenied):
        await executor.replace(namespace="other").run("wipe_orders")


def test_executor_refuses_a_non_policy() -> None:
    with pytest.raises(ToolUsageError, match="ToolPolicy"):
        Executor(shop(), policy=object())  # type: ignore[arg-type]


def test_tools_hides_tools_denied_to_everyone() -> None:
    policy = load_policy(
        {
            "default": "deny",
            "rules": [
                {"tools": "lookup_*", "allow": True},
                {"tools": "refund", "allow": True, "teams": "support"},
            ],
        }
    )
    assert policy.tools(shop()).names() == ["refund", "lookup_order"]
    assert len(load_policy({"default": "deny"}).tools(shop())) == 0
