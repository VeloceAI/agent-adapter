"""Review scenarios: node caches, code guards, bridges, MCP client approval, stacking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.cache import MemoryCache
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, FlowError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.exec import Executor, RunLimits
from agent_adapter.flow import Call, Flow, FlowRunner, NodeCache
from agent_adapter.hooks import PIIGuard
from agent_adapter.policy import PolicyError, load_policy
from tests.policy.helpers import Approvals, caller, shop


class Look(BaseModel):
    order_id: str = ""
    result: Any = None


def look_flow() -> Flow[Look]:
    call = Call("lookup_order", args=lambda s: {"order_id": s.order_id}, into="result")
    return Flow("look", state=Look).node("look", call, cache=NodeCache(scope="global")).end("look")


async def test_node_cache_never_replays_a_result_to_a_denied_caller() -> None:
    log: list[str] = []
    data = {
        "default": "deny",
        "rules": [{"tools": "lookup_order", "allow": True, "teams": "support"}],
    }
    executor = load_policy(data).apply(Executor(shop(log), cache=MemoryCache()))
    runner = FlowRunner(executor)
    first = await runner.run(look_flow(), {"order_id": "o1"}, caller(team="support", run_id="r1"))
    assert first.state.result["status"] == "shipped"
    with pytest.raises(FlowError, match="policy: default deny") as caught:  # no cached data
        await runner.run(look_flow(), {"order_id": "o1"}, caller(team="guest", run_id="r2"))
    assert "shipped" not in str(caught.value)
    assert log == ["lookup o1"]


async def test_node_cache_never_skips_a_policy_approval() -> None:
    log: list[str] = []
    port = Approvals()
    data = {"rules": [{"tools": "lookup_order", "confirm": True, "teams": "trainees"}]}
    executor = load_policy(data).apply(Executor(shop(log), cache=MemoryCache(), approval=port))
    runner = FlowRunner(executor)
    await runner.run(look_flow(), {"order_id": "o1"}, caller(team="support", run_id="r1"))
    await runner.run(look_flow(), {"order_id": "o1"}, caller(team="trainees", run_id="r2"))
    assert log == ["lookup o1", "lookup o1"]
    assert len(port.asked) == 1


async def test_a_code_approve_guard_still_asks_under_a_policy_confirm() -> None:
    guard = PIIGuard(kinds=["email"], on="args", action="approve")
    reg = Registry()
    sent: list[str] = []

    @reg.tool(confirm=guard.confirm)
    async def send_note(text: str) -> str:
        """Send a note."""
        sent.append(text)
        return "sent"

    port = Approvals()
    policy = load_policy({"rules": [{"tools": "send_note", "confirm": "text == 'urgent'"}]})
    executor = policy.apply(Executor(reg, hooks=[guard], approval=port))
    assert await executor.run("send_note", {"text": "mail ann@example.com"}) == "sent"
    assert await executor.run("send_note", {"text": "urgent"}) == "sent"
    assert await executor.run("send_note", {"text": "hello"}) == "sent"
    assert len(port.asked) == 2 and len(sent) == 3
    effective = policy.effective(reg.get("send_note"))
    assert effective.confirm == guard.confirm
    assert effective.confirm != (lambda call, ctx: True)
    assert hash(effective.confirm) == hash(effective.confirm)


async def test_undecidable_values_ask_through_the_executor() -> None:
    port = Approvals()
    data = {"rules": [{"tools": "refund", "confirm": "amount == 'lots'"}]}
    executor = load_policy(data).apply(Executor(shop(), approval=port))
    assert await executor.run("refund", {"order_id": "o", "amount": 1e308}) == "refunded"
    assert len(port.asked) == 1


async def test_native_bridge_path_with_a_policy() -> None:
    """What every native bridge does: its own gate on the spec, then `run_bound`."""
    from agent_adapter.integrations._native_gate import gate_decision, run_bound

    log: list[str] = []
    reg = shop(log)
    port = Approvals()
    data = {
        "rules": [
            {"tools": "wipe_orders", "allow": False},
            {"tools": "refund", "confirm": "amount > 100"},
        ]
    }
    executor = load_policy(data).apply(Executor(reg, approval=port))
    refund = reg.get("refund")
    big = {"order_id": "o", "amount": 500.0}
    native = await gate_decision(
        refund, big, port=port, integration="demo", call_id="c1", run_id="r1"
    )
    assert native is None  # the bridge sees the code's spec; the executor asks instead
    ctx = ToolContext(run_id="r1", call_id="c1")
    assert await run_bound(executor, refund, big, ctx) == "refunded"
    assert len(port.asked) == 1
    token = ApprovalDecision(approved=True, decided_by="framework")
    wiped = await run_bound(
        executor, reg.get("wipe_orders"), {}, ToolContext(run_id="r1", call_id="c2", approval=token)
    )
    assert "error" in wiped  # a framework approval never overrides a policy denial
    assert log == ["refund o 500.0"]


async def test_langchain_bridge_obeys_the_policy() -> None:
    pytest.importorskip("langchain_core")
    from agent_adapter.integrations.langchain import structured_tool

    log: list[str] = []
    reg = shop(log)
    port = Approvals()
    data = {
        "rules": [{"tools": "wipe_orders", "allow": False}, {"tools": "refund", "confirm": True}]
    }
    executor = load_policy(data).apply(Executor(reg, approval=port))
    wipe = structured_tool(reg.get("wipe_orders"), executor=executor)
    refund = structured_tool(reg.get("refund"), executor=executor)
    denied = await wipe.ainvoke({})
    assert "denied" in str(denied)
    assert await refund.ainvoke({"order_id": "o", "amount": 5.0}) == "refunded"
    assert log == ["refund o 5.0"] and len(port.asked) == 1


async def test_mcp_client_approval_and_a_policy_only_confirm() -> None:
    pytest.importorskip("mcp")
    from mcp import Client

    from agent_adapter.integrations.mcp import ERROR_KIND_META, mcp_server

    log: list[str] = []
    data = {
        "rules": [
            {"tools": "refund", "confirm": "amount > 100"},
            {"tools": "wipe_orders", "allow": False},
        ]
    }
    store = MemoryApprovalStore()
    executor = load_policy(data).apply(Executor(shop(log), approval=DeferredApproval(store)))
    server = mcp_server(shop(log), executor=executor, client_approval=True)
    async with Client(server) as client:
        big = await client.call_tool("refund", {"order_id": "o", "amount": 500.0})
        wipe = await client.call_tool("wipe_orders", {})
    assert big.meta is not None and big.meta[ERROR_KIND_META] == "approval-pending"
    assert wipe.meta is not None and wipe.meta[ERROR_KIND_META] == "approval-denied"
    assert log == []
    plain = load_policy(data).apply(Executor(shop(log)))
    async with Client(mcp_server(shop(log), executor=plain, client_approval=True)) as client:
        refused = await client.call_tool("refund", {"order_id": "o", "amount": 500.0})
    assert refused.meta is not None and refused.meta[ERROR_KIND_META] == "approval-denied"
    assert log == []


async def test_a_stacked_relaxation_never_loosens_an_earlier_policy() -> None:
    strict = load_policy({"limits": {"max_calls": 2}})
    relaxed = load_policy(
        {"limits": {"max_calls": 50}, "relax": ["limits.max_calls"]}, allow_relax=True
    )
    base = Executor(shop(), limits=RunLimits(max_calls=10))
    assert relaxed.apply(strict.apply(base)).limits.max_calls == 2
    assert strict.apply(relaxed.apply(base)).limits.max_calls == 2
    assert relaxed.apply(base).limits.max_calls == 50
    unlimited = load_policy({}).apply(relaxed.apply(Executor(shop())))
    assert unlimited.limits.max_calls == 50
    neither = load_policy({}).apply(load_policy({"default": "allow"}).apply(Executor(shop())))
    assert neither.limits.max_calls is None


def test_conditions_are_shown_without_their_strings(tmp_path: Path) -> None:
    import io

    from agent_adapter.cli.policy import run_policy

    data = {"rules": [{"tools": "refund", "confirm": "email == 'ceo@shop.example'"}]}
    assert "ceo@shop.example" not in str(load_policy(data).explain("refund"))
    path = tmp_path / "p.json"
    path.write_text(
        '{"rules": [{"tools": "refund", "confirm": "email == \'ceo@shop.example\'"}]}',
        encoding="utf-8",
    )
    import argparse

    out = io.StringIO()
    args = argparse.Namespace(file=path, env=None, allow_relax=False, tool=["refund"])
    assert run_policy(args, out) == 0
    assert "ceo@shop.example" not in out.getvalue()
    assert "email == '...'" in out.getvalue()


@pytest.mark.parametrize("confirm", [1, 1.5, ["x"], {"a": 1}])
def test_confirm_must_be_true_or_a_condition(confirm: Any) -> None:
    with pytest.raises(PolicyError) as caught:
        load_policy({"rules": [{"tools": "refund", "confirm": confirm}]})
    assert caught.value.key == "rules[0].confirm"


@pytest.mark.parametrize(
    "text",
    [
        "a0: &a0 [x, x]\na1: [*a0, *a0]\n",
        "rules: &a [{tools: x, allow: true, teams: *a}]\n",
        "default: &d allow\n",
    ],
)
def test_yaml_anchors_and_aliases_are_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match="anchors and aliases") as caught:
        load_policy(path)
    assert caught.value.line >= 1


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("p.yaml", "rules: " + "[" * 2000 + "]" * 2000 + "\n"),
        ("p.json", '{"rules": ' + "[" * 100_000 + "]" * 100_000 + "}"),
        ("p.json", '{"rules": ' + "[" * 40 + "]" * 40 + "}"),
    ],
)
def test_deep_nesting_is_refused(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match="nests"):
        load_policy(path)


async def test_run_call_after_replace_keeps_the_flow_policy() -> None:
    executor = load_policy({"rules": [{"tools": "wipe_orders", "allow": False}]}).apply(
        Executor(shop())
    )
    result = await executor.replace(hooks=[]).run_call(ToolCall(name="wipe_orders", arguments={}))
    assert result.error is not None and result.error.kind == "approval-denied"
    with pytest.raises(ApprovalDenied):
        await executor.run("wipe_orders")
