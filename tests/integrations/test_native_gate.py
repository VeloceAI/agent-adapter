import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalPending, IntegrationError, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations._native_gate import (
    TokenTable,
    arguments_digest,
    check_framework,
    decide_calls,
    dump_arguments,
    gate_decision,
    json_text,
    pending_message,
    run_bound,
    take_approval,
)
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.fakes.clock import FakeClock
from tests.integrations.kit import FakeRunner


async def remove(item: str, count: int = 1) -> str:
    """Remove an item."""
    return f"removed {item}"


def spec(confirm=True):
    return build_spec(remove, confirm=confirm)


def approve(reason="ok"):
    return ApprovalDecision(approved=True, reason=reason)


async def gate(spec_, args, port, *, call_id="c1", run_id="r1", edits=False):
    return await gate_decision(
        spec_, args, port=port, integration="demo", call_id=call_id, run_id=run_id, edits=edits
    )


async def test_gate_decision_asks_the_port_with_validated_arguments():
    port = ScriptedApproval([approve()])
    assert await gate(spec(), {"item": "a"}, port) == approve()
    (request,) = port.requests
    assert (request.tool, request.call_id, request.run_id) == ("remove", "c1", "r1")
    assert request.arguments == {"item": "a", "count": 1}
    assert request.approval_id.startswith("apr_") and request.created_at > 0


async def test_gate_decision_returns_a_denial_unchanged():
    denied = ApprovalDecision(approved=False, reason="no")
    assert await gate(spec(), {"item": "a"}, ScriptedApproval([denied])) == denied


async def test_gate_decision_skips_the_port_when_the_rule_says_no_or_args_are_invalid():
    port = ScriptedApproval([])
    rule = lambda call, ctx: call.arguments["item"] == "danger"  # noqa: E731
    assert await gate(spec(rule), {"item": "a"}, port) is None
    assert await gate(spec(False), {"item": "a"}, port) is None
    assert await gate(spec(), {"count": "x"}, port) is None
    assert port.requests == []


async def test_a_call_without_an_id_is_denied_without_asking():
    port = ScriptedApproval([])
    decision = await gate(spec(), {"item": "a"}, port, call_id="")
    assert decision is not None and not decision.approved and "no id" in decision.reason
    assert port.requests == []


async def test_gate_decision_turns_unsupported_edits_into_a_denial(caplog):
    port = ScriptedApproval([ApprovalDecision(approved=True, edited_args={"item": "b"})])
    with caplog.at_level(logging.WARNING):
        decision = await gate(spec(), {"item": "a"}, port)
    assert decision is not None and not decision.approved and decision.edited_args is None
    assert decision.reason == "argument edits are not supported by demo"
    assert "edits its arguments" in caplog.text and "'a'" not in caplog.text
    edited = ApprovalDecision(approved=True, edited_args={"item": "b"})
    assert await gate(spec(), {"item": "a"}, ScriptedApproval([edited]), edits=True) == edited


async def test_gate_decision_lets_pending_and_rule_errors_propagate():
    with pytest.raises(ApprovalPending):
        await gate(spec(), {"item": "a"}, ScriptedApproval([ApprovalPending("apr_1")]))
    with pytest.raises(ToolUsageError):
        await gate(spec(lambda c, x: "yes"), {"item": "a"}, ScriptedApproval([]))


def registry_of(*specs):
    registry = Registry()
    registry.add_many(specs)
    return registry


async def decide(port, calls, *, run_id="r1", tokens=None):
    tokens = tokens if tokens is not None else TokenTable()
    registry = registry_of(spec(), build_spec(remove, name="maybe", confirm=lambda c, x: False))
    return await decide_calls(
        port, calls, registry, run_id=run_id, integration="demo", tokens=tokens
    ), tokens


async def test_decide_calls_plans_asks_and_issues_tokens():
    port = RecordingApproval(ScriptedApproval([approve(), approve()]))
    decisions, tokens = await decide(
        port,
        [
            ("remove", {"item": "a"}, "c1"),
            ("maybe", {"item": "b"}, "c2"),
            ("remove", {"wrong": 1}, "c3"),
            ("remove", {"item": "d"}, ""),
            ("other_tool", {"x": 1}, "c5"),
        ],
    )
    assert [d.approved for d in decisions] == [True, True, False, False, True]
    assert [(r.tool, r.arguments) for r in port.requests] == [
        ("remove", {"item": "a", "count": 1}),
        ("other_tool", {"x": 1}),
    ]
    assert len(tokens) == 1  # only the approved registry call that needed approval
    assert tokens.take("remove", {"item": "a", "count": 1}, call_id="c1", run_id="r1")


async def test_decide_calls_refuses_edits_and_needs_a_run_id_for_tokens(caplog):
    edited = ApprovalDecision(approved=True, edited_args={"item": "z"})
    (decision,), tokens = await decide(ScriptedApproval([edited]), [("remove", {"item": "a"}, "c")])
    assert not decision.approved and "edits" in decision.reason and len(tokens) == 0
    with caplog.at_level(logging.WARNING):
        (decision,), tokens = await decide(
            ScriptedApproval([approve()]), [("remove", {"item": "a"}, "c")], run_id=""
        )
    assert decision.approved and len(tokens) == 0 and "without a run id" in caplog.text


async def test_decide_calls_raises_the_first_pending_after_asking_all():
    port = RecordingApproval(ScriptedApproval([ApprovalPending("apr_1"), approve()]))
    with pytest.raises(ApprovalPending):
        await decide(port, [("remove", {"item": "a"}, "c1"), ("remove", {"item": "b"}, "c2")])
    assert len(port.requests) == 2


async def test_take_approval_and_run_bound():
    tokens = TokenTable()
    tokens.issue("remove", {"item": "a", "count": 1}, approve(), call_id="c1", run_id="r1")
    ctx = ToolContext(run_id="r1", call_id="c1")
    assert (
        take_approval(tokens, spec(), {"item": "a"}, ToolContext(run_id="r2", call_id="c1")) is None
    )
    assert take_approval(tokens, spec(), {"item": "a"}, ToolContext(run_id="r1")) is None
    assert take_approval(tokens, spec(), {"wrong": 1}, ctx) is None
    assert take_approval(tokens, spec(), {"item": "a"}, ctx) == approve()
    assert take_approval(tokens, spec(), {"item": "a"}, ctx) is None
    runner = FakeRunner()
    assert await run_bound(runner, spec(False), {"item": "a"}, ToolContext()) == "removed a"
    error = await run_bound(runner, spec(False), {}, ToolContext())
    assert error["error"]["kind"] == "tool-args-invalid"


def test_helpers():
    assert pending_message(ApprovalPending("apr_9")) == "approval required: apr_9"
    assert dump_arguments(spec(), {"item": "a"}) == {"item": "a", "count": 1}
    assert dump_arguments(spec(), {}) is None
    assert arguments_digest("t", {"a": 1}) == arguments_digest("t", {"a": 1})
    assert arguments_digest("t", {"a": 1}) != arguments_digest("u", {"a": 1})
    assert arguments_digest("t", {"a": (1, 2)}) is None
    assert json_text("plain") == "plain" and json_text({"a": [1, "é"]}) == '{"a": [1, "é"]}'
    marker = object()
    assert json_text(marker) is marker


def test_check_framework_range_and_attributes():
    check_framework("fw", "1.23.0", ((1, 23), (2, 0)), {"x": True})
    for version in ("1.22.9", "2.0.0", "0.0.0", "dev"):
        with pytest.raises(IntegrationError, match="outside the supported range"):
            check_framework("fw", version, ((1, 23), (2, 0)), {})
    with pytest.raises(IntegrationError, match="lacks a, b") as info:
        check_framework("fw", "1.30", ((1, 23), (2, 0)), {"b": False, "a": False, "c": True})
    assert info.value.integration == "fw"


def test_a_token_is_taken_once_and_bound_to_tool_args_and_call_id():
    tokens = TokenTable()
    token = tokens.issue("remove", {"item": "a"}, approve(), call_id="c1", run_id="r1")
    assert token is not None and token.run_id == "r1"
    assert tokens.take("remove", {"item": "b"}, call_id="c1") is None
    assert tokens.take("other", {"item": "a"}, call_id="c1") is None
    assert tokens.take("remove", {"item": "a"}, call_id="c2") is None
    assert tokens.take("remove", {"item": "a"}, call_id="") is None
    assert tokens.take("remove", {"item": "a"}, call_id="c1") == token
    assert tokens.take("remove", {"item": "a"}, call_id="c1") is None


def test_take_can_require_the_run():
    tokens = TokenTable()
    tokens.issue("remove", {"item": "a"}, approve(), call_id="c1", run_id="r1")
    assert tokens.take("remove", {"item": "a"}, call_id="c1", run_id="r2") is None
    assert tokens.take("remove", {"item": "a"}, call_id="c1", run_id="r1") is not None


def test_take_with_any_call_id_returns_the_oldest():
    tokens = TokenTable()
    tokens.issue("remove", {"item": "a"}, approve("first"), call_id="c1", run_id="r")
    tokens.issue("remove", {"item": "a"}, approve("second"), call_id="c2", run_id="r")
    first = tokens.take("remove", {"item": "a"}, call_id=None)
    assert first is not None and first.call_id == "c1"


def test_tokens_need_a_call_id_and_json_and_expire(caplog):
    clock = FakeClock()
    tokens = TokenTable(ttl=10, clock=clock)
    with caplog.at_level(logging.WARNING):
        assert tokens.issue("remove", {"item": "a"}, approve(), call_id="", run_id="r") is None
        assert tokens.issue("remove", {"x": (1,)}, approve(), call_id="c", run_id="r") is None
    assert "cannot bind it" in caplog.text
    assert tokens.take("remove", {"x": (1,)}, call_id="c") is None
    tokens.issue("remove", {"item": "a"}, approve(), call_id="c1", run_id="r")
    assert len(tokens) == 1
    clock.advance(10)
    assert len(tokens) == 0 and tokens.take("remove", {"item": "a"}, call_id="c1") is None


def test_full_table_drops_the_oldest_and_clear_empties_it(caplog):
    tokens = TokenTable(max_tokens=2)
    for i in range(3):
        tokens.issue("remove", {"item": str(i)}, approve(), call_id=f"c{i}", run_id="r")
    assert len(tokens) == 2 and "dropped remove (call c0)" in caplog.text
    tokens.clear()
    assert len(tokens) == 0
    tokens.clear()
