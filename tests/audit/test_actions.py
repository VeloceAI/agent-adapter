"""Which events become audit entries, and what each entry keeps."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.audit import _actions
from agent_adapter.audit._actions import Mapper
from agent_adapter.core.types import Event


def digest(value: object) -> str:
    return "digest"


def entries(
    mapper: Mapper, type: str, call: str = "call_1", **data: Any
) -> list[tuple[str, dict[str, Any]]]:
    found = mapper.entries(Event(type, run_id="run_1", call_id=call, ts=5.0, data=data))
    return [(entry.action, dict(entry.data)) for entry in found]


@pytest.fixture
def mapper() -> Mapper:
    return Mapper(digest)


def test_ordinary_events_are_ignored(mapper: Mapper) -> None:
    assert entries(mapper, "tool.started", tool="refund", kind="tool") == []
    assert entries(mapper, "llm.finished", provider="p", model="m") == []
    assert entries(mapper, "memory.searched", backend="sqlite", count=1) == []
    assert entries(mapper, "custom.something", a=1) == []
    assert entries(mapper, "custom.audit.", a=1) == []


def test_approval_flow_records_request_decision_and_use(mapper: Mapper) -> None:
    assert entries(mapper, "approval.requested", tool="refund", approval_id="apr_1") == [
        ("approval.requested", {"tool": "refund", "approval_id": "apr_1"})
    ]
    assert entries(
        mapper, "approval.decided", tool="refund", approval_id="apr_1", approved=True
    ) == [("approval.decided", {"tool": "refund", "approval_id": "apr_1", "approved": True})]
    assert entries(mapper, "tool.finished", tool="refund", latency_ms=1) == [
        ("approval.consumed", {"tool": "refund", "approval_id": "apr_1", "ok": True})
    ]
    assert entries(mapper, "tool.finished", tool="refund") == []  # used once


def test_a_failed_approved_call_is_still_a_consumed_approval(mapper: Mapper) -> None:
    entries(mapper, "approval.decided", tool="t", approval_id="apr_2", approved=True)
    found = entries(mapper, "tool.failed", tool="t", error_kind="tool-failed")
    assert found == [("approval.consumed", {"tool": "t", "approval_id": "apr_2", "ok": False})]


def test_a_denial_is_recorded_once(mapper: Mapper) -> None:
    found = entries(mapper, "approval.decided", tool="t", approval_id="apr_3", approved=False)
    assert [action for action, _ in found] == ["approval.decided", "approval.denied"]
    assert entries(mapper, "tool.failed", tool="t", error_kind="approval-denied") == []


def test_a_denial_without_a_decision_is_recorded(mapper: Mapper) -> None:
    assert entries(mapper, "tool.failed", tool="t", error_kind="approval-denied") == [
        ("approval.denied", {"tool": "t", "error_kind": "approval-denied"})
    ]


@pytest.mark.parametrize("outcome", ["timed-out", "stopped"])
def test_an_expired_approval_is_recorded(mapper: Mapper, outcome: str) -> None:
    found = entries(mapper, "approval.expired", tool="t", approval_id="a", outcome=outcome)
    assert found == [("approval.expired", {"tool": "t", "approval_id": "a", "outcome": outcome})]


def test_a_redaction_keeps_counts_and_kinds_only(mapper: Mapper) -> None:
    found = entries(
        mapper,
        "guardrail.redacted",
        guard="pii_guard",
        stage="result",
        count=2,
        kinds=["card", "email"],
        tool="lookup",
        result="4111 1111 1111 1111",
    )
    assert found == [
        (
            "guardrail.redacted",
            {
                "guard": "pii_guard",
                "stage": "result",
                "count": 2,
                "kinds": ["card", "email"],
                "tool": "lookup",
            },
        )
    ]


def test_edited_arguments_keep_only_a_digest(mapper: Mapper) -> None:
    edited = {"amount": 5, "card": "4111 1111 1111 1111"}
    ((_, data),) = entries(
        mapper, "approval.decided", tool="t", approval_id="a", approved=True, edited_args=edited
    )
    assert data["edited"] is True and data["edited_digest"] == "digest"
    assert "4111" not in repr(data)
    ((_, flag),) = entries(
        mapper, "approval.decided", tool="t", approval_id="b", approved=True, edited=False
    )
    assert flag["edited"] is False


@pytest.mark.parametrize(
    ("kind", "action"),
    [("guardrail-blocked", "guardrail.blocked"), ("run-limit-exceeded", "limits.exceeded")],
)
def test_blocked_tool_calls_are_recorded(mapper: Mapper, kind: str, action: str) -> None:
    assert entries(mapper, "tool.failed", tool="t", error_kind=kind) == [
        (action, {"tool": "t", "error_kind": kind})
    ]
    assert entries(mapper, "tool.failed", tool="t", error_kind="tool-timeout") == []


def test_a_blocked_model_call_is_recorded(mapper: Mapper) -> None:
    found = entries(
        mapper, "llm.failed", provider="p", model="m", stage="call", error_kind="guardrail-blocked"
    )
    assert found == [
        (
            "guardrail.blocked",
            {"provider": "p", "model": "m", "stage": "call", "error_kind": "guardrail-blocked"},
        )
    ]
    assert entries(mapper, "llm.failed", provider="p", model="m", error_kind="model-timeout") == []


def test_budget_memory_sandbox_and_recovery_keep_their_listed_keys(mapper: Mapper) -> None:
    assert entries(
        mapper, "budget.exceeded", scope="user", limit_usd=1.0, spent_usd=1.2, window="day"
    ) == [
        ("budget.exceeded", {"scope": "user", "window": "day", "limit_usd": 1.0, "spent_usd": 1.2})
    ]
    found = entries(
        mapper, "memory.added", backend="sqlite", count=2, latency_ms=1, arguments={"text": "x"}
    )
    assert found == [("memory.added", {"backend": "sqlite", "count": 2})]
    found = entries(mapper, "memory.exported", backend="sqlite", count=1, scope_kind="user")
    assert found == [("memory.exported", {"backend": "sqlite", "count": 1, "scope_kind": "user"})]
    found = entries(mapper, "memory.deleted", backend="sqlite", count=4, scope="ann")
    assert found == [
        ("memory.deleted", {"backend": "sqlite", "count": 4, "scope_digest": "digest"})
    ]
    found = entries(
        mapper,
        "sandbox.finished",
        sandbox="code",
        backend="docker",
        language="python",
        exit_code=0,
        result={"stdout": "x"},
    )
    assert found == [
        (
            "sandbox.finished",
            {"sandbox": "code", "backend": "docker", "language": "python", "exit_code": 0},
        )
    ]
    found = entries(mapper, "run.recovered", kind="flow", step=3, replayed=2, rerun=0)
    assert found == [
        ("checkpoint.recovered", {"kind": "flow", "step": 3, "replayed": 2, "rerun": 0})
    ]


def test_app_events_are_namespaced_and_cleaned(mapper: Mapper) -> None:
    (entry,) = mapper.entries(
        Event(
            "custom.audit.refund_override",
            data={"rules": 3, "token": "abc", "arguments": {"a": 1}, "note": "ann@example.com"},
        )
    )
    assert (entry.action, entry.source) == ("app.refund_override", "app")
    assert dict(entry.data) == {"rules": 3, "token": "***", "note": "[EMAIL]"}


@pytest.mark.parametrize(
    "type",
    [
        "custom.audit.approval.decided",
        "custom.audit.guardrail.blocked",
        "custom.audit.budget.exceeded",
    ],
)
def test_app_events_can_never_mimic_first_party_actions(mapper: Mapper, type: str) -> None:
    (entry,) = mapper.entries(Event(type, data={"tool": "refund", "approved": True}))
    assert entry.action.startswith("app.") and entry.source == "app"


@pytest.mark.parametrize(
    "type",
    ["policy.loaded", "custom.policy.loaded", "guardrail.flagged", "custom.guardrail.blocked"],
)
def test_other_custom_or_unknown_events_are_ignored(mapper: Mapper, type: str) -> None:
    assert entries(mapper, type, rules=3) == []


def test_first_party_entries_are_marked_sdk(mapper: Mapper) -> None:
    (entry,) = mapper.entries(Event("approval.requested", data={"tool": "t", "approval_id": "a"}))
    assert entry.source == "sdk"


def test_ids_are_cleaned_too(mapper: Mapper) -> None:
    (entry,) = mapper.entries(Event("custom.audit.x", run_id="ann@example.com", call_id="c_1"))
    assert (entry.run_id, entry.call_id) == ("[EMAIL]", "c_1")


def test_tracked_decisions_are_bounded(mapper: Mapper, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_actions, "_MAX_TRACKED", 2)
    for n in range(3):
        entries(
            mapper, "approval.decided", call=f"c{n}", tool="t", approval_id=f"a{n}", approved=True
        )
    assert entries(mapper, "tool.finished", call="c0", tool="t") == []
    assert entries(mapper, "tool.finished", call="c2", tool="t")[0][0] == "approval.consumed"
