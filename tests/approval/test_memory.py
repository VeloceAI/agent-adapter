"""Approval memory: similar past decisions, secrets left out, and it never decides."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval.memory import (
    ApprovalContext,
    approval_context,
    argument_similarity,
    comparable_fields,
    decided,
)
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest


def request(
    tool: str = "refund", approval_id: str = "apr_new", kind: str = "tool", **arguments: Any
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        run_id="run_1",
        call_id="c1",
        tool=tool,
        kind=kind,
        arguments=arguments,
    )


def record(
    approved: bool | None,
    *,
    approval_id: str,
    by: str = "ann",
    at: float = 100.0,
    edited: dict[str, Any] | None = None,
    tool: str = "refund",
    **arguments: Any,
) -> ApprovalRecord:
    req = request(tool, approval_id, **arguments)
    if approved is None:
        return ApprovalRecord(request=req, status="pending")
    decision = ApprovalDecision(approved=approved, decided_by=by, decided_at=at, edited_args=edited)
    return ApprovalRecord(
        request=req, status="approved" if approved else "denied", decision=decision
    )


def test_comparable_fields_leave_out_secret_named_fields_and_flatten_nesting() -> None:
    fields = comparable_fields(
        {"order": {"id": 7, "api_key": "sk-123456789"}, "password": "x", "note": [1, 2]}
    )
    assert fields == {"orderid": "7", "note": "[1,2]"}


def test_comparable_fields_stop_at_the_depth_limit() -> None:
    fields = comparable_fields({"a": {"b": {"c": {"d": 1}}}})
    assert fields == {"abc": '{"d":1}'}


def test_similarity_counts_equal_shared_and_one_sided_fields() -> None:
    assert argument_similarity({}, {}) == (1.0, ())
    assert argument_similarity({"a": 1, "b": 2}, {"a": 1, "b": 2}) == (1.0, ("a", "b"))
    assert argument_similarity({"a": 1, "b": 2}, {"a": 1, "b": 3}) == (0.75, ("a",))
    assert argument_similarity({"a": 1, "b": 2}, {"a": 2, "b": 3}) == (0.5, ())
    assert argument_similarity({"a": 1}, {"b": 1}) == (0.0, ())


def test_secret_values_never_change_the_similarity() -> None:
    same = argument_similarity({"a": 1, "token": "one"}, {"a": 1, "token": "two"})
    assert same == (1.0, ("a",))


def test_decided_keeps_only_decisions_newest_first() -> None:
    items = [
        record(None, approval_id="p"),
        record(True, approval_id="old", at=1.0),
        record(False, approval_id="new", at=5.0),
    ]
    assert [r.approval_id for r in decided(items)] == ["new", "old"]


def test_context_counts_similar_decisions_of_the_same_tool() -> None:
    items = [
        record(True, approval_id="a1", by="ann", at=1.0, amount=20, reason="late"),
        record(True, approval_id="a2", by="bob", at=2.0, amount=20, reason="broken"),
        record(False, approval_id="a3", by="ann", at=3.0, amount=900, reason="late"),
        record(True, approval_id="o1", tool="ship", amount=20, reason="late"),
        record(True, approval_id="far", other=1),
        record(None, approval_id="pending", amount=20, reason="late"),
    ]
    found = approval_context(request(amount=20, reason="late"), items)
    assert (found.similar, found.approved, found.denied) == (3, 2, 1)
    assert found.deciders == ("ann", "bob")
    assert found.last_decided_at == 3.0
    assert [e.approval_id for e in found.examples] == ["a1", "a3", "a2"]
    assert found.examples[0].matched == ("amount", "reason")
    assert found.summary() == "3 similar: 2 approved, 1 denied (by ann, bob)"


def test_context_skips_the_request_itself_other_kinds_and_shows_at_most_n() -> None:
    items = [record(True, approval_id=f"a{n}", at=float(n), amount=1) for n in range(8)]
    items.append(record(True, approval_id="apr_new", amount=1))
    found = approval_context(request(amount=1), items, shown=2)
    assert found.similar == 8 and len(found.examples) == 2
    assert approval_context(request(kind="node", amount=1), items).similar == 0


def test_edited_approvals_are_marked() -> None:
    items = [record(True, approval_id="e", edited={"amount": 5}, amount=50)]
    assert approval_context(request(amount=50), items).examples[0].edited is True


def test_an_empty_context_says_so() -> None:
    empty = approval_context(request(amount=1), [])
    assert empty == ApprovalContext(tool="refund")
    assert empty.view()["summary"] == "no similar decisions yet"


def test_the_view_never_holds_argument_values() -> None:
    items = [record(True, approval_id="a", amount=20, api_key="sk-abcdefghijk")]
    view = approval_context(request(amount=20, api_key="sk-abcdefghijk"), items).view()
    text = repr(view)
    assert "sk-abcdefghijk" not in text and "api_key" not in text and "20" not in text
    assert view["examples"][0]["matched"] == ["amount"]


@pytest.mark.parametrize(
    "options", [{"min_similarity": 1.5}, {"min_similarity": -0.1}, {"shown": -1}]
)
def test_bad_options_are_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        approval_context(request(), [], **options)


def test_nested_keys_are_never_named_as_they_can_be_data() -> None:
    args: dict[str, Any] = {"recipients": {"a@example.com": True}, "subject": "hi"}
    items = [record(True, approval_id="n", **args)]
    found = approval_context(request(**args), items)
    assert found.examples[0].matched == ("recipients.*", "subject")
    assert "a@example.com" not in repr(found.view())
