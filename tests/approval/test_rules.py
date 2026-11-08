"""Proposed confirm rules: bounded to the evidence, counter-evidence, review dates, fail closed."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval.rules import (
    Condition,
    ProposedRule,
    RuleConfirm,
    accept,
    confirm_from_config,
    propose_rules,
)
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ToolCall
from tests.approval.test_memory import record

CTX = ToolContext(run_id="run_1")


def refunds() -> list[Any]:
    """The review's scenario: five approved EUR refunds of 10..50, by two reviewers."""
    return [
        record(
            True,
            approval_id=f"r{n}",
            by="ann" if n % 2 else "bob",
            at=float(n),
            amount=amount,
            currency="EUR",
            order=f"o-{n}",
        )
        for n, amount in enumerate((10, 20, 30, 40, 50))
    ]


def call(**arguments: Any) -> ToolCall:
    return ToolCall(name="refund", arguments=arguments)


def test_a_rule_bounds_every_argument_to_the_values_seen() -> None:
    (rule,) = propose_rules(refunds())
    assert rule.conditions == (
        Condition(field="amount", op="between", values=(10, 50)),
        Condition(field="currency", op="==", values=("EUR",)),
        Condition(field="order", op="in", values=tuple(f"o-{n}" for n in range(5))),
    )
    assert (rule.evidence.approved, rule.evidence.denied) == (5, 0)
    assert rule.evidence.deciders == ("ann", "bob") and rule.review_days == 90.0
    assert "'currency' == 'EUR'" in rule.describe() and "review in 90 days" in rule.describe()


def test_the_reviews_escapes_all_ask() -> None:
    confirm = accept(propose_rules(refunds())[0], accepted_by="ann").confirm
    assert confirm(call(amount=30, currency="EUR", order="o-1"), CTX) is False
    asks = [
        call(amount=99_999_999, currency="EUR", order="o-1"),  # above the seen range
        call(amount=-1_000_000, currency="EUR", order="o-1"),  # below it
        call(amount=30, currency="USD", order="o-1"),
        call(amount=30, currency="EUR", order="o-9"),  # an order never seen
        call(amount=30, currency="EUR"),  # a covered field missing
        call(amount=30, currency="EUR", order="o-1", note="x"),  # a field never seen
        call(amount=True, currency="EUR", order="o-1"),
        ToolCall(name="ship", arguments={"amount": 30, "currency": "EUR", "order": "o-1"}),
    ]
    assert [confirm(item, CTX) for item in asks] == [True] * len(asks)


def test_secret_fields_neither_bound_nor_block_a_rule() -> None:
    items = [
        record(True, approval_id=f"s{n}", by=f"p{n}", region="eu", api_key=f"sk-{n}" * 4)
        for n in range(5)
    ]
    (rule,) = propose_rules(items)
    assert [c.field for c in rule.conditions] == ["region"]
    confirm = accept(rule, accepted_by="ann").confirm
    assert confirm(call(region="eu", api_key="sk-other-key-123"), CTX) is False


def test_one_reviewer_or_too_few_approvals_propose_nothing() -> None:
    single = [record(True, approval_id=f"a{n}", by="ann", region="eu") for n in range(9)]
    assert propose_rules(single) == []
    assert propose_rules(single, min_deciders=1) != []
    assert propose_rules(refunds()[:4]) == []


def test_a_denial_inside_the_bounds_stops_the_rule_and_outside_is_counted() -> None:
    inside = record(False, approval_id="d1", amount=30, currency="EUR", order="o-1")
    assert propose_rules([*refunds(), inside]) == []
    (loose,) = propose_rules([*refunds(), inside], max_denied=1)
    assert loose.evidence.denied == 1
    outside = record(False, approval_id="d2", amount=900, currency="EUR", order="o-1")
    edited = record(
        True, approval_id="e", edited={"amount": 1}, amount=5, currency="EUR", order="o-1"
    )
    (rule,) = propose_rules([*refunds(), outside, edited])
    assert (rule.evidence.denied, rule.evidence.denied_outside) == (0, 2)


def test_unboundable_shapes_get_no_rule() -> None:
    many = [record(True, approval_id=f"m{n}", by=f"p{n}", note=f"text {n}") for n in range(12)]
    assert propose_rules(many) == []
    nested = [record(True, approval_id=f"n{n}", by=f"p{n}", order={"id": 1}) for n in range(5)]
    assert propose_rules(nested) == []
    assert propose_rules(refunds(), tools=["ship"]) == []


def test_each_argument_shape_gets_its_own_rule() -> None:
    other = [record(True, approval_id=f"x{n}", by=f"p{n}", region="eu") for n in range(5)]
    rules = propose_rules([*refunds(), *other])
    assert sorted(len(r.conditions) for r in rules) == [1, 3]


def test_a_rule_stops_skipping_after_its_review_date() -> None:
    accepted = accept(propose_rules(refunds())[0], accepted_by="ann")
    assert accepted.config["review_by"] == pytest.approx(accepted.accepted_at + 90 * 86_400, abs=1)
    rule = accepted.confirm
    late = RuleConfirm(rule.tool, rule.conditions, rule.review_by, clock=lambda: rule.review_by + 1)
    assert late(call(amount=30, currency="EUR", order="o-1"), CTX) is True


def test_the_generated_code_runs_and_cannot_be_broken_by_odd_names() -> None:
    odd: dict[str, Any] = {"x\nimport os": "y'\n"}
    items = [record(True, approval_id=f"q{n}", by=f"p{n}", **odd) for n in range(5)]
    accepted = accept(propose_rules(items)[0], accepted_by="ann")
    namespace: dict[str, Any] = {}
    exec(accepted.code, namespace)  # the code a person commits
    confirm = namespace["confirm"]
    assert confirm(call(**{"x\nimport os": "y'\n"}), CTX) is False
    assert accepted.code.count("\n") == 4


@pytest.mark.parametrize("name", ["", "  ", "x" * 121, "ann\nbob"])
def test_accept_needs_a_named_person(name: str) -> None:
    with pytest.raises(ConfigError):
        accept(propose_rules(refunds())[0], accepted_by=name)


def test_accept_refuses_anything_but_a_proposal() -> None:
    with pytest.raises(ConfigError):
        accept({"tool": "refund"}, accepted_by="ann")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    [
        {"min_count": 1},
        {"min_count": True},
        {"max_denied": -1},
        {"min_deciders": 0},
        {"review_days": 0},
    ],
)
def test_bad_options_are_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        propose_rules([], **options)


def good(**condition: Any) -> dict[str, Any]:
    base = {"field": "n", "op": "between", "values": [1, 2]}
    return {"tool": "t", "conditions": [{**base, **condition}], "review_by": 2e9}


@pytest.mark.parametrize(
    "config",
    [
        {**good(), "extra": 1},
        {**good(), "tool": ""},
        {**good(), "review_by": float("inf")},
        {**good(), "conditions": "n"},
        good(field="api_key"),
        good(values=[2, 1]),
        good(values=[1, float("nan")]),
        good(values=[True, 2]),
        good(op="==", values=[1, 2]),
        good(op="in", values=[1.5, 2]),
        good(op=">="),
        good(values=[]),
        {**good(), "conditions": [good()["conditions"][0]] * 2},
        {**good(), "conditions": [{"field": "n"}]},
        ["tool", "conditions", "review_by"],
    ],
)
def test_bad_configs_are_refused(config: Any) -> None:
    with pytest.raises(ConfigError):
        confirm_from_config(config)


def test_an_in_rule_matches_exact_types_only() -> None:
    config = {"tool": "t", "conditions": [{"field": "n", "op": "in", "values": [1, "a"]}]}
    confirm = confirm_from_config({**config, "review_by": 2e9})
    assert confirm(ToolCall(name="t", arguments={"n": "a"}), CTX) is False
    assert confirm(ToolCall(name="t", arguments={"n": True}), CTX) is True
    assert confirm(ToolCall(name="t", arguments={"n": 1.0}), CTX) is True


def test_a_proposed_rule_is_data(tmp_path: Any) -> None:
    rule = propose_rules(refunds())[0]
    assert isinstance(rule, ProposedRule) and not callable(rule)
