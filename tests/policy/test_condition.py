"""Condition: the safe expression grammar, its evaluation, and what it refuses."""

from __future__ import annotations

import pytest

from agent_adapter.policy import Condition, ConditionError


@pytest.mark.parametrize(
    ("text", "args", "expected"),
    [
        ("amount > 100", {"amount": 150}, True),
        ("amount > 100", {"amount": 100}, False),
        ("amount >= 100", {"amount": 100}, True),
        ("amount < -5", {"amount": -10}, True),
        ("0 < amount <= 10", {"amount": 10}, True),
        ("0 < amount <= 10", {"amount": 11}, False),
        ("currency == 'EUR'", {"currency": "EUR"}, True),
        ("currency != 'EUR'", {"currency": "EUR"}, False),
        ("country in ['FR', 'DE']", {"country": "DE"}, True),
        ("country not in ['FR', 'DE']", {"country": "DE"}, False),
        ("'vip' in tags", {"tags": ["vip", "new"]}, True),
        ("'gift' in note", {"note": "a gift card"}, True),
        ("not express", {"express": True}, False),
        ("express == true", {"express": True}, True),
        ("coupon == null", {"coupon": None}, True),
        ("amount > 50 and currency == 'USD'", {"amount": 60, "currency": "EUR"}, False),
        ("amount > 50 or currency == 'USD'", {"amount": 10, "currency": "USD"}, True),
        ("order.total > 20", {"order": {"total": 25}}, True),
        ("name < 'm'", {"name": "anna"}, True),
    ],
)
def test_condition_evaluates(text: str, args: dict[str, object], expected: bool) -> None:
    assert Condition(text).evaluate(args) is expected


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('x')",
        "open('f')",
        "amount.__class__",
        "_hidden > 1",
        "items[0] > 1",
        "lambda: 1",
        "amount if amount else 0",
        "amount + 1 > 2",
        "x is None",
        "{'a': 1} == x",
        "-x > 1",
        "-True == x",
        "x in [y]",
        "b'x' == x",
        "password == 'x'",
        "auth.api_key == 'x'",
        "",
        "   ",
        "amount >",
        "f'{x}' == 'y'",
        "x" * 501,
        " or ".join(f"a{i} == {i}" for i in range(40)),
        "1",
        "x",
        "x == -'a'",
        "x in [-null]",
        "x == -true",
    ],
)
def test_condition_rejects(text: str) -> None:
    with pytest.raises(ConditionError):
        Condition(text)


def test_condition_rejects_a_non_string() -> None:
    with pytest.raises(ConditionError):
        Condition(5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "args"),
    [
        ("amount > 100", {}),
        ("order.total > 1", {"order": 5}),
        ("amount > 100", {"amount": None}),
        ("amount > 100", {"amount": "lots"}),
        ("amount > 100", {"amount": True}),
        ("amount > 1", {"amount": float("nan")}),
        ("amount and express", {"amount": 3, "express": True}),
        ("x in limit", {"x": 1, "limit": 5}),
        ("x in note", {"x": 1, "note": "text"}),
        ("amount == '100'", {"amount": 100}),
        ("'100' != amount", {"amount": 100}),
    ],
)
def test_condition_undecidable_raises(text: str, args: dict[str, object]) -> None:
    with pytest.raises(ConditionError):
        Condition(text).evaluate(args)


def test_condition_identity_and_fields() -> None:
    first = Condition(" amount > 1 and order.total < 2 ")
    assert first == Condition("amount > 1 and order.total < 2")
    assert first != "amount > 1"
    assert hash(first) == hash(Condition("amount > 1 and order.total < 2"))
    assert first.fields == {"amount", "order"}
    assert "amount > 1" in repr(first)
    assert Condition("x == true").fields == {"x"}


def test_huge_ints_compare_exactly() -> None:
    assert Condition("units > 100").evaluate({"units": 10**400}) is True
    assert Condition("units < 100").evaluate({"units": -(10**400)}) is True


def test_shown_hides_string_literals() -> None:
    shown = Condition("email == 'ceo@shop.example' and amount > 100").shown()
    assert "ceo@shop.example" not in shown
    assert shown == "email == '...' and amount > 100"


def test_null_compares_with_strings() -> None:
    assert Condition("coupon == null").evaluate({"coupon": "SAVE"}) is False
