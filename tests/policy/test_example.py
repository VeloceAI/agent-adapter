"""The policy example runs offline and shows access, approval, guard and explain."""

import runpy
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "policy_shop.py"


def test_policy_example_runs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AA_ENV", raising=False)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "o-1: shipped",
        "refunded 30 on o-1",
        "approval asked for refund",
        "refunded 300 on o-2",
        "wipe_orders -> approval-denied",
        "refund -> approval-denied",
        "send_note -> guardrail-blocked",
        "allowed for some callers (rules[1] allows some callers)",
        "approval: when amount > 100 (rules[1]) [tools refund; teams support]",
        "budget budget: 20 USD per user per day, stop (budgets[0])",
    ]
