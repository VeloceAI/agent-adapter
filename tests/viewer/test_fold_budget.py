"""The run summary marks a run where a budget refused a call."""

from __future__ import annotations

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import BudgetExceeded
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.viewer._fold import Fold, summarise
from tests.fakes.sinks import RecordingSink
from tests.hooks.builtin.budgets.helpers import PRICES, spend


def record(event_type: str, ts: float = 1.0) -> dict:
    return {"kind": "event", "type": event_type, "ts": ts, "data": {}}


def test_a_budget_exceeded_event_marks_the_run():
    fold = Fold()
    for line in (record("run.started"), record("budget.exceeded"), record("run.finished", 2.0)):
        fold.add(line)
    assert summarise("run_1", [fold], 0.0, set())["budget_exceeded"] is True


def test_a_run_within_budget_has_no_mark():
    fold = Fold()
    fold.add(record("budget.reserved"))
    assert "budget_exceeded" not in summarise("run_1", [fold], 0.0, set())


async def test_the_hook_emits_what_the_viewer_folds():
    sink = RecordingSink()
    hook = BudgetHook(0.1, scope="principal", prices=PRICES)
    context = ToolContext(run_id="run_1", user_id="ana", sink=sink)
    await spend(hook, context)
    with pytest.raises(BudgetExceeded):
        await spend(hook, context)
    fold = Fold()
    for event in sink.events:
        fold.add({"kind": "event", "type": event.type, "ts": event.ts, "data": event.data})
    assert summarise("run_1", [fold], 0.0, set())["budget_exceeded"] is True
