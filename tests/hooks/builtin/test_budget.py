import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import BudgetExceeded, ConfigError
from agent_adapter.core.types import LedgerEntry
from agent_adapter.hooks import Hook
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.tracing import REQUIRED_DATA, EventType
from tests.fakes.sinks import RecordingLedger, RecordingSink
from tests.hooks.builtin.helpers import llm_inv, tool_inv


def ledger_with(*rows: tuple[str, str, float]) -> RecordingLedger:
    """Rows of (run_id, thread_id, cost_usd)."""
    ledger = RecordingLedger()
    for i, (run_id, thread_id, cost) in enumerate(rows):
        ledger.record(
            LedgerEntry(
                run_id=run_id, call_id=f"c{i}", kind="llm", cost_usd=cost, thread_id=thread_id
            )
        )
    return ledger


# run_1 / t1 spent 0.6; run_2 / t1 spent 0.5; run_3 / t2 spent 0.2. Thread t1: 1.1; all: 1.3.
LEDGER_ROWS = (("run_1", "t1", 0.6), ("run_2", "t1", 0.5), ("run_3", "t2", 0.2))


def ctx(run_id: str = "run_1", thread_id: str = "t1", **kw) -> ToolContext:
    return ToolContext(run_id=run_id, thread_id=thread_id, ledger=ledger_with(*LEDGER_ROWS), **kw)


@pytest.mark.parametrize(
    ("scope", "limit", "blocked", "spent"),
    [
        ("run", 0.7, False, 0.6),
        ("run", 0.6, True, 0.6),  # at the limit counts as exceeded
        ("thread", 1.2, False, 1.1),
        ("thread", 1.0, True, 1.1),
        ("global", 1.4, False, 1.3),
        ("global", 1.3, True, 1.3),
    ],
)
async def test_stop_per_scope(scope, limit, blocked, spent):
    hook = BudgetHook(limit, scope=scope)
    for method, inv in ((hook.before_tool, tool_inv(ctx())), (hook.before_llm, llm_inv(ctx()))):
        if not blocked:
            assert await method(inv) is None
            continue
        with pytest.raises(BudgetExceeded) as info:
            await method(inv)
        error = info.value
        assert (error.limit_usd, error.scope, error.run_id) == (limit, scope, "run_1")
        assert error.spent_usd == pytest.approx(spent)


async def test_thread_scope_without_thread_id_falls_back_to_the_run():
    hook = BudgetHook(0.65, scope="thread")
    assert await hook.before_tool(tool_inv(ctx(thread_id=""))) is None  # run_1 alone: 0.6
    with pytest.raises(BudgetExceeded) as info:
        await BudgetHook(0.6, scope="thread").before_tool(tool_inv(ctx(thread_id="")))
    assert info.value.scope == "run"


async def test_warn_logs_and_emits_once_per_run(caplog):
    sink = RecordingSink()
    hook = BudgetHook(0.5, on_exceed="warn", scope="global")
    first, second = ctx(sink=sink), ctx(run_id="run_2", sink=sink)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert await hook.before_tool(tool_inv(first)) is None
            assert await hook.before_llm(llm_inv(first)) is None
        assert await hook.before_tool(tool_inv(second)) is None
    warnings = [r for r in caplog.records if "budget" in r.getMessage()]
    assert len(warnings) == 2
    events = sink.of_type(EventType.BUDGET_WARNING)
    assert [e.run_id for e in events] == ["run_1", "run_2"]
    assert set(events[0].data) >= REQUIRED_DATA[EventType.BUDGET_WARNING]
    assert events[0].data["limit_usd"] == 0.5
    assert events[0].data["spent_usd"] == pytest.approx(1.3)
    assert events[0].data["scope"] == "global"


async def test_warn_without_sink_still_logs(caplog):
    with caplog.at_level(logging.WARNING):
        await BudgetHook(0.1, on_exceed="warn").before_tool(tool_inv(ctx()))
    assert "budget" in caplog.text


async def test_warned_runs_are_bounded():
    hook = BudgetHook(0.0, on_exceed="warn", scope="global")
    hook._max_warned_runs = 2
    for run_id in ("a", "b", "c", "a"):
        await hook.before_tool(tool_inv(ctx(run_id=run_id)))
    assert list(hook._warned) == ["c", "a"]


async def test_no_ledger_is_a_noop_with_one_warning(caplog):
    hook = BudgetHook(0.0)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert await hook.before_tool(tool_inv(ToolContext(run_id="r"))) is None
            assert await hook.before_llm(llm_inv(ToolContext(run_id="r"))) is None
    assert len([r for r in caplog.records if "no ledger" in r.getMessage()]) == 1


def test_config_round_trip_and_schema():
    hook = BudgetHook(2.0, on_exceed="warn", scope="thread")
    assert hook.config() == {"max_usd": 2.0, "on_exceed": "warn", "scope": "thread"}
    schema = BudgetHook.config_schema()
    assert schema["required"] == ["max_usd"] and schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"max_usd", "on_exceed", "scope", "per"}
    changed = hook.with_config({"max_usd": 5})
    assert isinstance(changed, Hook) and changed is not hook
    assert changed.config() == {"max_usd": 5, "on_exceed": "warn", "scope": "thread"}
    assert hook.config()["max_usd"] == 2.0
    with pytest.raises(ConfigError, match="unknown"):
        hook.with_config({"max_dollars": 1})


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((-1,), {}, "max_usd"),
        ((True,), {}, "max_usd"),
        ((float("nan"),), {}, "max_usd"),
        (("1",), {}, "max_usd"),
        ((1,), {"on_exceed": "explode"}, "on_exceed"),
        ((1,), {"scope": "team"}, "scope"),
    ],
)
def test_bad_config_is_rejected(args, kwargs, match):
    with pytest.raises(ConfigError, match=match):
        BudgetHook(*args, **kwargs)
