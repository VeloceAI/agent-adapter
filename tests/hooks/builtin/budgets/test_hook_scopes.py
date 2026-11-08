"""BudgetHook with counters: per user, per team, per UTC window; reserve, settle, release."""

from __future__ import annotations

import logging

import pytest

from agent_adapter.core.errors import BudgetExceeded, ConfigError
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.hooks.builtin.budgets import MemoryBudgetStore
from agent_adapter.tracing import REQUIRED_DATA, EventType
from tests.fakes.sinks import RecordingSink
from tests.hooks.builtin.budgets.helpers import (
    COST,
    DAY,
    ESTIMATE,
    PRICES,
    FrozenClock,
    Who,
    ask,
    bill,
    ctx,
    spend,
)
from tests.hooks.builtin.helpers import tool_inv


def hook(max_usd: float = 0.5, **kw) -> BudgetHook:
    kw.setdefault("scope", "principal")
    kw.setdefault("prices", PRICES)
    return BudgetHook(max_usd, **kw)


async def test_a_principal_is_charged_across_runs_and_users_are_separate():
    budget = hook(0.4)
    for run in ("run_1", "run_2"):
        await spend(budget, ctx(run, user="ana"))
    status = await budget.status(ctx(user="ana"))
    assert status.spent_usd == pytest.approx(2 * COST) and status.reserved_usd == 0.0
    with pytest.raises(BudgetExceeded) as info:
        await budget.before_llm(ask(ctx("run_3", user="ana")))
    assert info.value.scope == "principal" and info.value.run_id == "run_3"
    await spend(budget, ctx(user="bo"))  # bo has his own counter
    assert (await budget.status(ctx(user="bo"))).spent_usd == pytest.approx(COST)


@pytest.mark.parametrize("principal", [Who("ana"), {"id": "ana"}, "ana"])
async def test_the_server_principal_names_the_user_when_user_id_is_empty(principal):
    budget = hook()
    await spend(budget, ctx(principal=principal))
    assert (await budget.status(ctx(user="ana"))).spent_usd == pytest.approx(COST)


async def test_user_and_principal_scopes_share_one_counter():
    store = MemoryBudgetStore()
    await spend(hook(scope="user", store=store), ctx(user="ana"))
    status = await hook(scope="principal", store=store).status(ctx(user="ana"))
    assert status.spent_usd == pytest.approx(COST)


async def test_callers_without_an_identity_share_the_anonymous_counter():
    budget = hook()
    await spend(budget, ctx())
    assert (await budget.status(ctx(principal=Who("")))).spent_usd == pytest.approx(COST)
    assert (await budget.status(ctx(principal=object()))).spent_usd == pytest.approx(COST)


async def test_team_scope_counts_by_the_key_function():
    budget = hook(0.4, scope="team", scope_key=lambda c: c.extras["team"])
    await spend(budget, ctx(user="ana", extras={"team": "shop"}))
    await spend(budget, ctx(user="bo", extras={"team": "shop"}))
    with pytest.raises(BudgetExceeded):
        await budget.before_llm(ask(ctx(user="cy", extras={"team": "shop"})))
    assert await budget.before_llm(ask(ctx(extras={"team": "museum"}))) is None


@pytest.mark.parametrize("key", [lambda c: "", lambda c: 7, lambda c: c.extras["missing"]])
async def test_a_failing_team_key_refuses_the_call(key, caplog):
    budget = hook(scope="team", scope_key=key)
    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="scope_key"):
        await budget.before_llm(ask(ctx()))


@pytest.mark.parametrize(("per", "later", "resets"), [("day", DAY, True), ("month", DAY, False)])
async def test_windows_reset_on_utc_boundaries(per, later, resets):
    clock = FrozenClock()
    budget = hook(0.3, per=per, clock=clock)
    await spend(budget, ctx(user="ana"))
    first = await budget.status(ctx(user="ana"))
    assert first.window.startswith(per) and first.resets_at is not None
    clock.now += later
    status = await budget.status(ctx(user="ana"))
    assert (status.spent_usd == 0.0) is resets
    assert status.remaining_usd == pytest.approx(0.3 if resets else 0.1)


async def test_the_estimate_is_held_until_the_call_settles():
    budget = hook()
    inv = ask(ctx(user="ana"))
    await budget.before_llm(inv)
    held = await budget.status(ctx(user="ana"))
    assert held.reserved_usd == pytest.approx(ESTIMATE) and held.spent_usd == 0.0
    await budget.on_llm_end(inv, bill(inv))
    settled = await budget.status(ctx(user="ana"))
    assert (settled.reserved_usd, settled.spent_usd) == (0.0, pytest.approx(COST))
    await budget.on_llm_end(inv, bill(inv))  # settled once only
    assert (await budget.status(ctx(user="ana"))).spent_usd == pytest.approx(COST)


async def test_a_failed_call_releases_its_hold():
    budget = hook()
    inv = ask(ctx(user="ana"))
    await budget.before_llm(inv)
    assert await budget.on_llm_end(inv, None) is None
    status = await budget.status(ctx(user="ana"))
    assert (status.reserved_usd, status.spent_usd) == (0.0, 0.0)
    await budget.on_llm_end(inv, None)  # nothing left to release


async def test_a_cache_hit_costs_nothing():
    budget = hook()
    await spend(budget, ctx(user="ana"), cached=True)
    assert (await budget.status(ctx(user="ana"))).spent_usd == 0.0


async def test_an_estimate_that_does_not_fit_is_refused():
    budget = hook(0.25, estimate_usd=0.1)
    await spend(budget, ctx(user="ana"))  # 0.2 spent; 0.2 + 0.1 > 0.25
    with pytest.raises(BudgetExceeded) as info:
        await budget.before_llm(ask(ctx(user="ana")))
    assert info.value.spent_usd == pytest.approx(COST)


async def test_events_say_reserved_and_exceeded_without_naming_the_user():
    sink = RecordingSink()
    budget = hook(0.2)
    await spend(budget, ctx(user="ana@example.com", sink=sink))
    with pytest.raises(BudgetExceeded):
        await spend(budget, ctx(user="ana@example.com", sink=sink))
    reserved = sink.of_type(EventType.BUDGET_RESERVED)
    exceeded = sink.of_type(EventType.BUDGET_EXCEEDED)
    assert len(reserved) == 1 and len(exceeded) == 1
    assert set(reserved[0].data) >= REQUIRED_DATA[EventType.BUDGET_RESERVED]
    assert set(exceeded[0].data) >= REQUIRED_DATA[EventType.BUDGET_EXCEEDED]
    assert reserved[0].data["reserved_usd"] == pytest.approx(ESTIMATE)
    assert exceeded[0].data["window"] == "all"
    assert "ana" not in repr([e.data for e in sink.events])


async def test_warn_mode_never_refuses_and_warns_once_per_run(caplog):
    sink = RecordingSink()
    budget = hook(0.1, on_exceed="warn")
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            await spend(budget, ctx(user="ana", sink=sink))
        await budget.before_tool(tool_inv(ctx(user="ana", sink=sink)))
    assert len(sink.of_type(EventType.BUDGET_WARNING)) == 1
    assert (await budget.status(ctx(user="ana"))).spent_usd == pytest.approx(3 * COST)


async def test_tool_calls_are_checked_but_hold_nothing():
    budget = hook(0.2)
    assert await budget.before_tool(tool_inv(ctx(user="ana"))) is None
    assert (await budget.status(ctx(user="ana"))).reserved_usd == 0.0
    await spend(budget, ctx(user="ana"))
    with pytest.raises(BudgetExceeded):
        await budget.before_tool(tool_inv(ctx(user="ana")))


async def test_prices_come_from_the_ledger_when_the_hook_has_none():
    from agent_adapter.cost.ledger import Ledger

    budget = BudgetHook(1.0, scope="principal")
    await spend(budget, ctx(user="ana", ledger=Ledger(prices=PRICES)))
    assert (await budget.status(ctx(user="ana"))).spent_usd == pytest.approx(COST)


async def test_an_unpriced_model_counts_as_zero_and_warns_once(caplog):
    budget = BudgetHook(1.0, scope="principal")  # no prices, no ledger
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            await spend(budget, ctx(user="ana"))
    assert len([r for r in caplog.records if "no price" in r.getMessage()]) == 1
    assert (await budget.status(ctx(user="ana"))).spent_usd == 0.0


async def test_a_callable_estimate_sees_the_invocation():
    seen = []
    budget = hook(estimate_usd=lambda inv: seen.append(inv.model) or 0.05)
    inv = ask(ctx(user="ana"))
    await budget.before_llm(inv)
    assert seen == ["house"]
    assert (await budget.status(ctx(user="ana"))).reserved_usd == pytest.approx(0.05)


@pytest.mark.parametrize("bad", [lambda inv: "x", lambda inv: -1.0, lambda inv: float("inf")])
async def test_a_bad_estimate_refuses_the_call(bad):
    with pytest.raises(ConfigError, match="estimate_usd"):
        await hook(estimate_usd=bad).before_llm(ask(ctx(user="ana")))


async def test_two_hooks_on_one_store_share_the_limit():
    store = MemoryBudgetStore()
    worker_a, worker_b = hook(0.2, store=store), hook(0.2, store=store)
    await spend(worker_a, ctx(user="ana"))
    with pytest.raises(BudgetExceeded):
        await spend(worker_b, ctx(user="ana"))


async def test_named_budgets_on_one_store_keep_separate_counters():
    store = MemoryBudgetStore()
    await spend(hook(store=store, budget="chat"), ctx(user="ana"))
    assert (await hook(store=store, budget="search").status(ctx(user="ana"))).spent_usd == 0.0


@pytest.mark.parametrize(
    ("scope", "context"), [("run", ctx("r9")), ("thread", ctx(thread_id="t1"))]
)
async def test_run_and_thread_scopes_use_the_store_when_given(scope, context):
    budget = hook(0.2, scope=scope, store=MemoryBudgetStore())
    await spend(budget, context)
    with pytest.raises(BudgetExceeded) as info:
        await spend(budget, context)
    assert info.value.scope == scope


async def test_status_of_another_key_and_its_view():
    budget = hook(per="day", clock=FrozenClock())
    await spend(budget, ctx(user="bo"))
    status = await budget.status(ctx(user="ana"), scope_id="bo")
    view = status.view()
    assert view["key"] == "bo" and view["scope"] == "principal"
    assert view["spent_usd"] == pytest.approx(COST)
    assert view["remaining_usd"] == pytest.approx(0.3)
    assert view["window"] == "day:2027-01-15" and view["resets_at"] == 1_800_057_600.0


async def test_ledger_mode_status_reads_the_ledger():
    from agent_adapter.core.types import LedgerEntry
    from agent_adapter.cost.ledger import Ledger

    ledger = Ledger(prices=PRICES)
    ledger.record(LedgerEntry(run_id="r1", call_id="c1", kind="llm", cost_usd=0.4))
    status = await BudgetHook(1.0, scope="global").status(ctx(ledger=ledger))
    assert (status.scope, status.scope_id, status.spent_usd) == ("global", "*", 0.4)
    assert (await BudgetHook(1.0).status(ctx())).spent_usd == 0.0  # no ledger
