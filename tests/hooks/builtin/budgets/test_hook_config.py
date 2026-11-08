"""BudgetHook settings, tuning, failure handling and a real executor under concurrency."""

from __future__ import annotations

import asyncio
import logging

import pytest

from agent_adapter.core.errors import BudgetExceeded, ConfigError
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.types import Usage
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec.executor import Executor
from agent_adapter.hooks.builtin import budget as budget_module
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.hooks.builtin.budgets import BudgetUsage, MemoryBudgetStore
from agent_adapter.models.callable import CallableModel
from tests.hooks.builtin.budgets.helpers import COST, PRICES, ask, bill, ctx


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scope": "team"}, "scope_key"),
        ({"scope": "principal", "scope_key": lambda c: "x"}, "scope_key"),
        ({"per": "week"}, "per"),
        ({"store": object()}, "store"),
        ({"budget": ""}, "budget"),
        ({"estimate_usd": "1"}, "estimate_usd"),
        ({"estimate_usd": True}, "estimate_usd"),
        ({"estimate_usd": -0.5}, "estimate_usd"),
        ({"hold_s": 0}, "hold_s"),
        ({"hold_s": True}, "hold_s"),
    ],
)
def test_bad_settings_are_refused(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        BudgetHook(1.0, **kwargs)


def test_ledger_mode_is_kept_for_the_old_scopes():
    for scope in ("run", "thread", "global"):
        assert BudgetHook(1.0, scope=scope).store is None
    assert BudgetHook(1.0, per="day").counted
    assert isinstance(BudgetHook(1.0, scope="user").store, MemoryBudgetStore)


def test_config_names_per_only_when_set_and_with_config_keeps_the_store():
    store = MemoryBudgetStore()
    team = BudgetHook(2.0, scope="team", scope_key=lambda c: "t", store=store, budget="b")
    assert team.config() == {"max_usd": 2.0, "on_exceed": "stop"}  # scope not tunable
    daily = team.with_config({"per": "day", "max_usd": 3})
    assert daily.config() == {"max_usd": 3, "on_exceed": "stop", "per": "day"}
    assert daily.scope == "team"
    assert daily.store is store and daily.scope_key is team.scope_key and daily.budget == "b"
    assert BudgetHook(1.0).with_config({"per": "month"}).counted
    plain = BudgetHook(1.0, per="day").with_config({"per": None})
    assert plain.store is None  # back to the ledger


async def test_open_holds_are_bounded(monkeypatch, caplog):
    monkeypatch.setattr(budget_module, "MAX_HELD", 2)
    hook = BudgetHook(10.0, scope="principal", prices=PRICES)
    invs = [ask(ctx(user="ana")) for _ in range(3)]
    with caplog.at_level(logging.WARNING):
        for inv in invs:
            await hook.before_llm(inv)
    assert "unsettled hold" in caplog.text
    await hook.on_llm_end(invs[0], bill(invs[0]))  # forgotten: nothing settled
    assert (await hook.status(ctx(user="ana"))).spent_usd == 0.0


class BrokenStore(MemoryBudgetStore):
    async def settle(self, key, reservation_id, cost_usd, *, expire_at):
        raise OSError("disk gone")

    async def release(self, key, reservation_id):
        raise OSError("disk gone")


async def test_settle_and_release_failures_are_logged_not_raised(caplog):
    hook = BudgetHook(1.0, scope="principal", prices=PRICES, store=BrokenStore())
    first, second = ask(ctx(user="ana")), ask(ctx(user="ana"))
    await hook.before_llm(first)
    await hook.before_llm(second)
    with caplog.at_level(logging.ERROR):
        await hook.on_llm_end(first, bill(first))
        await hook.on_llm_end(second, None)
    assert caplog.text.count("budget settle failed") == 2
    assert "disk gone" not in caplog.text


class RefusingStore(MemoryBudgetStore):
    """Refuses every hold although the usage it reports fits."""

    async def reserve(self, key, reservation_id, amount_usd, **kw):
        from agent_adapter.hooks.builtin.budgets import Reserved

        return Reserved(False, BudgetUsage())


async def test_a_refused_hold_always_stops():
    hook = BudgetHook(1.0, scope="principal", prices=PRICES, store=RefusingStore())
    with pytest.raises(BudgetExceeded):
        await hook.before_llm(ask(ctx(user="ana")))


async def test_a_store_failure_fails_the_check_closed():
    class Down(MemoryBudgetStore):
        async def reserve(self, *args, **kw):
            raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await BudgetHook(1.0, scope="principal", store=Down()).before_llm(ask(ctx(user="a")))


async def test_concurrent_model_calls_through_the_executor_never_overspend():
    async def house(request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0.01)
        return LLMResponse(
            provider="host",
            model="house",
            text="ok",
            message=Message(role="assistant", content="ok"),
            usage=Usage(input_tokens=100_000),
        )

    hook = BudgetHook(1.0, scope="principal", estimate_usd=COST)
    executor = Executor(ledger=Ledger(prices=PRICES), hooks=[hook])
    model = executor.model(CallableModel(house, provider="host", model="house"))
    ask_once = LLMRequest(messages=(Message(role="user", content="hi"),))

    async def one(i: int) -> bool:
        context = executor.context(run_id=f"run_{i}", user_id="ana")
        try:
            await model.bind(context).generate(ask_once)
        except BudgetExceeded:
            return False
        return True

    results = await asyncio.gather(*(one(i) for i in range(12)))
    assert sum(results) == 5  # five holds of $0.20 fill $1.00 while all twelve race
    status = await hook.status(executor.context(user_id="ana"))
    assert status.spent_usd == pytest.approx(1.0) and status.reserved_usd == 0.0


def test_an_unknown_scope_is_refused():
    with pytest.raises(ConfigError, match="scope must be one of"):
        BudgetHook(1.0, scope="tenant")  # type: ignore[arg-type]


class OddPrices:
    def __init__(self, result: object) -> None:
        self.result = result

    def cost(self, model, usage, *, at=None, seconds=0.0, attempts=1):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("result", [ValueError("bad table"), -1.0])
async def test_a_broken_or_negative_price_counts_as_unpriced(result, caplog):
    hook = BudgetHook(1.0, scope="principal", prices=OddPrices(result))  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        inv = ask(ctx(user="ana"))
        await hook.before_llm(inv)
        await hook.on_llm_end(inv, bill(inv))
    status = await hook.status(ctx(user="ana"))
    assert (status.spent_usd, status.reserved_usd) == (0.0, 0.0)
    assert "bad table" not in caplog.text


async def test_a_call_refused_by_a_later_hook_gives_its_hold_back():
    async def house(request: LLMRequest) -> LLMResponse:
        raise AssertionError("never called")

    store = MemoryBudgetStore()
    user = BudgetHook(1.0, scope="user", estimate_usd=0.3, store=store)
    team = BudgetHook(0.0, scope="team", scope_key=lambda c: "desk", store=store)
    executor = Executor(hooks=[user, team])
    model = executor.model(CallableModel(house, provider="host", model="house"))
    context = executor.context(user_id="ana")
    refused = False
    try:
        await model.bind(context).generate(
            LLMRequest(messages=(Message(role="user", content="hi"),))
        )
    except BudgetExceeded:
        refused = True
    assert refused  # no gc.collect(): the release is explicit, at the end of the call
    assert (await user.status(context)).reserved_usd == 0.0
    assert len(user._holds) == 0
