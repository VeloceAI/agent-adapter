"""GET /budget: callers see their own user and team budgets; approvers see any key."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import Usage
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec.executor import Executor
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.hooks.builtin.budgets import MemoryBudgetStore
from agent_adapter.models.callable import CallableModel
from tests.hooks.builtin.budgets.helpers import PRICES
from tests.server.helpers import call, make_service

TEAMS = {"alice@example.com": "shop", "bob@example.com": "shop", "": "guests"}


async def house(request: LLMRequest) -> LLMResponse:
    return LLMResponse(
        provider="host",
        model="house",
        text="ok",
        message=Message(role="assistant", content="ok"),
        usage=Usage(input_tokens=100_000),  # $0.20
    )


def team_of(ctx: ToolContext) -> str:
    return TEAMS[ctx.extras["principal"].id]


def budget_service(**options: Any) -> Any:
    store = MemoryBudgetStore()
    hooks = [
        BudgetHook(0.4, scope="principal", prices=PRICES, store=store, estimate_usd=0.0),
        BudgetHook(1.0, scope="team", scope_key=team_of, prices=PRICES, store=store),
        BudgetHook(5.0, scope="global", per="day", prices=PRICES, store=store),
        BudgetHook(9.0),  # a run budget: per run, never listed here
    ]
    registry = Registry()
    executor = Executor(registry, hooks=hooks, ledger=Ledger(prices=PRICES))
    model = executor.model(CallableModel(house, provider="host", model="house"))

    @registry.tool
    async def summarise(ctx: ToolContext) -> str:
        """Ask the house model once."""
        ask = LLMRequest(messages=(Message(role="user", content="hi"),), max_output_tokens=10)
        return (await model.bind(ctx).generate(ask)).text

    return make_service(registry, executor=executor, **options)


def by_scope(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["scope"]: item for item in body["budgets"]}


async def test_a_caller_sees_its_own_user_and_team_budgets():
    service = budget_service()
    for _ in range(2):
        assert (
            await call(service, "POST", "/tools/summarise/invoke", body={"arguments": {}})
        ).status == 200
    mine = by_scope((await call(service, "GET", "/budget")).body)
    assert set(mine) == {"principal", "team"}
    assert mine["principal"]["spent_usd"] == pytest.approx(0.4)
    assert mine["principal"]["remaining_usd"] == 0.0 and mine["principal"]["window"] == "all"
    assert mine["team"]["key"] == "shop" and mine["team"]["spent_usd"] == pytest.approx(0.4)
    bob = by_scope((await call(service, "GET", "/budget", token="t-bob")).body)
    assert bob["principal"]["spent_usd"] == 0.0 and bob["team"]["spent_usd"] == pytest.approx(0.4)


async def test_spending_past_the_limit_is_refused_and_shows_in_the_budget():
    service = budget_service()
    for _ in range(2):
        await call(service, "POST", "/tools/summarise/invoke", body={"arguments": {}})
    over = await call(service, "POST", "/tools/summarise/invoke", body={"arguments": {}})
    assert over.status == 429 and over.body["type"].endswith("budget-exceeded")
    one = await call(service, "GET", "/budget/principal")
    assert [b["scope"] for b in one.body["budgets"]] == ["principal"]


async def test_only_approvers_read_another_key_or_the_global_budget():
    service = budget_service()
    await call(service, "POST", "/tools/summarise/invoke", body={"arguments": {}})
    peek = await call(service, "GET", "/budget/principal", query={"key": "bob@example.com"})
    assert peek.status == 403
    assert (await call(service, "GET", "/budget/global")).status == 403
    seen = await call(
        service, "GET", "/budget/principal", token="t-reviewer", query={"key": "alice@example.com"}
    )
    assert seen.body["budgets"][0]["spent_usd"] == pytest.approx(0.2)
    team = await call(service, "GET", "/budget/team", token="t-reviewer", query={"key": "shop"})
    assert team.body["budgets"][0]["spent_usd"] == pytest.approx(0.2)
    everything = by_scope((await call(service, "GET", "/budget", token="t-reviewer")).body)
    assert set(everything) == {"principal", "global"}  # the reviewer is in no team
    assert everything["global"]["spent_usd"] == pytest.approx(0.2)
    assert everything["global"]["window"].startswith("day:")


async def test_bad_requests_and_missing_budgets():
    service = budget_service()
    assert (await call(service, "GET", "/budget/run")).status == 400
    assert (await call(service, "GET", "/budget/team", query={"key": "a/b"})).status == 400
    unscoped = await call(service, "GET", "/budget", token="t-reviewer", query={"key": "shop"})
    assert unscoped.status == 400  # one key would name a user and a team at once
    assert (await call(service, "GET", "/budget", token=None)).status == 401
    assert (await call(service, "POST", "/budget", body={})).status == 405
    assert (await call(make_service(), "GET", "/budget")).status == 404


async def test_anonymous_callers_share_one_budget():
    service = budget_service(authenticate=None, allow_anonymous=True)
    await call(service, "POST", "/tools/summarise/invoke", body={"arguments": {}}, token=None)
    mine = by_scope((await call(service, "GET", "/budget", token=None)).body)
    assert mine["principal"]["key"] == "" and mine["principal"]["spent_usd"] == pytest.approx(0.2)
