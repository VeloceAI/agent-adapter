"""Team budgets: cap spend per user and per team per UTC day, shared by every worker."""

import asyncio
import tempfile
from pathlib import Path

from agent_adapter import (
    BudgetHook,
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    ToolContext,
    Usage,
)
from agent_adapter.errors import BudgetExceeded
from agent_adapter.hooks.builtin.budgets import BudgetStore, SQLiteBudgetStore

# Illustrative prices (USD per 1M tokens); load your own table in real code.
PRICES = Prices.load(
    {"version": "demo", "models": {"house-llm": {"provider": "host", "input": 2.0, "output": 8.0}}}
)


# --8<-- [start:hooks]
TEAMS = {"ana": "support", "bo": "support", "cy": "returns"}  # from your user directory


def team_of(ctx: ToolContext) -> str:
    """The team a call is charged to; an unknown user is refused (KeyError)."""
    return TEAMS[ctx.user_id]


def budgets(store: BudgetStore) -> list[BudgetHook]:
    """The same limits on every worker; the store holds the shared spend."""
    return [
        BudgetHook(0.50, scope="user", per="day", store=store),  # each agent: $0.50 a day
        BudgetHook(0.80, scope="team", per="day", scope_key=team_of, store=store),
    ]


# --8<-- [end:hooks]


async def house_llm(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: every call costs 100k input tokens ($0.20)."""
    return LLMResponse(
        provider="host",
        model="house-llm",
        text="ok",
        message=Message(role="assistant", content="ok"),
        usage=Usage(input_tokens=100_000),
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        # --8<-- [start:workers]
        store = SQLiteBudgetStore(Path(folder) / "budgets.db")  # one file, every process
        workers = [Executor(ledger=Ledger(prices=PRICES), hooks=budgets(store)) for _ in "ab"]
        house = CallableModel(house_llm, provider="host", model="house-llm")
        ask = LLMRequest(messages=(Message(role="user", content="Where is my parcel?"),))
        for turn, user in enumerate(["ana", "ana", "bo", "bo", "ana", "cy"]):
            worker = workers[turn % 2]  # requests land on either worker
            ctx = worker.context(user_id=user)  # the authenticated user, set by the host
            try:
                await worker.model(house).bind(ctx).generate(ask)
                print(user, "answered")
            except BudgetExceeded as exc:  # refused before the model is called
                print(user, "refused by the", exc.scope, "budget")
        # --8<-- [end:workers]
        # --8<-- [start:status]
        user_cap, team_cap = budgets(store)
        mine = await user_cap.status(workers[0].context(user_id="ana"))
        team = await team_cap.status(workers[0].context(user_id="ana"))
        print(mine.scope, mine.scope_id, round(mine.spent_usd, 2), round(mine.remaining_usd, 2))
        print(team.scope, team.scope_id, round(team.spent_usd, 2), team.window.split(":")[0])
        # --8<-- [end:status]


if __name__ == "__main__":
    asyncio.run(main())
