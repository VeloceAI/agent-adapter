"""Cookbook: cap what each user may spend per day, with one ledger per user and day."""

import asyncio
from datetime import date

from agent_adapter import (
    BudgetHook,
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    Registry,
    Usage,
)
from agent_adapter.errors import BudgetExceeded

# Illustrative prices: USD per 1M tokens.
PRICES = Prices.load(
    {"version": "demo", "models": {"house": {"provider": "host", "input": 2.0, "output": 8.0}}}
)


async def house(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: each call costs 50k input tokens ($0.10)."""
    reply = Message(role="assistant", content="ok")
    usage = Usage(input_tokens=50_000)
    return LLMResponse(provider="host", model="house", text="ok", message=reply, usage=usage)


# --8<-- [start:per_user]
base = Executor(Registry(), hooks=[BudgetHook(0.25, scope="global")])  # $0.25 a day
views: dict[tuple[str, date], Executor] = {}


def executor_for(user_id: str, today: date) -> Executor:
    """This user's executor for today: the same hooks, the user's own ledger."""
    key = (user_id, today)
    if key not in views:
        # "global" reads the context's ledger, so it sums only this user's day
        views[key] = base.replace(ledger=Ledger(prices=PRICES))
    return views[key]


# --8<-- [end:per_user]


async def main() -> None:
    # --8<-- [start:spend]
    ask = LLMRequest(messages=(Message(role="user", content="hi"),))
    model = CallableModel(house, provider="host", model="house")
    today = date(2026, 10, 30)
    for user in ("asha", "asha", "asha", "asha", "ravi"):
        executor = executor_for(user, today)
        try:
            async with executor.run_scope() as ctx:  # one run per request
                await executor.model(model).bind(ctx).generate(ask)  # $0.10
            print(user, "ok", round(executor.ledger.total_usd(), 2) if executor.ledger else 0)
        except BudgetExceeded as exc:
            print(user, "stopped", exc.scope, round(exc.spent_usd, 2))
    # --8<-- [end:spend]


if __name__ == "__main__":
    asyncio.run(main())
