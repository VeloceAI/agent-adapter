"""Shared builders for the BudgetHook scope tests: prices, contexts, model calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.types import LedgerEntry, Usage
from agent_adapter.cost.prices import Prices
from agent_adapter.hooks import LLMInvocation

# USD per 1M tokens: 100k input tokens cost $0.20; the estimate of "hi" with 1000 output
# tokens is 1 input token ($0.000002) + 1000 output tokens ($0.008).
PRICES = Prices.load(
    {"version": "t", "models": {"house": {"provider": "host", "input": 2.0, "output": 8.0}}}
)
ESTIMATE = 0.008002
COST = 0.2
DAY = 86_400.0
NOW = 1_800_000_000.0  # 2027-01-15T08:00:00Z


@dataclass(frozen=True)
class Who:
    """Stands in for the server's Principal."""

    id: str


def ctx(run_id: str = "run_1", *, user: str = "", principal: Any = None, **kw: Any) -> ToolContext:
    extras = dict(kw.pop("extras", {}))
    if principal is not None:
        extras["principal"] = principal
    return ToolContext(run_id=run_id, user_id=user, extras=extras, **kw)


def ask(context: ToolContext, *, model: str = "house") -> LLMInvocation:
    request = LLMRequest(messages=(Message(role="user", content="hi"),), max_output_tokens=1000)
    return LLMInvocation(request, "host", model, context)


def answer(*, cached: bool = False, model: str = "house", tokens: int = 100_000) -> LLMResponse:
    return LLMResponse(
        provider="host",
        model=model,
        text="ok",
        message=Message(role="assistant", content="ok"),
        usage=Usage(input_tokens=tokens),
        cached=cached,
    )


class FrozenClock:
    """A settable clock (epoch seconds)."""

    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def spend(hook: Any, context: ToolContext, **response: Any) -> None:
    """One full model call through the hook's phases."""
    inv = ask(context)
    await hook.before_llm(inv)
    await hook.on_llm_end(inv, bill(inv, **response))


def bill(
    inv: LLMInvocation, *, cached: bool = False, model: str = "house", tokens: int = 100_000
) -> LedgerEntry:
    """The ledger entry an instrumented call records: unpriced, or $0 for a cache hit."""
    return LedgerEntry.from_usage(
        Usage(input_tokens=tokens),
        kind="llm",
        run_id=inv.ctx.run_id,
        call_id="llm:0",
        model=model,
        cost_usd=0.0 if cached else None,
        cache_hit=cached,
    )
