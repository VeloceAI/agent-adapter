"""Budgets: price every call in a ledger, cap spend per run or per thread, cap call counts."""

import asyncio

from agent_adapter import (
    BudgetHook,
    CallableModel,
    Executor,
    Ledger,
    LedgerEntry,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    Registry,
    RunLimits,
    ToolCall,
    ToolContext,
    Usage,
)
from agent_adapter.errors import BudgetExceeded, RunLimitExceeded
from agent_adapter.tracing import EventLog

registry = Registry()

# --8<-- [start:prices]
# Illustrative prices (USD per 1M tokens, USD per unit); load your own table in real code.
PRICES = Prices.load(
    {
        "version": "demo",
        "models": {"house-llm": {"provider": "host", "input": 2.0, "output": 8.0}},
        "units": {"acme:geocode": {"provider": "acme", "per_unit": 0.05}},
    }
)
# --8<-- [end:prices]


# --8<-- [start:paid_tool]
@registry.tool
def geocode(address: str, ctx: ToolContext) -> str:
    """Geocode an address with a paid API."""
    if ctx.ledger is not None:  # record what the vendor bills: one unit per lookup
        bill = LedgerEntry(
            run_id=ctx.run_id,
            call_id=ctx.call_id,
            entry_id=f"{ctx.call_id}:geocode",  # not the call id: that is the tool's own entry
            thread_id=ctx.thread_id,  # so thread budgets see it
            kind="api",
            provider="acme",
            model="geocode",
            units=1,
        )
        ctx.ledger.record(bill)
    return f"{address}: 18.52N 73.86E"


@registry.tool(once_per_run=True)
def send_summary(text: str) -> str:
    """Email the run's summary (at most once per run)."""
    return "sent"


# --8<-- [end:paid_tool]


async def house_llm(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: every call costs 100k input tokens ($0.20)."""
    return LLMResponse(
        provider="host",
        model="house-llm",
        text="ok",
        message=Message(role="assistant", content="ok"),
        usage=Usage(input_tokens=100_000, output_tokens=0),
    )


# --8<-- [start:executor]
ledger, events = Ledger(prices=PRICES), EventLog()
executor = Executor(
    registry,
    ledger=ledger,
    sink=events,
    hooks=[BudgetHook(0.50)],  # scope="run": each run may spend $0.50
    limits=RunLimits(max_calls=10),  # and make at most 10 tool calls
)
house = CallableModel(house_llm, provider="host", model="house-llm")
llm = executor.model(house)  # hooks, ledger and events around every model call
# --8<-- [end:executor]


async def per_run() -> None:
    # --8<-- [start:per_run]
    ask = LLMRequest(
        messages=(Message(role="user", content="Summarise the returns policy"),), stage="answer"
    )
    async with executor.run_scope() as ctx:
        answered = 0
        try:
            while True:
                await llm.bind(ctx).generate(ask)  # $0.20 each
                answered += 1
        except BudgetExceeded as exc:  # raised before the call that would start over budget
            print(answered, exc.scope, exc.limit_usd, round(exc.spent_usd, 2))
        llm_calls = ledger.summary(ctx.run_id).by_kind["llm"]
        print(round(ledger.total_usd(ctx.run_id), 2), llm_calls.calls, llm_calls.errors)
    # --8<-- [end:per_run]


async def per_thread() -> None:
    # --8<-- [start:per_thread]
    chat = executor.replace(hooks=[BudgetHook(0.25, scope="thread", on_exceed="warn")])
    for run in ("run_a", "run_b"):  # two turns of one conversation
        ctx = chat.context(run_id=run, thread_id="chat-7")
        await chat.run("geocode", {"address": "Paris"}, ctx)  # $0.05 each
        await (
            chat.model(house)
            .bind(ctx)
            .generate(  # $0.20 each
                LLMRequest(messages=(Message(role="user", content="hi"),))
            )
        )
    print(round(ledger.total_usd(thread_id="chat-7"), 2), len(events.of_type("budget.warning")))
    # --8<-- [end:per_thread]


async def call_caps() -> None:
    # --8<-- [start:caps]
    async with executor.run_scope() as ctx:
        print(await executor.run("send_summary", {"text": "done"}, ctx))
        again = await executor.run_call(ToolCall(name="send_summary", arguments={"text": "x"}), ctx)
        print(again.error.kind if again.error else "")  # once_per_run
        geocoded = 0
        try:
            while True:
                await executor.run("geocode", {"address": "Berlin"}, ctx)
                geocoded += 1
        except RunLimitExceeded:
            print("max_calls reached after", geocoded, "geocodes")  # plus the one summary
    # --8<-- [end:caps]


async def main() -> None:
    await per_run()
    await per_thread()
    await call_caps()


if __name__ == "__main__":
    asyncio.run(main())
