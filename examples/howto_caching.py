"""Caching: tool results per run, thread or globally; LLM responses; what a hit costs."""

import asyncio

from agent_adapter import (
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    MemoryCache,
    Message,
    Prices,
    Registry,
    ToolCall,
)

registry = Registry()
runs = {"rates": 0, "profile": 0, "llm": 0}


# --8<-- [start:tools]
@registry.tool(cache_ttl=3600, cache_scope="global")  # same for every user: share it
def exchange_rate(currency: str) -> float:
    """Today's EUR rate for a currency."""
    runs["rates"] += 1
    return {"USD": 0.92, "GBP": 1.17}[currency]


@registry.tool(cache_ttl=600, cache_scope="thread")  # per conversation, never across users
def user_profile(field: str) -> str:
    """One field of the signed-in user's profile."""
    runs["profile"] += 1
    return f"{field}: value"


# --8<-- [end:tools]

# --8<-- [start:executor]
cache = MemoryCache(max_entries=10_000)  # RedisCache(url) to share across processes
ledger = Ledger(
    prices=Prices.load(
        {"version": "demo", "models": {"house": {"provider": "host", "input": 1.0, "output": 4.0}}}
    )
)
executor = Executor(registry, cache=cache, ledger=ledger)
# --8<-- [end:executor]


async def tool_cache() -> None:
    # --8<-- [start:tool_cache]
    call = ToolCall(name="exchange_rate", arguments={"currency": "USD"})
    first = await executor.run_call(call)  # a fresh run per call
    second = await executor.run_call(call)  # another run: the global entry still hits
    print(first.cached, second.cached, second.attempts, runs["rates"])

    for thread in ("chat-1", "chat-1", "chat-2"):
        ctx = executor.context(thread_id=thread)
        await executor.run("user_profile", {"field": "email"}, ctx)
    print(runs["profile"])  # chat-1 hit once; chat-2 has its own entry
    print(ledger.summary().tool_cache_hit_rate)
    # --8<-- [end:tool_cache]


async def llm_cache() -> None:
    # --8<-- [start:llm_cache]
    async def answer(request: LLMRequest) -> str:  # stands in for a paid model call
        runs["llm"] += 1
        return "Unused items can be returned within 30 days."

    llm = executor.model(CallableModel(answer, provider="host", model="house"), cache_ttl=86_400)
    ask = LLMRequest(messages=(Message(role="user", content="What is the returns policy?"),))
    first, second = await llm.generate(ask), await llm.generate(ask)
    fresh = await llm.generate(ask.model_copy(update={"response_cache": False}))  # opt out
    print(first.cached, second.cached, fresh.cached, runs["llm"])
    print(round(ledger.summary().response_cache_hit_rate, 2))  # a hit is recorded at $0
    # --8<-- [end:llm_cache]


async def main() -> None:
    await tool_cache()
    await llm_cache()


if __name__ == "__main__":
    asyncio.run(main())
