"""Cookbook: cache model answers across workers with Redis (fakeredis here, offline)."""

import asyncio

from fakeredis import FakeAsyncRedis, FakeServer

from agent_adapter import CallableModel, Executor, LLMRequest, Message, Registry
from agent_adapter.cache import RedisCache

server = FakeServer()  # one Redis that both "workers" reach
paid_calls = {"n": 0}


async def answer(request: LLMRequest) -> str:
    """Stands in for a paid model call."""
    paid_calls["n"] += 1
    return "Unused items can be returned within 30 days."


# --8<-- [start:workers]
def worker() -> Executor:
    """One process's executor; in real code RedisCache("redis://cache:6379/0", prefix=...)."""
    cache = RedisCache(client_factory=lambda: FakeAsyncRedis(server=server), prefix="llm:")
    return Executor(Registry(), cache=cache)


model = CallableModel(answer, provider="host", model="house")
web, batch = worker(), worker()
# --8<-- [end:workers]


async def main() -> None:
    # --8<-- [start:ask]
    ask = LLMRequest(
        messages=(Message(role="user", content="What is the returns policy?"),), stage="faq"
    )
    first = await web.model(model, cache_ttl=86_400).generate(ask)  # a miss: pays, stores
    again = await batch.model(model, cache_ttl=86_400).generate(ask)  # another worker: a hit
    other_stage = ask.model_copy(update={"stage": "triage"})  # stage is not part of the key
    third = await batch.model(model, cache_ttl=86_400).generate(other_stage)
    fresh = ask.model_copy(update={"response_cache": False})  # opt out for this request
    fourth = await web.model(model, cache_ttl=86_400).generate(fresh)
    print([r.cached for r in (first, again, third, fourth)], paid_calls["n"])
    # --8<-- [end:ask]


if __name__ == "__main__":
    asyncio.run(main())
