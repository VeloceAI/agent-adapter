"""Parallel retrieval: fan out to several retrievers, merge, stop early, run non-tool work."""

import asyncio
import contextlib

from agent_adapter import Executor, Registry, RunLimits, Step, ToolCall

registry = Registry()
INDEX = {"returns": ["returns policy", "returns label"], "shipping": ["shipping cost"]}


# --8<-- [start:tools]
@registry.tool(kind="search", parallel_safe=True, timeout=5, retries=1)
async def keyword_search(query: str) -> list[str]:
    """Keyword search over the help-center articles."""
    await asyncio.sleep(0.01)
    return INDEX.get(query, [])


@registry.tool(kind="search", parallel_safe=True, timeout=5)
async def vector_search(query: str) -> list[str]:
    """Semantic search over the help-center articles."""
    await asyncio.sleep(0.02)
    return [f"{query} policy", f"{query} similar"]


@registry.tool(parallel_safe=True)
def rerank(hits: list[str], top_k: int = 2) -> list[str]:
    """Order hits by relevance and keep the best."""
    return sorted(set(hits), key=len)[:top_k]


@registry.tool(kind="search", parallel_safe=True)
async def archive_search(query: str) -> list[str]:
    """Slow search over the archive."""
    await asyncio.sleep(30)  # cancelled when the caller stops early
    return []


# --8<-- [end:tools]

# --8<-- [start:executor]
# At most two concurrent "search" calls per run, whatever the batch size.
executor = Executor(registry, limits=RunLimits(per_kind={"search": 2}))
# --8<-- [end:executor]


async def fan_out_fan_in(query: str) -> list[str]:
    # --8<-- [start:fan_out]
    calls = [
        ToolCall(name="keyword_search", arguments={"query": query}),
        ToolCall(name="vector_search", arguments={"query": query}),
        ToolCall(name="keyword_search", arguments={"query": 7}),  # invalid: not a string
    ]
    results = await executor.run_many(calls, concurrency=8, on_error="collect")
    print([r.status for r in results])  # input order; one failure does not stop the rest

    merged: dict[str, None] = {}  # fan in: dedupe, keep first-seen order
    for result in results:
        if result.ok:
            merged.update(dict.fromkeys(result.value))
    return list(merged)
    # --8<-- [end:fan_out]


async def first_answer(query: str) -> str:
    # --8<-- [start:as_completed]
    calls = [
        ToolCall(name="archive_search", arguments={"query": query}),
        ToolCall(name="keyword_search", arguments={"query": query}),
    ]
    stream = executor.as_completed(calls)
    async with contextlib.aclosing(stream):
        async for result in stream:
            if result.ok and result.value:
                return f"{result.name}: {result.value[0]}"
    return "nothing found"
    # --8<-- [end:as_completed]


async def retrieve_then_rank(query: str) -> list[str]:
    # --8<-- [start:plan]
    plan = [
        Step(id="kw", tool="keyword_search", args={"query": query}),
        Step(id="vec", tool="vector_search", args={"query": query}),
        Step(
            id="best",
            tool="rerank",  # second wave: runs once both retrievers finished
            args=lambda deps: {"hits": deps["kw"].value + deps["vec"].value},
            depends_on=["kw", "vec"],
        ),
    ]
    by_step = await executor.run_plan(plan)
    return list(by_step["best"].value)
    # --8<-- [end:plan]


async def embed_all(texts: list[str]) -> list[int]:
    # --8<-- [start:parallel]
    async def embed(text: str) -> int:  # stands in for an embedding or LLM call
        await asyncio.sleep(0.01)
        return len(text)

    return await executor.parallel(*(lambda t=t: embed(t) for t in texts), timeout=10)
    # --8<-- [end:parallel]


async def main() -> None:
    print(await fan_out_fan_in("returns"))
    print(await first_answer("returns"))
    print(await retrieve_then_rank("returns"))
    print(await embed_all(["returns policy", "shipping cost"]))


if __name__ == "__main__":
    asyncio.run(main())
