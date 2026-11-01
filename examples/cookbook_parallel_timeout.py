"""Cookbook: ask several retrievers at once; a per-tool timeout and a deadline for the batch."""

import asyncio
import contextlib

from agent_adapter import Executor, Registry, ToolCall, ToolResult

registry = Registry()


# --8<-- [start:tools]
@registry.tool(kind="search", parallel_safe=True, timeout=0.5)
async def articles(query: str) -> list[str]:
    """Search the help-center articles."""
    await asyncio.sleep(0.01)
    return [f"{query}: article"]


@registry.tool(kind="search", parallel_safe=True, timeout=0.5)
async def forum(query: str) -> list[str]:
    """Search community forum posts."""
    await asyncio.sleep(0.02)
    return [f"{query}: post"]


@registry.tool(kind="search", parallel_safe=True, timeout=0.3)  # this backend hangs
async def archive(query: str) -> list[str]:
    """Search the old archive."""
    await asyncio.sleep(30)
    return []


# --8<-- [end:tools]

executor = Executor(registry)


def calls(query: str, *names: str) -> list[ToolCall]:
    return [ToolCall(name=name, arguments={"query": query}) for name in names]


async def wait_for_all(query: str) -> list[ToolResult]:
    # --8<-- [start:all]
    results = await executor.run_many(calls(query, "articles", "forum", "archive"))
    return results  # input order; the hung archive is a tool-timeout, the rest are ok
    # --8<-- [end:all]


async def until_deadline(query: str, seconds: float) -> list[str]:
    # --8<-- [start:deadline]
    hits: list[str] = []
    stream = executor.as_completed(calls(query, "archive", "forum", "articles"))
    with contextlib.suppress(TimeoutError):  # keep what arrived in time
        async with asyncio.timeout(seconds), contextlib.aclosing(stream):
            async for result in stream:  # completion order
                if result.ok:
                    hits += result.value
    return hits  # leaving the block cancelled whatever was still running
    # --8<-- [end:deadline]


async def main() -> None:
    results = await wait_for_all("returns")
    print([(r.name, r.status, r.error.kind if r.error else "") for r in results])
    print(sorted(await until_deadline("returns", seconds=0.15)))


if __name__ == "__main__":
    asyncio.run(main())
