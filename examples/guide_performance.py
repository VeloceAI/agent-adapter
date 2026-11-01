"""Performance guide: import cost, concurrency caps, cache hits and per-call overhead."""

import asyncio
import subprocess
import sys
import time

from agent_adapter import Executor, MemoryCache, Registry, RunLimits, ToolCall, run_sync
from agent_adapter.runtime import LoopLocal, LoopRunner

registry = Registry()
in_flight = {"now": 0, "peak": 0}


# --8<-- [start:import]
def import_cost() -> tuple[bool, list[str]]:
    """In a fresh interpreter: is pydantic loaded, and which adapter modules are?"""
    code = (
        "import sys, agent_adapter\n"
        "print('pydantic' in sys.modules)\n"
        "print(sorted(m for m in sys.modules if m.startswith('agent_adapter')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return out[0] == "True", eval(out[1])  # a list literal printed by the child


# --8<-- [end:import]


# --8<-- [start:tools]
@registry.tool(kind="search", parallel_safe=True)
async def search(query: str) -> str:
    """Search an index that allows two calls at a time."""
    in_flight["now"] += 1
    in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
    await asyncio.sleep(0.01)
    in_flight["now"] -= 1
    return query


@registry.tool(cache_ttl=300, cache_scope="global")
def rates() -> dict[str, float]:
    """Exchange rates, the same for everyone."""
    return {"EUR": 0.92}


@registry.tool
def echo(text: str) -> str:
    """Return the text."""
    return text


# --8<-- [end:tools]

# --8<-- [start:executor]
# Build once per process, at import time; share it between requests.
executor = Executor(registry, cache=MemoryCache(), limits=RunLimits(per_kind={"search": 2}))
# --8<-- [end:executor]


async def caps_and_cache() -> None:
    # --8<-- [start:caps]
    calls = [ToolCall(name="search", arguments={"query": str(i)}) for i in range(8)]
    await executor.run_many(calls, concurrency=8)  # asks for 8 at once
    print("peak search calls:", in_flight["peak"])  # the per-kind cap holds it at 2
    first = await executor.run_call(ToolCall(name="rates"))
    hit = await executor.run_call(ToolCall(name="rates"))
    print("cache:", first.cached, hit.cached, hit.attempts)  # a hit runs nothing
    # --8<-- [end:caps]


# --8<-- [start:measure]
async def overhead_us(n: int = 2_000) -> float:
    """Average microseconds per `run_call` of a trivial sync tool, in one run."""
    ctx = executor.context()
    call = ToolCall(name="echo", arguments={"text": "x"})
    await executor.run_call(call, ctx)  # warm up
    start = time.perf_counter()
    for _ in range(n):
        await executor.run_call(call, ctx)
    return (time.perf_counter() - start) / n * 1e6


# --8<-- [end:measure]


# --8<-- [start:sync]
made: list[int] = []


def new_client() -> int:
    """Stands in for an httpx or Redis client: expensive to build (TLS, pools)."""
    made.append(len(made) + 1)
    return made[-1]


client: LoopLocal[int] = LoopLocal(new_client)  # one client per event loop


async def which_client() -> int:
    return client.get()


def sync_callers() -> None:
    per_call = [run_sync(which_client, timeout=5) for _ in range(3)]  # a loop per call
    with LoopRunner() as runner:  # in real code: one module-level runner per process
        warm = [runner.run(which_client, timeout=5) for _ in range(3)]  # one loop, kept
    print("run_sync clients:", per_call, "| LoopRunner clients:", warm)


# --8<-- [end:sync]


async def main() -> None:
    loaded, modules = import_cost()
    print("pydantic loaded:", loaded, "| adapter modules:", modules)
    await caps_and_cache()
    if "--measure" in sys.argv:
        print(f"{await overhead_us():.0f} us per call")


if __name__ == "__main__":
    asyncio.run(main())
    sync_callers()  # sync code: no event loop running here
