"""Call async tools from sync code: a WSGI view, a worker task, a script, a running loop."""

import asyncio
import threading

from agent_adapter import Executor, Registry, ToolCall, run_sync
from agent_adapter.runtime import LoopLocal

registry = Registry()
opened: list[str] = []  # the thread each client was made in


# --8<-- [start:client]
class RatesClient:
    """Stands in for an httpx / redis client: bound to the loop that made it."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        opened.append(threading.current_thread().name)

    async def aclose(self) -> None:
        """Close the connection pool."""


async def close(client: RatesClient) -> None:
    await client.aclose()


rates_client = LoopLocal(RatesClient, close=close)  # one client per loop, closed with it


@registry.tool(parallel_safe=True, timeout=10)
async def shipping_rate(item: str) -> float:
    """Shipping rate for an item, in percent of the order value."""
    client = rates_client.get()
    assert client.loop is asyncio.get_running_loop()
    return 18.0 if item == "laptop" else 5.0


# --8<-- [end:client]

# --8<-- [start:executor]
executor = Executor(registry)  # one per process; safe to share across run_sync calls
# --8<-- [end:executor]


# --8<-- [start:view]
def rate_view(item: str) -> dict[str, object]:
    """A Flask / Django (WSGI) view: plain sync code, no loop of its own."""
    rate = run_sync(executor.run("shipping_rate", {"item": item}), timeout=30)
    return {"item": item, "rate": rate}


# --8<-- [end:view]


# --8<-- [start:task]
def nightly_rates(items: list[str]) -> list[float]:
    """A Celery / RQ task or a cron script: one loop for the whole batch."""

    async def batch() -> list[float]:
        calls = [ToolCall(name="shipping_rate", arguments={"item": i}) for i in items]
        return [r.unwrap() for r in await executor.run_many(calls)]

    return run_sync(batch, timeout=300)  # a callable: the coroutine is made inside the loop


# --8<-- [end:task]


async def notebook_cell() -> dict[str, object]:
    # --8<-- [start:running_loop]
    # Sync code called from inside a running loop (a notebook, an async callback):
    # run_sync runs a fresh loop in a worker thread and blocks this thread until done.
    return rate_view("book")
    # --8<-- [end:running_loop]


def main() -> None:
    print(rate_view("laptop"))
    print(nightly_rates(["laptop", "book", "laptop"]))
    print(asyncio.run(notebook_cell()))
    print(len(opened))  # one client per run_sync loop, each closed with its loop
    print(opened[0] == "MainThread", opened[2].startswith("agent_adapter-run_sync"))


if __name__ == "__main__":
    main()
