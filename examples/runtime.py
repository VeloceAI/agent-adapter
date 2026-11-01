"""Call async tools from sync code with run_sync; keep one client per event loop."""

import asyncio

from agent_adapter import Executor, Registry, ToolContext, run_sync
from agent_adapter.runtime import LoopLocal, current_context, new_id, use_context

registry = Registry()
created: list["Client"] = []


# --8<-- [start:loop_local]
class Client:
    """Stands in for an httpx / redis client bound to the loop that created it."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.closed = False
        created.append(self)

    async def aclose(self) -> None:
        self.closed = True


async def close_client(client: Client) -> None:
    await client.aclose()


clients = LoopLocal(Client, close=close_client)


@registry.tool
async def fetch(path: str) -> str:
    """Fetch a page."""
    client = clients.get()  # this loop's client, created on first use
    assert client.loop is asyncio.get_running_loop()
    return f"GET {path}"


# --8<-- [end:loop_local]

executor = Executor(registry)  # hooks, approval, budget and ledger apply to each call


def flask_view() -> str:
    # --8<-- [start:run_sync]
    result: str = run_sync(executor.run("fetch", {"path": "/returns"}), timeout=30)
    # --8<-- [end:run_sync]
    return result


async def with_context() -> str:
    # --8<-- [start:current]
    with use_context(ToolContext(run_id=new_id("run_"))):
        ctx = current_context()  # what a nested helper sees
        return ctx.run_id[:4] if ctx else "-"
    # --8<-- [end:current]


def main() -> None:
    print(flask_view())
    print(flask_view())  # a second run_sync call: a new loop, so a new client
    print([client.closed for client in created])  # each closed when its loop ended
    print(asyncio.run(with_context()))


if __name__ == "__main__":
    main()
