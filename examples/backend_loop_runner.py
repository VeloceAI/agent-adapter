"""A warm sync process (Lambda, Dramatiq, gRPC sync servicer): one loop, clients kept warm."""

import asyncio

from agent_adapter import Executor, Registry
from agent_adapter.runtime import LoopLocal, LoopRunner

registry = Registry()
created: list["Client"] = []


class Client:
    """Stands in for an httpx / Redis client bound to the loop that created it."""

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
async def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    client = clients.get()  # the same client on every call: one loop for the process
    assert client.loop is asyncio.get_running_loop()
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


# --8<-- [start:runner]
executor = Executor(registry)
runner = LoopRunner()  # module level: the loop starts on the first run(), not at import


def handler(event: dict[str, str], context: object = None) -> dict[str, object]:
    """A sync entry point called once per request or message."""
    rate = runner.run(executor.run("shipping_rate", {"region": event["region"]}), timeout=30)
    return {"status": "ok", "rate": rate}


# --8<-- [end:runner]


def main() -> None:
    print(handler({"region": "us"}))
    print(handler({"region": "eu"}))
    print(len(created), "client")  # one client for both calls
    # --8<-- [start:close]
    runner.close()  # also runs at interpreter exit; closes LoopLocal values first
    # --8<-- [end:close]
    print([client.closed for client in created])


if __name__ == "__main__":
    main()
