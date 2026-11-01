"""Turn on run recording in each kind of app: every set-up is one `record(...)` line."""

import asyncio
import functools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import agent_adapter as aa
from agent_adapter import Executor, Registry
from agent_adapter.viewer import record

registry = Registry()


@registry.tool
def find_hotel(city: str) -> str:
    """Find a hotel in a city."""
    return f"Hotel Central, {city}"


def plain_script() -> str:
    # --8<-- [start:script]
    record()  # the default executor: aa.call(...) and integrations given no executor=
    aa.configure(registry)  # before or after: record() follows the default executor
    # --8<-- [end:script]
    return str(aa.call(find_hotel, city="Lisbon"))


# --8<-- [start:fastapi]
@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan: one executor per worker process, recorded."""
    executor = Executor(registry)
    record(executor, source="fastapi")  # each worker writes its own files
    app.state.executor = executor
    yield


# --8<-- [end:fastapi]


# --8<-- [start:celery]
@functools.cache
def worker_executor() -> Executor:
    """One per Celery worker process, built after the prefork fork."""
    executor = Executor(registry)
    record(executor, source="celery")
    return executor


# --8<-- [end:celery]


def framework_executor() -> Executor:
    # --8<-- [start:framework]
    executor = Executor(registry)
    record(executor, source="langgraph")  # name your framework or backend
    # then hand it over: attach("langgraph", registry, executor=executor),
    # tool_node(registry, executor=executor), compile_flow(flow, executor=executor), ...
    # --8<-- [end:framework]
    return executor


async def fastapi_demo() -> str:
    app = SimpleNamespace(state=SimpleNamespace())
    async with lifespan(app):
        return str(await app.state.executor.run(find_hotel, {"city": "Porto"}))


def main() -> None:
    print(plain_script())
    print(asyncio.run(fastapi_demo()))
    print(worker_executor().run_sync(find_hotel, {"city": "Faro"}))
    print(framework_executor().run_sync(find_hotel, {"city": "Braga"}))


if __name__ == "__main__":
    main()
