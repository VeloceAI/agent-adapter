"""Pipelines: build the executor on the worker, then call tools from tasks, UDFs and actors."""

import functools
from collections.abc import Iterable, Iterator
from typing import Any

from backend_common import Backend, build_backend

from agent_adapter import ToolCall, run_sync

TIMEOUT_S = 300


# --8<-- [start:backend]
@functools.cache
def backend() -> Backend:
    """One per worker process: never pickle an Executor into a task, build it there."""
    return build_backend()


# --8<-- [end:backend]


# --8<-- [start:task]
async def shipping_rates(regions: list[str]) -> dict[str, Any]:
    """An async task body (Airflow 3 `@task`, Prefect 3 `@flow`/`@task`, Dagster async op)."""
    executor = backend().executor
    calls = [ToolCall(name="shipping_rate", arguments={"region": r}) for r in regions]
    async with executor.run_scope() as ctx:  # one run: one budget and one cost line
        results = await executor.run_many(calls, ctx, concurrency=8)
        cost = backend().ledger.total_usd(ctx.run_id)
    rates = {region: r.value if r.ok else None for region, r in zip(regions, results, strict=True)}
    return {"rates": rates, "cost_usd": cost}  # small and JSON-safe: fine for XCom/results


def shipping_rates_sync(regions: list[str]) -> dict[str, Any]:
    """The same for a sync task, a Ray task or a Dagster sync op."""
    return run_sync(shipping_rates(regions), timeout=TIMEOUT_S)


# --8<-- [end:task]


# --8<-- [start:partition]
def rates_for_partition(regions: Iterable[str]) -> Iterator[float | None]:
    """A Spark `mapPartitions` / `pandas_udf` body: one loop per partition, not per row."""
    rows = list(regions)
    rates = shipping_rates_sync(rows)["rates"]
    yield from (rates[region] for region in rows)


# --8<-- [end:partition]
