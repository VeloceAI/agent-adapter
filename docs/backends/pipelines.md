# Data pipelines

:material-check-circle:{ .status-ok } Available · `pip install agent-adapter` (no extra)

Airflow, Prefect, Dagster, Ray and Spark run your code on workers, often in other
processes. Build the executor on the worker, call tools from the task, return a small JSON
result.

| File | Does |
|---|---|
| `examples/backend_pipeline.py` | a per-process `backend()`, an async task, a sync task, a Spark partition body |
| `tests/docs/test_backend_handler_examples.py` | runs them offline |

Data enters as task arguments, goes through `run_many` to the tools as one run, and
leaves as the task's return value (rates and the run's cost).

## One executor per worker process

```python
--8<-- "examples/backend_pipeline.py:backend"
```

An `Executor` or `Registry` holds locks, so it cannot be pickled. Never pass one into a
task, a Ray remote function or a Spark closure: pass names and arguments, and call
`backend()` inside.

## Task bodies

```python
--8<-- "examples/backend_pipeline.py:task"
```

| Tool | Write | Runs |
|---|---|---|
| Airflow 3 | `@task async def ...` (async `PythonOperator`) | `await shipping_rates(...)` on the worker's loop |
| Airflow, sync `@task` | `def ...` | `shipping_rates_sync(...)` |
| Prefect 3 | `@flow` / `@task` on `async def` | `await shipping_rates(...)` |
| Dagster | async `@op` / `@asset` | `await shipping_rates(...)`; sync ops `shipping_rates_sync` |
| Ray | `@ray.remote def ...` | `shipping_rates_sync(...)`; an actor can hold `backend()` as its tool host |
| Spark | `mapPartitions` / `pandas_udf` | `rates_for_partition` below |

## Spark partitions

```python
--8<-- "examples/backend_pipeline.py:partition"
```

One `run_sync` per partition, not per row: each `run_sync` builds a loop. Each Python
worker builds its own `backend()` once.

## Rules

| Rule | Why |
|---|---|
| build the executor in the worker (`functools.cache`) | executors do not pickle; forked workers must not share loop-bound clients |
| one run per task | `BudgetHook` caps it; the ledger sums its cost |
| return JSON-safe values | XCom, Prefect results and Dagster outputs serialise them |
| set `concurrency=` on `run_many` | a big batch must not open hundreds of connections |
| approvals in a pipeline | use `AutoDeny`, or park with `DeferredApproval` and a shared store, then resume in a later run |
| retries | the task framework retries the task; the tool's `retries=` retries one call |

See also: [Queues](queues.md), [Scripts and batch jobs](scripts.md),
[Runtime](../concepts/runtime.md).
