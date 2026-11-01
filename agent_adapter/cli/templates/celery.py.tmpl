"""Starter Celery tasks: one task runs one tool. Demo: python tasks.py (no worker needed)"""

import functools
import os
from typing import Any

from celery import Celery

from agent_adapter import Executor, Registry, run_sync

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@functools.cache
def executor() -> Executor:
    """One per worker process, built on first use (after the prefork fork)."""
    return Executor(registry)  # add hooks=, approval=, limits= here


app = Celery("tools", broker=os.environ.get("CELERY_BROKER_URL", "memory://"))


def run_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Run one tool call; an error raises, so Celery records the failure."""
    return run_sync(executor().run(name, arguments), timeout=60)


task = app.task(name="tools.run")(run_tool)  # call (not decorate): Celery is untyped

if __name__ == "__main__":
    print(task.apply(args=["shipping_rate", {"region": "us"}]).get())  # here, no worker
