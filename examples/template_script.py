"""Starter script: one tool, one executor, a sync call. Run: python main.py"""

import logging

from agent_adapter import Executor, LoggingHook, Registry, run_sync

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


# One per process. Add approval=, limits= and more hooks here; confirm tools fail closed.
executor = Executor(registry, hooks=[LoggingHook()])

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(run_sync(executor.run("shipping_rate", {"region": "us"})))
