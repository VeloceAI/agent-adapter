"""A worker process for the kill test: it dies (`os._exit`) inside a tool body, mid-run."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import SQLiteCheckpointStore
from agent_adapter.flow import Call, Flow, FlowRunner

KEY = b"kill-test-key-of-32-bytes-length"
registry = Registry()
LOG = Path(os.environ.get("AA_KILL_LOG", "calls.log"))


class Order(BaseModel):
    order: str = "o-1"
    note: str = ""


def _ran(tool: str) -> int:
    with LOG.open("a", encoding="utf-8") as out:
        out.write(tool + "\n")
    return LOG.read_text(encoding="utf-8").split().count(tool)


@registry.tool
def charge(order: str) -> str:
    """Charge an order."""
    _ran("charge")
    return f"charged {order}"


@registry.tool(idempotent=True)
def ship(order: str) -> str:
    """Ship an order; the first worker is killed here."""
    if _ran("ship") == 1 and os.environ.get("AA_KILL") == "1":
        os._exit(3)  # no finally, no cleanup: a real kill
    return f"shipped {order}"


flow = (
    Flow("killed", state=Order)
    .node("pay", Call("charge", args=lambda s: {"order": s.order}, into="note"))
    .node("send", Call("ship", args=lambda s: {"order": s.order}, into="note"))
    .edge("pay", "send")
    .end("send")
)


def runner(path: str) -> FlowRunner:
    return FlowRunner(
        Executor(registry), checkpoints=SQLiteCheckpointStore(path), checkpoint_key=KEY
    )


async def _main(path: str) -> None:
    r = runner(path)
    await r.run(flow, {}, r.executor.context(run_id="run-1"))


def main(path: str) -> None:
    """Run the flow once; with `AA_KILL=1` the process dies inside `ship`."""
    asyncio.run(_main(path))
