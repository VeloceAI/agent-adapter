"""Shared pieces of the durable flow tests: a crash, counting tools, a crashing store."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Annotated, Any

from pydantic import BaseModel

from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.exec.checkpoints import Checkpoint, MemoryCheckpointStore
from agent_adapter.flow import Flow, FlowRunner, Merge

KEY = b"k" * 32


class Crash(BaseException):
    """The process died here: nothing below it runs, nothing is cleaned up."""


class Order(BaseModel):
    order: str = "o-1"
    items: list[str] = []
    log: Annotated[list[str], Merge.append] = []
    note: str = ""


RAN: Counter[str] = Counter()
CRASH_AT: dict[str, int] = {}  # tool name -> crash inside its body on this call number
registry = Registry()


def _maybe_crash(tool: str) -> None:
    RAN[tool] += 1
    if CRASH_AT.get(tool) == RAN[tool]:
        raise Crash(tool)


@registry.tool
async def charge(order: str) -> str:
    """Charge an order (a side effect: not idempotent)."""
    _maybe_crash("charge")
    return f"charged {order}"


@registry.tool
async def ship(order: str) -> str:
    """Ship an order (a side effect: not idempotent)."""
    _maybe_crash("ship")
    return f"shipped {order}"


@registry.tool(idempotent=True)
async def lookup(order: str) -> str:
    """Read a value (safe to repeat)."""
    _maybe_crash("lookup")
    return f"value {order}"


@registry.tool(confirm=True)
async def refund(order: str) -> str:
    """Refund an order (needs approval)."""
    _maybe_crash("refund")
    return f"refunded {order}"


def reset() -> None:
    RAN.clear()
    CRASH_AT.clear()


class YieldingStore(MemoryCheckpointStore):
    """A memory store that yields to the loop before each read and save, like a network one."""

    async def latest(self, run_id: str) -> Any:
        await asyncio.sleep(0)
        return await super().latest(run_id)

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        await asyncio.sleep(0)
        await super().save(run_id, step, checkpoint)


class CrashingStore(MemoryCheckpointStore):
    """A memory store that dies right after (or before) saving a chosen step."""

    def __init__(self) -> None:
        super().__init__()
        self.after: int | None = None
        self.before: int | None = None
        self.saves: list[tuple[int, str]] = []

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        if step == self.before:
            self.before = None
            raise Crash(f"before save {step}")
        await super().save(run_id, step, checkpoint)
        self.saves.append((step, checkpoint.status))
        if step == self.after:
            self.after = None
            raise Crash(f"after save {step}")


def runner(store: Any, **options: Any) -> FlowRunner:
    """A durable runner on the shared registry."""
    executor = options.pop("executor", None) or Executor(registry)
    return FlowRunner(executor, checkpoints=store, checkpoint_key=KEY, **options)


def node_fn(executor: Executor, *tools: str) -> Any:
    """A node function that calls `tools` in order through `executor`."""

    async def node(state: Order, ctx: Any) -> dict[str, Any]:
        done = [await executor.run(tool, {"order": state.order}, ctx) for tool in tools]
        return {"log": done}

    return node


def three_steps() -> Flow[Order]:
    """charge -> ship -> note, one tool call per superstep."""
    from agent_adapter.flow import Call

    return (
        Flow("checkout", state=Order)
        .node("pay", Call("charge", args=lambda s: {"order": s.order}, into="note"))
        .node("send", Call("ship", args=lambda s: {"order": s.order}, into="note"))
        .node("read", Call("lookup", args=lambda s: {"order": s.order}, into="note"))
        .edge("pay", "send")
        .edge("send", "read")
        .end("read")
    )
