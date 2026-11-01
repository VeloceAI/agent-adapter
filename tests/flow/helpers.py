"""Shared state models, tools and runners for the flow tests."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from pydantic import BaseModel

from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import FlowRunner, Merge, keyed
from tests.fakes.clock import FakeClock


class Hit(BaseModel):
    source_id: str
    text: str = ""


class State(BaseModel):
    query: str = ""
    items: list[Any] = []
    hits: Annotated[list[Hit], keyed(by="source_id")] = []
    log: Annotated[list[str], Merge.append] = []
    total: Annotated[int, Merge.sum] = 0
    n: int = 0
    note: str = ""


registry = Registry()
CALLS: list[str] = []


@registry.tool(parallel_safe=True)
async def search(query: str) -> list[dict]:
    """Search the index."""
    CALLS.append(query)
    return [{"source_id": f"src-{query}", "text": query}, {"source_id": "shared", "text": "s"}]


@registry.tool(parallel_safe=True)
async def bump(n: int) -> int:
    """Add one."""
    return n + 1


@registry.tool(parallel_safe=True)
async def boom(reason: str = "no") -> str:
    """Always fails."""
    raise RuntimeError(reason)


@registry.tool(parallel_safe=True)
async def slow(seconds: float) -> str:
    """Sleep a while."""
    await asyncio.sleep(seconds)
    return "late"


@registry.tool(parallel_safe=True, confirm=True)
async def publish(text: str) -> str:
    """Publish (needs approval)."""
    return f"published {text}"


PAID: list[int] = []
NUKED: list[str] = []


@registry.tool(parallel_safe=True, confirm=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    PAID.append(amount)
    return f"paid {amount}"


@registry.tool(parallel_safe=True, confirm=True)
async def nuke() -> str:
    """Delete everything (needs approval)."""
    NUKED.append("nuked")
    return "nuked"


def runner(**options: Any) -> tuple[FlowRunner, Executor, FakeClock]:
    """A runner over the shared registry with a fake clock; executor options go through."""
    clock = FakeClock()
    runner_options = {
        key: options.pop(key)
        for key in ("hooks", "max_steps", "concurrency", "model")
        if key in options
    }
    if "flow_approval" in options:
        runner_options["approval"] = options.pop("flow_approval")
    if "flow_sink" in options:
        runner_options["sink"] = options.pop("flow_sink")
    executor = Executor(registry, clock=clock, sleep=clock.sleep, rng=lambda: 0.0, **options)
    return FlowRunner(executor, **runner_options), executor, clock


async def resume_until_done(
    run: FlowRunner, flow: Any, result: Any, decision: ApprovalDecision | None = None
) -> tuple[Any, int]:
    """Resume `result` with `decision` (approve) until it stops pausing; `(final, resumes)`."""
    decision = decision or ApprovalDecision(approved=True)
    resumes = 0
    while result.status == "paused":
        result = await run.resume(flow, result.checkpoint, decision)
        resumes += 1
    return result, resumes
