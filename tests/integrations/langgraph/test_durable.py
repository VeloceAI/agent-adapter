"""Claims and sealed task state: one use per task run, and a sealed state only its task opens."""

from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalAlreadyDecided, FlowError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow
from agent_adapter.integrations.langgraph import compile_flow, run_compiled
from agent_adapter.integrations.langgraph._durable import (
    Seal,
    Surfaced,
    TaskState,
    open_sealed,
    seal,
)
from agent_adapter.integrations.langgraph._spent import ResumeClaims, claim, task_scope
from tests.integrations.langgraph.support import config

pytestmark = pytest.mark.contract

SEAL = Seal(b"k" * 16, "graph", "run", "node:task", {})


def test_claims_are_held_by_one_live_execution_and_spent_when_it_returns():
    now = [0.0]
    claims = ResumeClaims(max_entries=2, ttl=10.0, clock=lambda: now[0])
    from agent_adapter.integrations.langgraph._spent import Owner

    first, second = Owner("r", "n"), Owner("r", "n")
    claims.take(("r", "n", "a", "1"), first)
    claims.take(("r", "n", "a", "1"), first)  # the holder may take it again
    with pytest.raises(ApprovalAlreadyDecided):
        claims.take(("r", "n", "a", "1"), second)
    claims.release(first)
    claims.take(("r", "n", "a", "1"), second)  # released by a pause: the re-run takes it
    claims.spend(second)
    with pytest.raises(ApprovalAlreadyDecided):
        claims.take(("r", "n", "a", "1"), first)
    for name in ("b", "c"):  # the oldest spent keys go first past max_entries
        owner = Owner("r", "n")
        claims.take(("r", "n", name, "1"), owner)
        claims.spend(owner)
    claims.take(("r", "n", "a", "1"), Owner("r", "n"))
    now[0] = 100.0  # and every spent key expires
    owner = Owner("r", "n")
    claims.take(("r", "n", "b", "1"), owner)
    claims.clear()


def test_a_claim_needs_its_task_scope_and_occurrences_count():
    from agent_adapter.core.errors import ToolUsageError

    with pytest.raises(ToolUsageError, match="one_use"):
        claim("r-out", "ns", "x")
    with pytest.raises(ToolUsageError, match="one_use"), task_scope("r-out", "other-ns"):
        claim("r-out", "ns", "x")
    with task_scope("r-in", "ns"):
        claim("r-in", "ns", "y")
        claim("r-in", "ns", "y")  # a second occurrence in the same run is its own key
    with pytest.raises(ApprovalAlreadyDecided), task_scope("r-in", "ns"):
        claim("r-in", "ns", "y")


def _state() -> TaskState:
    request = ApprovalRequest(approval_id="a", run_id="run", call_id="c", tool="t")
    return TaskState(surfaced=[Surfaced(request, True)], journal={}, items={0: ("ok", 1)})


def test_a_sealed_state_opens_only_for_its_task_with_its_key():
    sealed = seal(SEAL, _state())
    opened = open_sealed(SEAL, sealed, "f")
    assert opened.items == {0: ("ok", 1)} and opened.surfaced[0].request.tool == "t"
    for other in (
        Seal(b"x" * 16, "graph", "run", "node:task", {}),
        Seal(SEAL.key, "graph", "run", "node:other", {}),
        Seal(SEAL.key, "graph", "other-run", "node:task", {}),
        Seal(SEAL.key, "other-graph", "run", "node:task", {}),
    ):
        with pytest.raises(FlowError, match="resume in the pausing process"):
            open_sealed(other, sealed, "f")
    for broken in ({}, "x", {"payload": {}, "mac": 1}):
        with pytest.raises(FlowError, match="resume in the pausing process"):
            open_sealed(SEAL, broken, "f")
    tampered = {"payload": {**sealed["payload"], "items": {"$nope": 1}}, "mac": ""}
    tampered["mac"] = seal_mac(tampered["payload"])
    with pytest.raises(FlowError, match="cannot be loaded"):
        open_sealed(SEAL, tampered, "f")


def seal_mac(payload: dict[str, Any]) -> str:
    from agent_adapter.flow._seal import mac

    return mac(SEAL.key, payload)


class Weird:
    """A value no canonical encoding knows."""


registry = Registry()


@registry.tool
async def odd() -> Any:
    """Return a value that cannot be sealed."""
    return Weird()


@registry.tool(confirm=True)
async def send(text: str) -> str:
    """Send (needs approval)."""
    return text


class Note(BaseModel):
    text: str = ""


async def test_a_pause_whose_finished_calls_cannot_be_sealed_fails_closed():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))

    async def code(state: Note, ctx: ToolContext) -> dict[str, Any]:
        await executor.run("odd", {}, ctx)
        return {"text": await executor.run("send", {"text": "x"}, ctx)}

    flow = Flow("odd_flow", state=Note).node("go", code).end("go")
    app = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver())
    with pytest.raises(FlowError, match="state-not-sealable"):
        await run_compiled(app, {}, config())
    [record] = await store.list()
    assert record.consumed and record.decision.reason == "state-not-sealable"
