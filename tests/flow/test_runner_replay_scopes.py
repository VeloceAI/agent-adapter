"""Replay reaches every scope a node opens: Map(fn=) items, nested flows, non-JSON arguments,
and a node's own executor with its own approval store."""

import datetime as dt

import pytest

from agent_adapter import Registry
from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow, FlowRunner, Map
from tests.fakes.sinks import RecordingLedger
from tests.flow.helpers import State, resume_until_done

APPROVE = ApprovalDecision(approved=True)
registry = Registry()
RAN: list[str] = []


@registry.tool(confirm=True, parallel_safe=True)
async def pay(amount: int) -> str:
    """Pay an amount (needs approval)."""
    RAN.append(f"pay{amount}")
    return f"paid {amount}"


@registry.tool(confirm=True)
async def xnuke() -> str:
    """Delete everything (needs approval)."""
    RAN.append("xnuke")
    return "nuked"


@registry.tool(once_per_run=True, parallel_safe=True)
async def read(key: str) -> str:
    """Read a value once per run."""
    RAN.append(f"read {key}")
    return f"value of {key}"


@registry.tool(parallel_safe=True)
async def fetch(key: str) -> str:
    """Read a value."""
    RAN.append(f"fetch {key}")
    return f"value of {key}"


@registry.tool
async def stamp(when: dt.datetime) -> str:
    """Record a time (a non-JSON argument)."""
    RAN.append(f"stamp {when.year}")
    return when.isoformat()


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()


async def pending_records(*stores):
    return [r for store in stores for r in await store.list(status="pending")]


async def test_map_fn_items_replay_their_own_calls_across_resumes():
    ledger = RecordingLedger()
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()), ledger=ledger)
    run = FlowRunner(executor)

    async def per_item(item, state, ctx):
        if item == 1:
            await executor.run("read", {"key": "setup"}, ctx)  # once_per_run: never re-counted
        value = await executor.run("fetch", {"key": f"m{item}"}, ctx)
        return f"{value}: {await executor.run('pay', {'amount': item}, ctx)}"

    flow = (
        Flow("map_replay", state=State).node("go", Map("items", fn=per_item, into="log")).end("go")
    )
    paused = await run.run(flow, {"items": [1, 2]})
    run_id = paused.pending.run_id
    done, resumes = await resume_until_done(run, flow, paused)
    assert resumes == 2 and done.status == "done"
    assert done.state.log == ["value of m1: paid 1", "value of m2: paid 2"]
    assert sorted(RAN) == ["fetch m1", "fetch m2", "pay1", "pay2", "read setup"]  # each once
    tool_entries = [e for e in ledger.entries(run_id) if e.kind == "tool"]
    assert len(tool_entries) == len(RAN) == 5


async def test_a_nested_flow_replays_its_calls_when_the_outer_node_resumes():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))

    async def inner_read(state, ctx):
        return {"note": await executor.run("read", {"key": "inner"}, ctx)}

    inner = Flow("inner", state=State).node("go", inner_read).end("go")

    async def outer(state, ctx):
        inner_done = await FlowRunner(executor).run(inner, {}, ctx)
        return {
            "note": f"{inner_done.state.note} | {await executor.run('pay', {'amount': 1}, ctx)}"
        }

    flow = Flow("outer", state=State).node("go", outer).end("go")
    run = FlowRunner(executor)
    done, resumes = await resume_until_done(run, flow, await run.run(flow, {}))
    assert resumes == 1 and done.state.note == "value of inner | paid 1"
    assert RAN == ["read inner", "pay1"]


async def test_an_approved_inner_side_effect_runs_once_across_outer_resumes():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))

    async def inner_nuke(state, ctx):
        return {"note": await executor.run("xnuke", {}, ctx)}

    inner = Flow("inner_nuke", state=State).node("go", inner_nuke).end("go")

    async def outer(state, ctx):
        runner = FlowRunner(executor)
        result = await runner.run(inner, {}, ctx)
        while result.status == "paused":  # the node approves its inner flow itself
            result = await runner.resume(inner, result.checkpoint, APPROVE, ctx)
        return {"note": f"{result.state.note} | {await executor.run('pay', {'amount': 1}, ctx)}"}

    flow = Flow("outer_nuke", state=State).node("go", outer).end("go")
    run = FlowRunner(executor)
    done, _ = await resume_until_done(run, flow, await run.run(flow, {}))
    assert done.state.note == "nuked | paid 1"
    assert RAN == ["xnuke", "pay1"]


async def test_a_call_with_non_json_arguments_is_journaled():
    executor = Executor(registry, approval=DeferredApproval(MemoryApprovalStore()))
    when = dt.datetime(2026, 10, 29, 12, 0, tzinfo=dt.UTC)

    async def go(state, ctx):
        stamped = await executor.run("stamp", {"when": when}, ctx)
        return {"note": f"{stamped} | {await executor.run('pay', {'amount': 3}, ctx)}"}

    flow = Flow("dt_args", state=State).node("go", go).end("go")
    run = FlowRunner(executor)
    done, _ = await resume_until_done(run, flow, await run.run(flow, {}))
    assert done.state.note == f"{when.isoformat()} | paid 3"
    assert RAN == ["stamp 2026", "pay3"]


async def test_a_node_with_its_own_executor_and_store_resumes_twice_and_runs_once():
    runner_store, own_store = MemoryApprovalStore(), MemoryApprovalStore()
    own = Executor(registry, approval=DeferredApproval(own_store))

    async def pays(state, ctx):
        first = await own.run("pay", {"amount": 1}, ctx)
        second = await own.run("pay", {"amount": 2}, ctx)
        return {"note": f"{first}, {second}"}

    flow = Flow("own_store", state=State).node("go", pays).end("go")
    run = FlowRunner(Executor(registry, approval=DeferredApproval(runner_store)))
    done, resumes = await resume_until_done(run, flow, await run.run(flow, {}))
    assert resumes == 2 and done.state.note == "paid 1, paid 2"
    assert RAN == ["pay1", "pay2"]
    assert await pending_records(runner_store, own_store) == []
    consumed = [r for r in await own_store.list() if r.consumed]
    assert len(consumed) == 2


async def test_an_own_store_decided_otherwise_denies_the_resumed_call():
    own_store = MemoryApprovalStore()
    own = Executor(registry, approval=DeferredApproval(own_store))
    denied = ApprovalDecision(approved=False, reason="no")

    async def pays(state, ctx):
        try:
            return {"note": await own.run("pay", {"amount": 9}, ctx)}
        except ApprovalAlreadyDecided as exc:
            return {"note": exc.kind}

    flow = Flow("own_denied", state=State).node("go", pays).end("go")
    run = FlowRunner(Executor(registry, approval=DeferredApproval(MemoryApprovalStore())))
    paused = await run.run(flow, {})
    await own_store.decide(paused.pending.approval_id, denied)  # decided out of band
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "approval-already-decided" and RAN == []
