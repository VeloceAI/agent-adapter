"""Resume must answer only the request it was given for: the review's adversarial probes."""

import asyncio
import dataclasses

import pytest

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalPending, FlowError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.flow import Flow, FlowRunner, Map
from tests.fakes.approval import ScriptedApproval
from tests.flow.helpers import NUKED, PAID, State, runner

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")


@pytest.fixture(autouse=True)
def _reset():
    PAID.clear()
    NUKED.clear()


def deferred():
    return DeferredApproval(MemoryApprovalStore())


def one_node(action, name="sec"):
    return Flow(name, state=State).node("go", action).end("go")


def with_executor(fn):
    """A node calling tools through the executor in `holder` (as user code holds its own)."""
    holder = {}

    async def go(state, ctx):
        return await fn(holder["executor"], state, ctx)

    return go, holder


async def test_p1_arguments_changed_between_pause_and_resume_go_to_the_port():
    amount = {"value": 1}

    async def pays(executor, state, ctx):
        return {"note": await executor.run("pay", {"amount": amount["value"]}, ctx)}

    go, holder = with_executor(pays)
    run, holder["executor"], _ = runner(approval=deferred())
    flow = one_node(go)
    paused = await run.run(flow, {})
    assert paused.pending.arguments == {"amount": 1}  # the approver sees the real arguments
    amount["value"] = 1_000_000
    again = await run.resume(flow, paused.checkpoint, APPROVE)
    assert again.status == "paused" and again.pending.arguments == {"amount": 1_000_000}
    assert PAID == []


async def test_p2_one_approval_never_covers_a_concurrent_call_with_other_arguments():
    turn = []

    async def pays(executor, state, ctx):
        turn.append(1)
        if len(turn) == 1:
            return {"note": await executor.run("pay", {"amount": 1}, ctx)}
        big, small = await asyncio.gather(
            executor.run("pay", {"amount": 999}, ctx),
            executor.run("pay", {"amount": 1}, ctx),
            return_exceptions=True,
        )
        if isinstance(big, Exception):
            raise big
        return {"note": f"{big} {small}"}

    go, holder = with_executor(pays)
    run, holder["executor"], _ = runner(approval=deferred())
    flow = one_node(go)
    paused = await run.run(flow, {})
    again = await run.resume(flow, paused.checkpoint, APPROVE)
    assert again.status == "paused" and again.pending.arguments == {"amount": 999}
    assert PAID == [1]  # only the approved call ran


async def test_p3_a_port_that_names_no_tool_still_binds_the_answer_to_the_tool():
    turn = []

    async def calls(executor, state, ctx):
        turn.append(1)
        name = "publish" if len(turn) == 1 else "nuke"
        args = {"text": "x"} if name == "publish" else {}
        return {"note": await executor.run(name, args, ctx)}

    port = ScriptedApproval([ApprovalPending("apr_x"), ApprovalPending("apr_y")])
    go, holder = with_executor(calls)
    run, holder["executor"], _ = runner(approval=port)
    flow = one_node(go)
    paused = await run.run(flow, {})
    assert paused.pending.tool == "publish"  # filled in from the request
    again = await run.resume(flow, paused.checkpoint, APPROVE)
    assert again.status == "paused" and again.pending.tool == "nuke" and NUKED == []


async def test_p4_a_reused_flow_call_id_for_another_tool_is_not_answered():
    turn = []

    async def calls(executor, state, ctx):
        turn.append(1)
        call_id = f"{ctx.call_id}:0"  # the deterministic id a Call node would use
        if len(turn) == 1:
            return {"note": await executor.run("publish", {"text": "x"}, ctx, call_id=call_id)}
        return {"note": await executor.run("nuke", {}, ctx, call_id=call_id)}

    go, holder = with_executor(calls)
    run, holder["executor"], _ = runner(approval=deferred())
    flow = one_node(go)
    paused = await run.run(flow, {})
    assert paused.pending.call_id == "sec:go:1:0"
    again = await run.resume(flow, paused.checkpoint, APPROVE)
    assert again.status == "paused" and again.pending.tool == "nuke" and NUKED == []


async def test_p9_a_nested_flow_never_sees_the_outer_answer():
    turn = []
    inner_status = []

    async def outer(executor, state, ctx):
        turn.append(1)
        if len(turn) > 1:
            inner = one_node(lambda s, c: inner_pay(executor, c))
            result = await FlowRunner(executor).run(inner, {}, ctx)
            inner_status.append(result.status)
        return {"note": await executor.run("pay", {"amount": 1}, ctx)}

    async def inner_pay(executor, ctx):
        return {"note": await executor.run("pay", {"amount": 1}, ctx)}

    go, holder = with_executor(outer)
    run, holder["executor"], _ = runner(approval=deferred())
    flow = one_node(go)
    paused = await run.run(flow, {})
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert inner_status == ["paused"]  # the same-named inner node asked the port
    assert done.status == "done" and PAID == [1]


async def test_p5_a_task_that_outlives_the_resumed_node_asks_the_port():
    later = []
    turn = []

    async def spawns(executor, state, ctx):
        turn.append(1)
        if len(turn) == 1:
            await executor.run("pay", {"amount": 1}, ctx)

        async def background():
            await asyncio.sleep(0.01)
            return await executor.run("pay", {"amount": 1}, ctx)

        later.append(asyncio.create_task(background()))
        return {"note": "returned"}

    go, holder = with_executor(spawns)
    run, holder["executor"], _ = runner(approval=deferred())
    flow = one_node(go)
    paused = await run.run(flow, {})
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "returned"
    with pytest.raises(ApprovalPending):
        await later[0]
    assert PAID == []


async def test_p6_a_checkpoint_resumes_once():
    store = MemoryApprovalStore()
    run, executor, _ = runner(approval=DeferredApproval(store))
    flow = one_node(Map("items", tool="pay", into="log"))
    paused = await run.run(flow, {"items": [{"amount": 5}]})
    assert (await run.resume(flow, paused.checkpoint, APPROVE)).status == "done"
    with pytest.raises(ApprovalAlreadyDecided):
        await run.resume(flow, paused.checkpoint, APPROVE)  # the same runner
    with pytest.raises(ApprovalAlreadyDecided):
        await FlowRunner(executor).resume(flow, paused.checkpoint, APPROVE)  # via the store
    assert PAID == [5]
    record = await store.get(paused.pending.approval_id)
    assert record.status == "approved" and record.consumed


async def test_p6_a_gate_checkpoint_resumes_once():
    store = MemoryApprovalStore()
    flow = Flow("gate_once", state=State).node("go", lambda s: {"n": s.n + 1}, pause=True).end("go")
    run, executor, _ = runner(flow_approval=DeferredApproval(store))
    paused = await run.run(flow, {})
    assert (await run.resume(flow, paused.checkpoint, APPROVE)).state.n == 1
    with pytest.raises(ApprovalAlreadyDecided):
        await FlowRunner(executor, approval=DeferredApproval(store)).resume(
            flow, paused.checkpoint, APPROVE
        )


async def test_a_store_decision_must_agree_with_the_resume_decision():
    store = MemoryApprovalStore()
    run, _, _ = runner(approval=DeferredApproval(store))
    flow = one_node(Map("items", tool="pay", into="log"))
    paused = await run.run(flow, {"items": [{"amount": 5}]})
    await store.decide(paused.pending.approval_id, DENY)  # decided out of band
    with pytest.raises(ApprovalAlreadyDecided):
        await run.resume(flow, paused.checkpoint, APPROVE)
    assert PAID == []

    paused = await run.run(flow, {"items": [{"amount": 6}]})
    await store.decide(paused.pending.approval_id, APPROVE)  # the same decision is fine
    assert (await run.resume(flow, paused.checkpoint, APPROVE)).status == "done"
    assert PAID == [6]


async def test_a_checkpoint_bound_to_another_store_record_is_refused():
    store = MemoryApprovalStore()
    run, _, _ = runner(approval=DeferredApproval(store))
    flow = one_node(Map("items", tool="pay", into="log"))
    paused = await run.run(flow, {"items": [{"amount": 5}]})
    forged = paused.pending.model_copy(update={"arguments": {"amount": 500}})
    with pytest.raises(FlowError, match="checkpoint-tampered"):  # the seal binds `pending`
        await run.resume(flow, dataclasses.replace(paused.checkpoint, pending=forged), APPROVE)
    assert PAID == []


async def test_p7_resume_in_another_run_is_not_answered():
    run, executor, _ = runner(approval=deferred())
    flow = one_node(Map("items", tool="pay", into="log"))
    paused = await run.run(flow, {"items": [{"amount": 5}]})
    other = executor.context(run_id="run_other")
    again = await run.resume(flow, paused.checkpoint, APPROVE, other)
    assert again.status == "paused" and again.pending.run_id == "run_other"
    assert PAID == []


async def test_a_map_needing_two_approvals_finishes_after_two_resumes():
    run, _, _ = runner(approval=deferred())
    flow = one_node(Map("items", tool="pay", into="log"), name="two_map")
    paused = await run.run(flow, {"items": [{"amount": 1}, {"amount": 2}]})
    first = paused.pending
    second_pause = await run.resume(flow, paused.checkpoint, APPROVE)
    assert second_pause.status == "paused" and second_pause.pending.call_id != first.call_id
    done = await run.resume(flow, second_pause.checkpoint, APPROVE)
    assert done.status == "done" and sorted(done.state.log) == ["paid 1", "paid 2"]
    assert sorted(PAID) == [1, 2]  # each approved item ran once


async def test_a_function_node_needing_two_approvals_finishes_after_two_resumes():
    async def pays(executor, state, ctx):
        first = await executor.run("pay", {"amount": 1}, ctx)
        second = await executor.run("pay", {"amount": 2}, ctx)
        return {"note": f"{first}, {second}"}

    go, holder = with_executor(pays)
    run, holder["executor"], _ = runner(approval=deferred())
    flow = one_node(go, name="two_fn")
    paused = await run.run(flow, {})
    assert paused.pending.arguments == {"amount": 1}
    second = await run.resume(flow, paused.checkpoint, APPROVE)
    assert second.status == "paused" and second.pending.arguments == {"amount": 2}
    done = await run.resume(flow, second.checkpoint, APPROVE)
    assert done.state.note == "paid 1, paid 2"
    assert PAID == [1, 2]  # the re-run replays pay(1) from the node's journal
