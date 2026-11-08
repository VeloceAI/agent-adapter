"""A pause inside a sub-flow pauses the whole run; resume continues the child where it stopped."""

import json

import pytest
from pydantic import BaseModel

from agent_adapter.approval import AutoApprove, AutoDeny, DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided, FlowError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, FlowRunner, Map, SubFlow
from tests.fakes.sinks import RecordingSink
from tests.flow.helpers import PAID, registry, resume_until_done, runner
from tests.flow.subflow_helpers import Order, child_flow, parent_flow, pay_flow

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")


@pytest.fixture(autouse=True)
def _reset():
    PAID.clear()


def deferred():
    return DeferredApproval(MemoryApprovalStore())


async def test_a_child_gate_pauses_the_parent_and_resume_finishes_it():
    sink = RecordingSink()
    run, _, _ = runner(approval=deferred(), sink=sink)
    flow = parent_flow(child_flow(gate=True))
    paused = await run.run(flow, {"item": "mug"})
    assert paused.status == "paused" and paused.pending.kind == "node"
    assert paused.pending.tool == "node:child.check"
    assert paused.pending.call_id == "parent:sub:1/child:check:1"
    meta = paused.pending.metadata
    assert (meta["flow"], meta["node"], meta["visit"]) == ("parent", "sub", 1)
    assert meta["subflow"] == {
        "flow": "child",
        "node": "check",
        "visit": 1,
        "scope": "parent:sub:1/",
    }
    assert paused.checkpoint.frontier == ("sub",)
    paused_types = [(e.type, e.data.get("flow")) for e in sink.events if e.type == "node.paused"]
    assert paused_types == [("node.paused", "child"), ("node.paused", "parent")]
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.status == "done"
    assert done.state.log == ["start", "count mug", "check", "after checked 3"]
    count_runs = [e for e in sink.events if e.type == "node.started" and e.data["node"] == "count"]
    assert len(count_runs) == 1  # the child's finished superstep does not run again


async def test_a_child_gate_edit_applies_to_the_child_state():
    run, _, _ = runner(approval=deferred())
    flow = parent_flow(child_flow(gate=True))
    paused = await run.run(flow, {})
    done = await run.resume(
        flow, paused.checkpoint, ApprovalDecision(approved=True, edited_args={"qty": 10})
    )
    assert done.state.qty == 10 and done.state.note == "checked 10"


async def test_a_tool_approval_inside_the_child_pauses_and_runs_once():
    run, _, _ = runner(approval=deferred())
    flow = parent_flow(pay_flow())
    paused = await run.run(flow, {})
    assert paused.pending.kind == "tool" and paused.pending.tool == "pay"
    assert paused.pending.call_id == "parent:sub:1/payer:pay:1:0"
    done, resumes = await resume_until_done(run, flow, paused)
    assert resumes == 1 and done.state.note == "paid 3" and PAID == [3]


async def test_a_denied_child_tool_fails_the_subflow_node():
    run, _, _ = runner(approval=deferred())
    flow = parent_flow(pay_flow())
    paused = await run.run(flow, {})
    with pytest.raises(FlowError) as caught:
        await run.resume(flow, paused.checkpoint, DENY)
    assert caught.value.node == "sub" and PAID == []


async def test_a_denied_child_gate_ends_the_run_denied_at_its_path():
    sink = RecordingSink()
    run, _, _ = runner(approval=AutoDeny(), sink=sink)
    done = await run.run(parent_flow(child_flow(gate=True)), {})
    assert done.status == "denied" and done.denied_at == "sub/check"
    failed = [e for e in sink.events if e.type == "node.failed"]
    assert failed[0].data["node"] == "sub" and failed[0].data["error_kind"] == "approval-denied"
    child_end = [e for e in sink.events if e.type == "flow.finished" and e.data["flow"] == "child"]
    assert child_end[0].data["status"] == "denied"


async def test_an_approved_child_gate_runs_straight_through():
    run, _, _ = runner(approval=AutoApprove())
    done = await run.run(parent_flow(child_flow(gate=True)), {})
    assert done.status == "done" and done.state.note == "checked 3"


async def test_a_paused_checkpoint_round_trips_through_json():
    run, _, _ = runner(approval=deferred())
    flow = parent_flow(child_flow(gate=True))
    paused = await run.run(flow, {"item": "mug"})
    text = run.dump_checkpoint(paused.checkpoint)
    assert "children" in json.loads(text)["payload"]
    loaded = run.load_checkpoint(flow, text)
    assert loaded._children["sub"].frontier == ("check",)
    done = await run.resume(flow, loaded, APPROVE)
    assert done.status == "done" and done.state.note == "checked 3"


async def test_a_changed_child_run_in_the_json_is_refused():
    run, _, _ = runner(approval=deferred())
    flow = parent_flow(child_flow(gate=True))
    paused = await run.run(flow, {})
    document = json.loads(run.dump_checkpoint(paused.checkpoint))
    document["payload"]["children"]["sub"]["steps"] = 0
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        run.load_checkpoint(flow, json.dumps(document))


async def test_a_checkpoint_of_another_child_graph_is_refused():
    run, _, _ = runner(approval=deferred())
    paused = await run.run(parent_flow(child_flow(gate=True)), {})
    changed = parent_flow(child_flow("child", gate=True).node("extra", lambda s: None))
    with pytest.raises(FlowError):
        await run.resume(changed, paused.checkpoint, APPROVE)


async def test_a_child_checkpoint_resumes_once():
    run, _, _ = runner(approval=deferred())
    flow = parent_flow(child_flow(gate=True))
    paused = await run.run(flow, {})
    await run.resume(flow, paused.checkpoint, APPROVE)
    with pytest.raises(ApprovalAlreadyDecided):
        await run.resume(flow, paused.checkpoint, APPROVE)


async def test_nested_sub_flows_pause_and_resume_through_every_level():
    inner = child_flow("inner", gate=True)
    middle = Flow("middle", state=Order).node("deep", SubFlow(inner)).end("deep")
    flow = parent_flow(middle)
    run, _, _ = runner(approval=deferred())
    paused = await run.run(flow, {})
    assert paused.pending.call_id == "parent:sub:1/middle:deep:1/inner:check:1"
    assert paused.pending.metadata["subflow"]["subflow"]["node"] == "check"
    text = run.dump_checkpoint(paused.checkpoint)
    done = await run.resume(flow, run.load_checkpoint(flow, text), APPROVE)
    assert done.status == "done" and done.state.note == "checked 3"


async def test_two_parallel_subflows_both_pause_and_both_resume():
    gated = child_flow(gate=True)
    flow = (
        Flow("pair", state=Order)
        .node("start", lambda s: None)
        .node("left", SubFlow(gated, into={"note": "note"}))
        .node("right", SubFlow(gated, into={"qty": "qty"}))
        .edge("start", ["left", "right"])
        .end("left", "right")
    )
    run, _, _ = runner(approval=deferred())
    paused = await run.run(flow, {})
    assert paused.pending.call_id == "pair:left:1/child:check:1"
    second = await run.resume(flow, paused.checkpoint, APPROVE)
    assert second.status == "paused" and second.pending.call_id == "pair:right:1/child:check:1"
    done = await run.resume(flow, second.checkpoint, APPROVE)
    assert done.status == "done" and done.state.note == "checked 1" and done.state.qty == 1


async def test_a_pausing_sibling_and_a_finished_subflow_keep_their_work():
    flow = (
        Flow("mixed", state=Order)
        .node("start", lambda s: None)
        .node("sub", SubFlow(child_flow(), into={"note": "note"}))
        .node("payer", SubFlow(pay_flow(), into={"total": "qty"}))
        .edge("start", ["sub", "payer"])
        .end("sub", "payer")
    )
    run, _, _ = runner(approval=deferred())
    paused = await run.run(flow, {})
    assert paused.pending.tool == "pay"
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "checked 1" and done.state.total == 1 and PAID == [1]


async def test_the_gate_port_of_the_runner_answers_child_gates():
    run = FlowRunner(Executor(registry), approval=AutoApprove())
    done = await run.run(parent_flow(child_flow(gate=True)), {})
    assert done.status == "done"


async def test_a_child_map_keeps_its_finished_items_across_resumes():
    items_child = (
        Flow("items", state=Order)
        .node("pay", Map("log", tool="pay", args_fn=lambda n: {"amount": int(n)}, into="log"))
        .end("pay")
    )
    flow = (
        Flow("outer", state=Order)
        .node(
            "sub", SubFlow(items_child, input=lambda s: {"log": ["1", "2"]}, into={"note": "item"})
        )
        .end("sub")
    )
    run, _, _ = runner(approval=deferred())
    paused = await run.run(flow, {})
    done, resumes = await resume_until_done(run, flow, paused)
    assert resumes == 2 and done.status == "done" and sorted(PAID) == [1, 2]


async def test_a_child_node_with_two_approvals_resumes_twice_and_pays_each_once():
    holder = {}

    async def pays(state, ctx):
        first = await holder["executor"].run("pay", {"amount": 1}, ctx)
        second = await holder["executor"].run("pay", {"amount": 2}, ctx)
        return {"note": f"{first}, {second}"}

    child = Flow("twice", state=Order).node("go", pays).end("go")
    run, executor, _ = runner(approval=deferred())
    holder["executor"] = executor
    flow = parent_flow(child)
    done, resumes = await resume_until_done(run, flow, await run.run(flow, {}))
    assert resumes == 2 and done.state.note == "paid 1, paid 2" and PAID == [1, 2]


async def test_a_child_state_that_cannot_be_sealed_withdraws_the_pause():
    class Odd(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        thing: object = None
        note: str = ""

    child = (
        Flow("odd", state=Odd)
        .node("make", lambda s: {"thing": object()})
        .node("pay", Call("pay", args={"amount": 5}, into="note"))
        .edge("make", "pay")
        .end("pay")
    )
    store = MemoryApprovalStore()
    run, _, _ = runner(approval=DeferredApproval(store))
    flow = Flow("outer", state=Order).node("sub", SubFlow(child, input=lambda s: {})).end("sub")
    with pytest.raises(FlowError, match="state-not-sealable"):
        await run.run(flow, {})
    assert await store.list(status="pending") == [] and PAID == []
