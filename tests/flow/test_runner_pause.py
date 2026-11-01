import pytest

from agent_adapter.approval import (
    AutoApprove,
    AutoDeny,
    CallbackApproval,
    DeferredApproval,
    MemoryApprovalStore,
)
from agent_adapter.core.errors import ApprovalPending, FlowError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import RunLimits
from agent_adapter.flow import Call, Flow, FlowCheckpoint, Map
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.fakes.sinks import RecordingSink
from tests.flow.helpers import State, runner

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")


def review_flow(pause=lambda s: {"draft": s.note}):
    seen = {}

    def publish(state, ctx):
        seen["approval"] = ctx.approval
        return {"log": [f"publish {state.note}"]}

    flow = (
        Flow("review", state=State)
        .node("write", lambda s: {"note": s.query.upper(), "log": ["write"]})
        .node("publish", publish, pause=pause)
        .edge("write", "publish")
        .end("publish")
    )
    return flow, seen


async def test_pause_asks_the_port_with_a_node_request_and_runs_with_ctx_approval():
    flow, seen = review_flow()
    port = RecordingApproval(AutoApprove())
    run, _, _ = runner(flow_approval=port)
    result = await run.run(flow, {"query": "hello"})
    assert result.status == "done" and result.state.log == ["write", "publish HELLO"]
    [request] = port.requests
    assert request.kind == "node" and request.tool == "node:review.publish"
    assert request.arguments == {"draft": "HELLO"}
    assert request.call_id == "review:publish:1" and request.run_id.startswith("run_")
    assert request.metadata == {"flow": "review", "node": "publish", "visit": 1}
    assert seen["approval"].approved is True


async def test_pause_true_sends_an_empty_payload_through_the_executor_port():
    flow, _ = review_flow(pause=True)
    port = RecordingApproval(AutoApprove())
    run, _, _ = runner(approval=port)  # the executor's port is the fallback
    await run.run(flow, {"query": "x"})
    assert port.requests[0].arguments == {}


async def test_pending_pause_returns_a_checkpoint_and_resume_runs_the_node():
    flow, seen = review_flow()
    sink = RecordingSink()
    port = ScriptedApproval([ApprovalPending("apr_wait")])
    run, _, _ = runner(flow_approval=port, sink=sink)
    paused = await run.run(flow, {"query": "hello"})
    assert paused.status == "paused" and paused.steps == 1
    assert paused.state.log == ["write"] and "publish" not in paused.visits
    assert paused.pending.approval_id == "apr_wait" and paused.pending.kind == "node"
    checkpoint = paused.checkpoint
    assert isinstance(checkpoint, FlowCheckpoint)
    assert checkpoint.frontier == ("publish",) and checkpoint.pending == paused.pending
    assert "run.finished" not in sink.types()
    [event] = sink.of_type("node.paused")
    assert event.data == {"flow": "review", "node": "publish", "approval_id": "apr_wait"}
    assert event.call_id == "review:publish:1"

    done = await run.resume(flow, checkpoint, APPROVE)
    assert done.status == "done" and done.state.log == ["write", "publish HELLO"]
    assert done.visits == {"write": 1, "publish": 1} and done.steps == 2
    assert seen["approval"] == APPROVE
    assert len(port.requests) == 1  # the resume decision answers; the port is not asked again
    assert sink.types().count("run.started") == 1 and sink.types().count("run.finished") == 1
    run_ids = {e.run_id for e in sink.events}
    assert len(run_ids) == 1  # resume continues the same run


async def test_resume_with_edited_args_updates_state_before_the_node_runs():
    flow, _ = review_flow()
    run, _, _ = runner(flow_approval=ScriptedApproval([ApprovalPending("apr_1")]))
    paused = await run.run(flow, {"query": "hello"})
    edit = ApprovalDecision(approved=True, edited_args={"note": "EDITED", "log": ["edited"]})
    done = await run.resume(flow, paused.checkpoint, edit)
    assert done.state.note == "EDITED"
    assert done.state.log == ["write", "edited", "publish EDITED"]  # edits merge by rule


async def test_resume_with_invalid_edits_is_a_flow_error():
    flow, _ = review_flow()
    for edits, message in [({"n": "x"}, "edited state is invalid"), ({"zz": 1}, "unknown state")]:
        run, _, _ = runner(flow_approval=ScriptedApproval([ApprovalPending("apr_1")]))
        paused = await run.run(flow, {"query": "hello"})
        bad = ApprovalDecision(approved=True, edited_args=edits)
        with pytest.raises(FlowError, match=message) as info:
            await run.resume(flow, paused.checkpoint, bad)
        assert info.value.node == "publish"


async def test_resume_denied_stops_with_denied_at():
    flow, _ = review_flow()
    sink = RecordingSink()
    run, _, _ = runner(flow_approval=ScriptedApproval([ApprovalPending("apr_1")]), sink=sink)
    paused = await run.run(flow, {"query": "hello"})
    denied = await run.resume(flow, paused.checkpoint, DENY)
    assert denied.status == "denied" and denied.denied_at == "publish"
    assert denied.state.log == ["write"] and denied.checkpoint is None
    finished = sink.of_type("flow.finished")[-1]
    assert finished.data == {"flow": "review", "status": "denied", "steps": 1}
    assert sink.of_type("run.finished")[-1].data == {"ok": False}


async def test_port_denial_and_missing_port_deny_the_flow(caplog):
    flow, _ = review_flow()
    run, _, _ = runner(flow_approval=AutoDeny())
    result = await run.run(flow, {"query": "x"})
    assert result.status == "denied" and result.denied_at == "publish"
    run, _, _ = runner()
    result = await run.run(flow, {"query": "x"})
    assert result.status == "denied"
    assert "no approval port configured" in caplog.text


async def test_granted_gates_are_not_asked_again_after_a_later_gate_pends():
    flow = (
        Flow("gates", state=State)
        .node("start", lambda s: None)
        .node("a", lambda s: {"log": ["a"]}, pause=True)
        .node("b", lambda s: {"log": ["b"]}, pause=True)
        .edge("start", ["a", "b"])
        .end("a", "b")
    )
    port = ScriptedApproval([APPROVE, ApprovalPending("apr_b")])
    run, _, _ = runner(flow_approval=port)
    paused = await run.run(flow, {})
    assert paused.pending.metadata["node"] == "b" and paused.checkpoint.decisions == {"a": APPROVE}
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.log == ["a", "b"] and len(port.requests) == 2


async def test_pause_payload_errors_are_flow_errors():
    def broken(state):
        raise RuntimeError("secret")

    for pause, message in [
        (broken, "pause payload failed: RuntimeError"),
        (lambda s: [1], "must return a mapping"),
    ]:
        flow, _ = review_flow(pause=pause)
        run, _, _ = runner(flow_approval=AutoApprove())
        with pytest.raises(FlowError, match=message) as info:
            await run.run(flow, {"query": "x"})
        assert info.value.node == "publish" and "secret" not in str(info.value)


async def test_a_failing_port_is_a_flow_error():
    flow, _ = review_flow()
    run, _, _ = runner(flow_approval=ScriptedApproval([ConnectionError("down")]))
    with pytest.raises(FlowError, match="approval failed: ConnectionError") as info:
        await run.run(flow, {"query": "x"})
    assert info.value.node == "publish"


async def test_resume_checks_its_inputs():
    flow, _ = review_flow()
    run, _, _ = runner(flow_approval=ScriptedApproval([ApprovalPending("apr_1")]))
    paused = await run.run(flow, {"query": "x"})
    checkpoint = paused.checkpoint
    other = Flow("other", state=State).node("publish", lambda s: None).end("publish")
    with pytest.raises(FlowError, match="cannot resume flow 'other'"):
        await run.resume(other, checkpoint, APPROVE)
    from dataclasses import replace

    with pytest.raises(FlowError, match="nothing pending"):
        await run.resume(flow, replace(checkpoint, pending=None), APPROVE)
    with pytest.raises(FlowError, match="not in the checkpoint's frontier"):
        await run.resume(flow, replace(checkpoint, frontier=("write",)), APPROVE)
    with pytest.raises(ToolUsageError, match="expected an ApprovalDecision"):
        await run.resume(flow, checkpoint, True)
    with pytest.raises(ToolUsageError, match="expected a FlowCheckpoint"):
        await run.resume(flow, {"flow": "review"}, APPROVE)


def publish_flow(action):
    return Flow("tools", state=State).node("go", action).end("go")


async def test_a_pending_tool_pauses_the_flow_and_resume_answers_it():
    sink = RecordingSink()
    store = MemoryApprovalStore()
    run, _, _ = runner(approval=DeferredApproval(store), sink=sink)
    flow = publish_flow(Call("publish", args=lambda s: {"text": s.query}, into="note"))
    paused = await run.run(flow, {"query": "hi"})
    pending = paused.pending
    assert paused.status == "paused" and pending.kind == "tool"
    assert pending.tool == "publish" and pending.call_id == "tools:go:1:0"
    assert pending.metadata == {"flow": "tools", "node": "go", "visit": 1}
    assert sink.of_type("node.paused")[0].data["approval_id"] == pending.approval_id

    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.status == "done" and done.state.note == "published hi"
    decided = sink.of_type("approval.decided")[-1]
    assert decided.data["approved"] is True and decided.call_id == "tools:go:1:0"


async def test_resumed_tool_approval_can_edit_its_arguments():
    run, _, _ = runner(approval=DeferredApproval(MemoryApprovalStore()))
    flow = publish_flow(Call("publish", args={"text": "draft"}, into="note"))
    paused = await run.run(flow, {})
    edit = ApprovalDecision(approved=True, edited_args={"text": "final"})
    assert (await run.resume(flow, paused.checkpoint, edit)).state.note == "published final"


async def test_a_pending_map_item_pauses_and_resumes():
    port = ScriptedApproval([ApprovalPending("apr_item")])
    run, _, _ = runner(approval=port)
    flow = publish_flow(Map("items", tool="publish", into="log"))
    items = [{"text": "a"}]
    paused = await run.run(flow, {"items": items})
    assert paused.pending.call_id == "tools:go:1:0" and paused.pending.tool == "publish"
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.log == ["published a"]


async def test_resumed_tool_answer_leaves_other_requests_to_the_executor_port():
    port = ScriptedApproval([ApprovalPending("apr_0"), APPROVE])
    run, _, _ = runner(approval=port, limits=RunLimits(max_concurrency=1))
    flow = publish_flow(Map("items", tool="publish", into="log"))
    paused = await run.run(flow, {"items": [{"text": "a"}, {"text": "b"}]})
    assert paused.pending.call_id == "tools:go:1:0"
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.log == ["published a", "published b"]
    # item 1 finished before the pause and is kept; item 0 is answered by the resume
    assert [r.call_id for r in port.requests] == ["tools:go:1:0", "tools:go:1:1"]


async def test_approval_pending_raised_by_a_node_function_pauses():
    def waits(state, ctx):
        raise ApprovalPending("apr_fn")

    run, _, _ = runner()
    paused = await run.run(publish_flow(waits), {})
    assert paused.status == "paused" and paused.pending.call_id == "tools:go:1"
    assert paused.pending.approval_id == "apr_fn" and paused.pending.kind == "tool"


async def test_callback_approval_bool_results_work_as_gates():
    flow, _ = review_flow()
    run, _, _ = runner(flow_approval=CallbackApproval(lambda request: True))
    assert (await run.run(flow, {"query": "x"})).status == "done"
