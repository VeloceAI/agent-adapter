import dataclasses

import pytest
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalDenied, ApprovalPending, FlowError
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.flow import Call, Flow, Map
from tests.fakes.sinks import RecordingSink
from tests.flow.helpers import CALLS, State, registry, runner

APPROVE = ApprovalDecision(approved=True)
DENY = ApprovalDecision(approved=False, reason="no")


def deferred():
    return DeferredApproval(MemoryApprovalStore())


def one_node(action):
    return Flow("tools", state=State).node("go", action).end("go")


async def test_a_function_node_calling_a_confirm_tool_pauses_and_resumes():
    holder = {}

    async def go(state, ctx):
        # a fresh call id on every visit, as a model-driven tool loop would make
        value = await holder["executor"].run("publish", {"text": state.query}, ctx)
        return {"note": value}

    run, executor, _ = runner(approval=deferred())
    holder["executor"] = executor
    flow = one_node(go)
    paused = await run.run(flow, {"query": "hi"})
    assert paused.status == "paused" and paused.pending.tool == "publish"
    assert paused.pending.call_id.startswith("call_")
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.status == "done" and done.state.note == "published hi"


async def test_a_function_node_batch_with_fresh_ids_pauses_and_resumes():
    holder = {}
    turn = []

    async def go(state, ctx):
        turn.append(1)
        calls = [ToolCall(name="publish", arguments={"text": "a"}, id=f"model_{len(turn)}")]
        [result] = await holder["executor"].run_many(calls, ctx)
        return {"note": result.unwrap()}

    run, executor, _ = runner(approval=deferred())
    holder["executor"] = executor
    flow = one_node(go)
    paused = await run.run(flow, {})
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "published a" and len(turn) == 2


async def test_run_tool_loop_inside_a_node_pauses_and_resumes():
    tool_loop = pytest.importorskip("agent_adapter.exec.tool_loop")  # WP08
    from agent_adapter import Message
    from tests.fakes.models import ScriptedModel, calls, text

    holder = {}
    replies = [calls(("publish", {"text": "x"})), calls(("publish", {"text": "x"})), text("ok")]

    async def go(state, ctx):
        result = await tool_loop.run_tool_loop(
            holder["model"],
            registry.subset(names=["publish"]),
            [Message(role="user", content="publish x")],
            executor=holder["executor"],
            ctx=ctx,
        )
        for tool_result in result.tool_results:
            tool_result.unwrap()  # a pending result re-raises ApprovalPending: the flow pauses
        return {"note": result.final.text}

    run, executor, _ = runner(approval=deferred())
    holder["executor"], holder["model"] = executor, ScriptedModel(replies)
    flow = one_node(go)
    paused = await run.run(flow, {})
    assert paused.status == "paused"
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "ok"


async def test_only_the_paused_node_runs_again_on_resume():
    CALLS.clear()
    sink = RecordingSink()
    flow = (
        Flow("siblings", state=State)
        .node("start", lambda s: None)
        .node("ask", Call("publish", args={"text": "p"}, into="note"))
        .node("side", Call("search", args={"query": "q"}, into="hits"))
        .edge("start", ["ask", "side"])
        .end("ask", "side")
    )
    run, _, _ = runner(approval=deferred(), sink=sink)
    paused = await run.run(flow, {})
    assert paused.status == "paused" and paused.state.hits == []
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.state.note == "published p" and len(done.state.hits) == 2
    assert CALLS == ["q"]  # the sibling's tool ran once
    assert done.visits == {"start": 1, "ask": 1, "side": 1} and done.steps == 2
    side_starts = [e for e in sink.of_type("node.started") if e.data["node"] == "side"]
    assert len(side_starts) == 1


async def test_a_denied_tool_on_resume_fails_the_node():
    run, _, _ = runner(approval=deferred())
    flow = one_node(Call("publish", args={"text": "x"}, into="note"))
    paused = await run.run(flow, {})
    with pytest.raises(FlowError, match="approval-denied") as info:
        await run.resume(flow, paused.checkpoint, DENY)
    assert info.value.node == "go" and isinstance(info.value.__cause__, ApprovalDenied)


async def test_a_denied_map_item_on_resume_is_skipped():
    run, _, _ = runner(approval=deferred())
    flow = one_node(Map("items", tool="publish", into="log", on_error="skip"))
    paused = await run.run(flow, {"items": [{"text": "a"}]})
    done = await run.resume(flow, paused.checkpoint, DENY)
    assert done.status == "done" and done.state.log == []


async def test_resume_without_ctx_keeps_a_caller_owned_run_open():
    sink = RecordingSink()
    run, executor, _ = runner(approval=deferred(), sink=sink)
    flow = one_node(Call("publish", args={"text": "x"}, into="note"))
    ctx = executor.context(run_id="run_caller")
    paused = await run.run(flow, {}, ctx)
    done = await run.resume(flow, paused.checkpoint, APPROVE)
    assert done.status == "done"
    assert "run.started" not in sink.types() and "run.finished" not in sink.types()
    assert {e.run_id for e in sink.events} == {"run_caller"}


async def test_resume_rejects_a_checkpoint_with_another_state_model():
    class Other(BaseModel):
        x: int = 0

    run, _, _ = runner(approval=deferred())
    flow = one_node(Call("publish", args={"text": "x"}, into="note"))
    paused = await run.run(flow, {})
    wrong = dataclasses.replace(paused.checkpoint, state=Other())
    with pytest.raises(FlowError, match="checkpoint state is Other, not State"):
        await run.resume(flow, wrong, APPROVE)


async def test_a_pending_without_a_tool_pauses_but_resume_is_refused():
    async def go(state, ctx):
        raise ApprovalPending("apr_x")

    run, _, _ = runner()
    flow = one_node(go)
    paused = await run.run(flow, {})
    assert paused.status == "paused" and paused.pending.tool == ""
    with pytest.raises(FlowError, match="names no tool; resume refused"):
        await run.resume(flow, paused.checkpoint, APPROVE)


async def test_a_tool_raising_pending_itself_gets_its_call_named():
    from agent_adapter import Registry
    from agent_adapter.exec import Executor
    from agent_adapter.flow import FlowRunner

    own = Registry()

    @own.tool
    async def waits() -> str:
        """Waits for someone."""
        raise ApprovalPending("apr_tool")

    flow = (
        Flow("self_pending", state=State)
        .node("go", Map("items", tool="waits", into="log"))
        .end("go")
    )
    flow2 = Flow("self_pending2", state=State).node("go", Call("waits", into="note")).end("go")
    runner_ = FlowRunner(Executor(own))
    paused = await runner_.run(flow2, {})
    assert paused.pending.tool == "waits" and paused.pending.call_id == "self_pending2:go:1:0"
    mapped = await runner_.run(flow, {"items": [{}]})
    assert mapped.pending.tool == "waits" and mapped.pending.call_id == "self_pending:go:1:0"
