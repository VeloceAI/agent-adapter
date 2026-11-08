"""`compile_flow` / `run_compiled`: graph shape, errors, gates, ports and bound resumes."""

from typing import Annotated, Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from agent_adapter.approval import (
    AutoApprove,
    AutoDeny,
    CallbackApproval,
    DeferredApproval,
    MemoryApprovalStore,
)
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    FlowCompileError,
    FlowDefinitionError,
    FlowError,
    FlowLoopExceeded,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, Merge, Route
from agent_adapter.integrations.langgraph import (
    INTEGRATION,
    LangGraphApproval,
    compile_flow,
    run_compiled,
)
from agent_adapter.integrations.langgraph._spent import CLAIMS
from tests.fakes.sinks import RecordingSink
from tests.integrations.kit import FakeRunner
from tests.integrations.langgraph.support import bound, config

pytestmark = pytest.mark.contract

registry = Registry()
SENT: list[str] = []


@registry.tool(confirm=True)
async def send(text: str) -> str:
    """Send a text (needs approval)."""
    SENT.append(text)
    return f"sent {text}"


class Note(BaseModel):
    text: str = ""
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)


def mark(name: str, **update: Any) -> Any:
    def node(state: Note) -> dict[str, Any]:
        return {"log": [name], **update}

    node.__name__ = name
    return node


def gated(*, payload: bool = False) -> Flow[Note]:
    pause = (lambda s: {"text": s.text}) if payload else True
    return (
        Flow("gated", state=Note)
        .node("start", mark("start"))
        .node("gate", mark("gate"), pause=pause)
        .edge("start", "gate")
        .end("gate")
    )


def sending() -> Flow[Note]:
    spec = registry.get("send")  # a ToolSpec: the default executor has no registry
    return Flow("sending", state=Note).node("ask", Call(spec, args={"text": "hi"}, into="text"))


def app_for(flow: Flow[Any], executor: Executor | None = None, **compile: Any) -> Any:
    sending_flow = flow if flow.name != "sending" else flow.end("ask")
    graph = compile_flow(sending_flow, executor=executor)
    return graph.compile(checkpointer=InMemorySaver(), **compile)


@pytest.fixture(autouse=True)
def _clean():
    SENT.clear()
    CLAIMS.clear()


def test_compile_flow_returns_a_state_graph_with_reserved_channels():
    graph = compile_flow(gated())
    assert isinstance(graph, StateGraph)
    assert set(graph.nodes) == {"start", "gate"}
    assert {"text", "log", "aa_visits", "aa_loop_counts"} <= set(graph.channels)
    assert graph.aa_flow.gated == ("gate",)
    app = graph.compile(interrupt_before=["start"])
    assert app.interrupt_before_nodes == ["start", "gate"]
    assert graph.compile(interrupt_before="*").interrupt_before_nodes == "*"
    assert INTEGRATION.compile_flow(gated()).aa_flow.graph.name == "gated"


def test_compile_flow_refuses_what_it_cannot_compile():
    with pytest.raises(ToolUsageError, match="Flow"):
        compile_flow("not a flow")  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="Executor"):
        compile_flow(gated(), executor=FakeRunner())  # type: ignore[arg-type]
    with pytest.raises(FlowDefinitionError):
        compile_flow(Flow("empty", state=Note))
    reserved = Flow("reserved", state=Note).node("__end__", mark("x")).end("__end__")
    with pytest.raises(FlowCompileError, match="reserved by LangGraph") as caught:
        compile_flow(reserved)
    assert caught.value.target == "langgraph"
    with pytest.raises(ToolUsageError, match="only checkpoint_key"):
        INTEGRATION.compile_flow(gated(), name="x")


async def test_run_compiled_needs_a_compiled_flow_and_a_valid_state():
    with pytest.raises(ToolUsageError, match="compile_flow"):
        await run_compiled(object(), {}, config())
    app = app_for(gated(), Executor(registry, approval=AutoApprove()))
    with pytest.raises(ToolUsageError, match="state must be"):
        await run_compiled(app, 42, config())
    with pytest.raises(ToolUsageError, match="config must be a mapping"):
        await run_compiled(app, {}, None)  # type: ignore[arg-type]
    with pytest.raises(FlowError, match="initial state"):
        await run_compiled(app, {"log": "not a list"}, config())


async def test_without_a_checkpointer_a_flow_without_pauses_runs_with_its_own_run_id():
    flow = Flow("plain", state=Note).node("a", mark("a")).node("b", mark("b", text="x"))
    app = compile_flow(flow.edge("a", "b").end("b")).compile()
    result = await run_compiled(app, Note(), {})
    assert (result.status, result.steps, dict(result.visits)) == ("done", 2, {"a": 1, "b": 1})
    assert result.state == Note(text="x", log=["a", "b"])
    with pytest.raises(ToolUsageError, match="thread_id"):
        await app.ainvoke({"text": ""})  # a compiled node needs a run id
    with pytest.raises(ToolUsageError, match="checkpointer"):
        await run_compiled(app, ApprovalDecision(approved=True), {})


async def test_a_gate_whose_port_decides_at_once_does_not_stop_the_run():
    sink = RecordingSink()
    approve = Executor(registry, approval=AutoApprove(), sink=sink)
    result = await run_compiled(app_for(gated(), approve), {}, config())
    assert (result.status, result.state.log) == ("done", ["start", "gate"])
    assert [e.type for e in sink.events if e.type.startswith("flow.")] == [
        "flow.started",
        "flow.finished",
    ]
    asked: list[str] = []

    def callback(request: Any) -> bool:
        asked.append(request.tool)
        return True

    calls = Executor(registry, approval=CallbackApproval(callback))
    assert (await run_compiled(app_for(gated(), calls), {}, config())).status == "done"
    assert asked == ["node:gated.gate", "node:gated.gate"]  # a port without a store: asked twice
    denied = await run_compiled(app_for(gated(), Executor(approval=AutoDeny())), {}, config())
    assert (denied.status, denied.denied_at, denied.state.log) == ("denied", "gate", ["start"])
    no_port = await run_compiled(app_for(gated(), Executor()), {}, config())
    assert (no_port.status, no_port.denied_at) == ("denied", "gate")


async def test_a_langgraph_approval_gate_pauses_before_the_superstep_and_resumes_bound():
    app = app_for(gated(payload=True))  # the default executor: LangGraphApproval
    cfg = config()
    paused = await run_compiled(app, {"text": "draft"}, cfg)
    assert paused.status == "paused" and paused.pending.tool == "node:gated.gate"
    assert paused.pending.arguments == {"text": "draft"}
    done = await run_compiled(app, ApprovalDecision(approved=True, edited_args={"text": "e"}), cfg)
    assert done.status == "done" and done.state == Note(text="e", log=["start", "gate"])
    again = await run_compiled(app, {"text": "draft"}, cfg2 := config())
    assert again.status == "paused"
    denied = await run_compiled(app, ApprovalDecision(approved=False, reason="no"), cfg2)
    assert (denied.status, denied.denied_at, denied.state.log) == ("denied", "gate", ["start"])


async def test_a_direct_denial_of_a_langgraph_gate_raises_in_the_node_and_reports_denied():
    app = app_for(gated())
    cfg = config()
    await run_compiled(app, {}, cfg)
    request = (await run_compiled(app, None, cfg)).pending  # continuing re-asks: still paused
    assert request is not None and request.kind == "node"
    first = await app.ainvoke(None, cfg, version="v2")  # the gated node interrupts itself
    [interrupt] = first.interrupts
    denial = Command(resume={interrupt.id: bound(interrupt, ApprovalDecision(approved=False))})
    result = await run_compiled(app, denial, cfg)
    assert (result.status, result.denied_at) == ("denied", "gate")


async def test_forged_or_foreign_resume_values_are_refused_without_running_the_tool():
    app = app_for(sending())
    cfg, other = config(), config()
    paused = await run_compiled(app, {}, cfg)
    assert paused.pending.tool == "send" and paused.pending.call_id == "sending:ask:1:0"
    await run_compiled(app, {}, other)
    [interrupt] = (await app.aget_state(cfg)).interrupts
    [foreign] = (await app.aget_state(other)).interrupts
    assert interrupt.value["binding"] != foreign.value["binding"]  # bound to the run
    for value in (
        {"binding": foreign.value["binding"], "decision": {"approved": True}},
        {"approved": True},
        {"binding": "0" * 64, "decision": {"approved": True}},
    ):
        with pytest.raises(ValidationError):
            await run_compiled(app, Command(resume={interrupt.id: value}), cfg)
    assert SENT == []
    done = await run_compiled(app, ApprovalDecision(approved=True), cfg)
    assert done.state.text == "sent hi" and SENT == ["hi"]
    with pytest.raises(FlowError, match="nothing waits"):
        await run_compiled(app, ApprovalDecision(approved=True), cfg)


async def test_a_store_decision_made_out_of_band_resumes_with_none():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    app = app_for(sending(), executor)
    cfg = config()
    paused = await run_compiled(app, {}, cfg)
    await store.decide(paused.pending.approval_id, ApprovalDecision(approved=True))
    done = await run_compiled(app, None, cfg)
    assert done.status == "done" and SENT == ["hi"]
    assert (await store.get(paused.pending.approval_id)).consumed


async def test_a_store_backed_gate_decided_otherwise_fails_closed():
    store = MemoryApprovalStore()
    app = app_for(gated(), Executor(approval=DeferredApproval(store)))
    cfg = config()
    paused = await run_compiled(app, {}, cfg)
    await store.decide(paused.pending.approval_id, ApprovalDecision(approved=False))
    with pytest.raises(ApprovalAlreadyDecided):
        await run_compiled(app, ApprovalDecision(approved=True), cfg)
    denied = await run_compiled(app, None, cfg)
    assert (denied.status, denied.denied_at) == ("denied", "gate")


async def test_two_store_backed_gates_hold_their_superstep_until_both_are_decided():
    flow = (
        Flow("two_gates", state=Note)
        .node("start", mark("start"))
        .node("left", mark("left"), pause=True)
        .node("right", mark("right"), pause=True)
        .node("side", mark("side"))
        .edge("start", ["left", "right", "side"])
        .end("left", "right", "side")
    )
    store = MemoryApprovalStore()
    app = app_for(flow, Executor(approval=DeferredApproval(store)))
    cfg = config()
    first = await run_compiled(app, {}, cfg)
    assert first.pending.tool == "node:two_gates.left"
    second = await run_compiled(app, ApprovalDecision(approved=True), cfg)
    assert second.status == "paused" and second.pending.tool == "node:two_gates.right"
    assert second.state.log == ["start"]  # nothing of the superstep ran
    done = await run_compiled(app, ApprovalDecision(approved=True), cfg)
    assert sorted(done.state.log) == ["left", "right", "side", "start"]
    assert not await store.list(status="pending")


async def test_recursion_limit_is_max_steps_and_names_the_next_node():
    class Count(BaseModel):
        n: int = 0

    flow = (
        Flow("spin", state=Count)
        .node("tick", lambda s: {"n": s.n + 1})
        .route("tick", Route(lambda s: "tick", ["tick"]), max_loops=1000)
        .end()
    )
    app = compile_flow(flow, executor=Executor()).compile(checkpointer=InMemorySaver())
    with pytest.raises(FlowLoopExceeded) as caught:
        await run_compiled(app, {}, config(recursion_limit=5))
    assert (caught.value.node, caught.value.limit) == ("tick", 5)
    plain = compile_flow(flow, executor=Executor()).compile()
    with pytest.raises(FlowLoopExceeded):
        await run_compiled(plain, {}, {"recursion_limit": 3})


async def test_a_route_past_max_loops_without_on_exhausted_raises():
    class Count(BaseModel):
        n: int = 0

    flow = (
        Flow("loopy", state=Count)
        .node("tick", lambda s: {"n": s.n + 1})
        .route("tick", Route(lambda s: "tick", ["tick"]), max_loops=2)
        .end()
    )
    app = compile_flow(flow, executor=Executor()).compile(checkpointer=InMemorySaver())
    with pytest.raises(FlowLoopExceeded):
        await run_compiled(app, {}, config())


async def test_invalid_gate_edits_and_unknown_update_keys_are_flow_errors():
    app = app_for(gated(), Executor(approval=LangGraphApproval()))
    cfg = config()
    await run_compiled(app, {}, cfg)
    with pytest.raises(FlowError, match="edited state is invalid"):
        await run_compiled(app, ApprovalDecision(approved=True, edited_args={"log": 3}), cfg)
    bad = Flow("bad", state=Note).node("a", lambda s: {"nope": 1}).end("a")
    with pytest.raises(FlowError, match="unknown state fields"):
        await run_compiled(compile_flow(bad, executor=Executor()).compile(), {}, {})


async def test_a_paused_interrupt_that_is_not_an_approval_pauses_without_a_request():
    from langgraph.types import interrupt

    def asks(state: Note) -> dict[str, Any]:
        interrupt("free text")
        return {}

    flow = Flow("asks", state=Note).node("q", asks).end("q")
    app = compile_flow(flow, executor=Executor()).compile(checkpointer=InMemorySaver())
    result = await run_compiled(app, {}, config())
    assert result.status == "paused" and result.pending is None


async def test_several_gates_in_one_superstep_need_a_store_backed_port():
    flow = (
        Flow("gates", state=Note)
        .node("start", mark("start"))
        .node("left", mark("left"), pause=True)
        .node("right", mark("right"), pause=True)
        .edge("start", ["left", "right"])
        .end("left", "right")
    )
    app = app_for(flow, Executor(approval=LangGraphApproval()))
    cfg = config()
    paused = await run_compiled(app, {}, cfg)
    assert paused.pending.tool == "node:gates.left"
    for answer in (ApprovalDecision(approved=True), None):
        with pytest.raises(ToolUsageError, match="several gates in one superstep"):
            await run_compiled(app, answer, cfg)
    assert (await app.aget_state(cfg)).values["log"] == ["start"]  # nothing of it ran


def test_compile_flow_refuses_a_subflow_node():
    from tests.flow.subflow_helpers import child_flow, parent_flow

    with pytest.raises(FlowCompileError, match="SubFlow") as caught:
        compile_flow(parent_flow(child_flow()), executor=Executor(registry))
    assert caught.value.target == "langgraph"
