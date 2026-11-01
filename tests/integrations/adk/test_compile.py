"""`compile_flow` / `run_workflow`: shape, errors, events, and replays that never re-run work."""

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    FlowCompileError,
    FlowDefinitionError,
    FlowError,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, FlowRunner, Route, node_model
from agent_adapter.integrations import adk
from agent_adapter.integrations.adk._run import _visits
from agent_adapter.integrations.adk.compile import _flow_plan
from agent_adapter.integrations.base import FlowCompiler
from tests.fakes.models import ScriptedModel
from tests.fakes.sinks import RecordingLedger, RecordingSink
from tests.flow.parity_flows import Board, mark, review_flow
from tests.integrations.kit import FakeRunner

pytestmark = pytest.mark.contract

APPROVE = ApprovalDecision(approved=True)
CALLS: list[str] = []
registry = Registry()


@registry.tool(parallel_safe=True)
async def count_up(text: str) -> str:
    """A tool that records each real run."""
    CALLS.append(text)
    return f"counted {text}"


@registry.tool(parallel_safe=True, confirm=True)
async def publish(text: str) -> str:
    """Publish (needs approval)."""
    CALLS.append(f"publish {text}")
    return f"published {text}"


@pytest.fixture(autouse=True)
def fresh():
    CALLS.clear()


def executor(**options: Any) -> Executor:
    store = MemoryApprovalStore()
    return Executor(registry, approval=DeferredApproval(store), **options)


def side_flow() -> Flow[Board]:
    """A tool pause beside a sibling that calls a tool: the sibling's call runs once."""
    return (
        Flow("side", state=Board)
        .node("start", mark("start"))
        .node("ask", Call("publish", args={"text": "hello"}, into="sent"))
        .node("side", Call("count_up", args={"text": "side"}, into="note"))
        .edge("start", ["ask", "side"])
        .end("ask", "side")
    )


async def pause_and_resume(workflow, sessions, decision=APPROVE):
    paused = await adk.run_workflow(workflow, {}, session_service=sessions)
    assert paused.status == "paused"
    message = adk.resume_message(paused.pending.metadata["interrupt_id"], decision)
    return paused, await adk.run_workflow(workflow, message, session_service=sessions)


def test_compile_flow_builds_a_one_driver_workflow():
    workflow = adk.compile_flow(review_flow, executor=executor())
    assert isinstance(workflow, Workflow) and workflow.name == "review"
    assert [n.name for n in workflow.graph.nodes] == ["__START__", "aa_driver"]
    named = adk.compile_flow(review_flow, executor=executor(), name="reviews")
    assert named.name == "reviews"
    assert _flow_plan(named).graph.name == "review"
    assert isinstance(adk.INTEGRATION, FlowCompiler)
    assert adk.INTEGRATION.compile_flow(review_flow, executor=executor()).name == "review"


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"executor": FakeRunner()}, FlowCompileError),
        ({"name": "not an identifier"}, FlowCompileError),
        ({"max_steps": 0}, ToolUsageError),
        ({"max_steps": True}, ToolUsageError),
    ],
)
def test_compile_flow_refuses_what_cannot_run(options, error):
    with pytest.raises(error):
        adk.compile_flow(review_flow, **{"executor": executor(), **options})


def test_an_invalid_flow_is_refused_before_compiling():
    broken = Flow("broken", state=Board).node("a", mark("a")).edge("a", "missing")
    with pytest.raises(FlowDefinitionError):
        adk.compile_flow(broken, executor=executor())


async def test_run_workflow_refuses_foreign_workflows_and_bad_inputs():
    from google.adk.workflow import node

    foreign = Workflow(name="foreign", edges=[(START, node(lambda: 1, name="one"))])
    with pytest.raises(ToolUsageError):
        await adk.run_workflow(foreign, {})
    workflow = adk.compile_flow(review_flow, executor=executor())
    with pytest.raises(ToolUsageError):
        await adk.run_workflow(workflow, ["not", "a", "state"])
    with pytest.raises(FlowError, match="initial state"):
        await adk.run_workflow(workflow, {"topic": 3})
    with pytest.raises(ToolUsageError, match="no interrupt"):
        await adk.run_workflow(workflow, adk.resume_message("adk-other", APPROVE))


async def test_a_plain_adk_runner_takes_the_state_as_json():
    from google.genai import types

    flow = Flow("plain", state=Board).node("only", mark("only", note="hi")).end("only")
    workflow = adk.compile_flow(flow, executor=executor())
    sessions = InMemorySessionService()
    runner = Runner(app_name="plain", node=workflow, session_service=sessions)

    async def turn(text: str) -> list[Any]:
        session = await sessions.create_session(app_name="plain", user_id="u")
        message = types.Content(role="user", parts=[types.Part(text=text)])
        run = runner.run_async(user_id="u", session_id=session.id, new_message=message)
        return [e async for e in run]

    events = await turn(json.dumps({"flag": "x"}))
    (output,) = [e.output for e in events if e.output and "status" in e.output]
    assert output["status"] == "done" and output["state"]["note"] == "hi"
    assert output["state"]["flag"] == "x" and output["visits"] == {"only": 1}
    for bad in ("not json", "[1, 2]"):
        with pytest.raises(FlowError, match="initial state"):
            await turn(bad)


async def test_a_state_model_instance_starts_a_run():
    flow = Flow("model", state=Board).node("only", mark("only")).end("only")
    workflow = adk.compile_flow(flow, executor=executor())
    result = await adk.run_workflow(workflow, Board(flag="set"))
    assert result.status == "done" and result.state.flag == "set" and result.checkpoint is None


async def test_node_errors_reach_the_caller_unchanged():
    def explode(state: Board) -> dict[str, Any]:
        raise ValueError("boom")

    flow = Flow("explode", state=Board).node("bad", explode).end("bad")
    sink = RecordingSink()
    workflow = adk.compile_flow(flow, executor=executor(sink=sink))
    with pytest.raises(FlowError) as raised:
        await adk.run_workflow(workflow, {})
    assert raised.value.node == "bad" and "ValueError" in str(raised.value)
    finished = sink.of_type("flow.finished")
    assert [e.data["status"] for e in finished] == ["failed"]
    assert [e.data["ok"] for e in sink.of_type("run.finished")] == [False]


async def test_replays_emit_node_events_and_ledger_entries_once():
    sink, ledger = RecordingSink(), RecordingLedger()
    workflow = adk.compile_flow(side_flow(), executor=executor(sink=sink, ledger=ledger))
    _, final = await pause_and_resume(workflow, InMemorySessionService())
    assert final.status == "done" and final.state.note == "counted side"
    assert final.state.sent == "published hello"
    assert CALLS == ["side", "publish hello"]  # the sibling's tool call ran once
    assert sorted(e.stage for e in ledger.entries()) == ["count_up", "publish"]
    assert len(sink.of_type("run.started")) == 1 and len(sink.of_type("run.finished")) == 1
    assert [e.data["status"] for e in sink.of_type("flow.finished")] == ["paused", "done"]
    started = [e.data["node"] for e in sink.of_type("node.started")]
    assert sorted(started) == ["ask", "ask", "side", "start"]  # ask: first try + resumed run


async def test_routes_are_asked_once_per_real_visit():
    asked: list[str] = []

    def choose(state: Board) -> str:
        asked.append(state.flag)
        return "gate"

    flow = (
        Flow("routed", state=Board)
        .node("pick", mark("pick", flag="p"))
        .node("gate", mark("gate"), pause=True)
        .route("pick", Route(choose, targets=["gate"]))
        .end("gate")
    )
    workflow = adk.compile_flow(flow, executor=executor())
    _, final = await pause_and_resume(workflow, InMemorySessionService())
    assert final.status == "done" and final.state.log == ["pick", "gate"]
    assert asked == ["p"]  # the resumed driver read the stored label


async def test_nodes_see_the_adk_context_ids_and_the_model():
    seen: dict[str, Any] = {}

    async def look(state: Board, ctx: ToolContext) -> dict[str, Any]:
        seen.update(ctx=ctx, model=node_model(ctx))
        return {"note": "seen"}

    flow = Flow("look", state=Board).node("look", look).end("look")
    model = ScriptedModel(["hi"])
    workflow = adk.compile_flow(flow, executor=executor(), model=model)
    sessions = InMemorySessionService()
    result = await adk.run_workflow(workflow, {}, session_service=sessions)
    ctx = seen["ctx"]
    adk_ctx = ctx.extras["adk"]
    assert result.status == "done" and seen["model"] is not None
    assert ctx.run_id == adk_ctx.invocation_id and ctx.thread_id == adk_ctx.session.id
    assert ctx.call_id == "look:look:1"


class Opaque:
    """A class with no canonical encoding."""


class Loose(BaseModel):
    thing: Any = None


async def test_a_node_output_adk_cannot_store_fails_closed():
    flow = Flow("opaque", state=Loose).node("make", lambda s: {"thing": Opaque()}).end("make")
    workflow = adk.compile_flow(flow, executor=executor())
    with pytest.raises(FlowError, match="state-not-sealable"):
        await adk.run_workflow(workflow, {})
    assert (await FlowRunner(executor()).run(flow, {})).status == "done"  # local keeps it


async def test_a_pause_withdraws_the_records_its_resume_asks_afresh():
    run_executor = executor()
    store = run_executor.approval.store

    async def both(state: Board, ctx: ToolContext) -> dict[str, Any]:
        sent = await asyncio.gather(
            run_executor.run("publish", {"text": "a"}, ctx),
            run_executor.run("publish", {"text": "b"}, ctx),
        )
        return {"note": " | ".join(sent)}

    flow = Flow("both", state=Board).node("both", both).end("both")
    workflow = adk.compile_flow(flow, executor=run_executor)
    sessions = InMemorySessionService()
    result = await adk.run_workflow(workflow, {}, session_service=sessions)
    first = result.pending.approval_id
    [superseded] = [r for r in await store.list() if r.approval_id != first]
    assert superseded.status == "denied" and superseded.consumed
    assert superseded.decision.reason == "superseded"
    resumes = 0
    while result.status == "paused":
        message = adk.resume_message(result.pending.metadata["interrupt_id"], APPROVE)
        result = await adk.run_workflow(workflow, message, session_service=sessions)
        resumes += 1
    assert result.state.note == "published a | published b" and resumes == 2
    assert sorted(CALLS) == ["publish a", "publish b"]
    assert await store.list(status="pending") == []
    assert all(record.consumed for record in await store.list())


@pytest.mark.filterwarnings("ignore:.*ResumabilityConfig.*:UserWarning")
async def test_a_resumable_app_replays_outputs_and_visits_count_each_path_once():
    from google.adk.apps import App
    from google.adk.apps.app import ResumabilityConfig
    from google.genai import types

    workflow = adk.compile_flow(side_flow(), executor=executor())
    app = App(
        name="side", root_agent=workflow, resumability_config=ResumabilityConfig(is_resumable=True)
    )
    sessions = InMemorySessionService()
    runner = Runner(app=app, session_service=sessions)
    session = await sessions.create_session(app_name="side", user_id="u")

    async def turn(message: Any) -> list[Any]:
        run = runner.run_async(user_id="u", session_id=session.id, new_message=message)
        return [e async for e in run]

    first = await turn(types.Content(role="user", parts=[types.Part(text="{}")]))
    (call,) = [c for e in first for c in e.get_function_calls() if c.name == "adk_request_input"]
    await turn(adk.resume_message(call.id, APPROVE))
    stored = await sessions.get_session(app_name="side", user_id="u", session_id=session.id)
    assert _visits(_flow_plan(workflow), stored.events) == {"start": 1, "ask": 1, "side": 1}
    assert CALLS == ["side", "publish hello"]


async def test_a_gate_the_port_decides_at_once_runs_without_pausing():
    from agent_adapter.approval import AutoApprove

    flow = Flow("auto", state=Board).node("go", mark("go"), pause=True).end("go")
    workflow = adk.compile_flow(flow, executor=Executor(registry, approval=AutoApprove()))
    result = await adk.run_workflow(workflow, {})
    assert result.status == "done" and result.state.log == ["go"]


async def test_a_failing_route_is_a_flow_error():
    def broken(state: Board) -> str:
        raise KeyError("label")

    flow = (
        Flow("bad_route", state=Board)
        .node("pick", mark("pick"))
        .node("next", mark("next"))
        .route("pick", Route(broken, targets=["next"]))
        .end("next")
    )
    with pytest.raises(FlowError, match="route from 'pick' failed"):
        await adk.run_workflow(adk.compile_flow(flow, executor=executor()), {})


@registry.tool(parallel_safe=True)
async def opaque() -> Any:
    """A result with no canonical encoding."""
    return Opaque()


async def test_a_pause_after_an_unsealable_call_is_refused_and_withdrawn():
    run_executor = executor()

    async def go(state: Board, ctx: ToolContext) -> dict[str, Any]:
        await run_executor.run("opaque", {}, ctx)
        return {"sent": await run_executor.run("publish", {"text": "x"}, ctx)}

    flow = Flow("unsealable", state=Board).node("go", go).end("go")
    with pytest.raises(FlowError, match="state-not-sealable"):
        await adk.run_workflow(adk.compile_flow(flow, executor=run_executor), {})
    (record,) = await run_executor.approval.store.list()
    assert record.status == "denied" and record.consumed and CALLS == []


async def test_a_node_raising_cancelled_error_ends_the_run_as_cancelled():
    async def stop(state: Board) -> dict[str, Any]:
        raise asyncio.CancelledError

    flow = Flow("halt", state=Board).node("stop", stop).end("stop")
    sink = RecordingSink()
    workflow = adk.compile_flow(flow, executor=executor(sink=sink))
    with pytest.raises(FlowError, match="cancelled inside ADK"):
        await adk.run_workflow(workflow, {})
    assert [e.data["ok"] for e in sink.of_type("run.finished")] == [False]
    assert [e.data["status"] for e in sink.of_type("flow.finished")] == ["failed"]
