"""Resume rules: errors spend answers, replays stay positional, missing state fails closed."""

import asyncio
from collections import Counter
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Interrupt
from pydantic import BaseModel, ValidationError

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    FlowError,
    ToolError,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, Merge
from agent_adapter.hooks.base import Hook
from agent_adapter.integrations.langgraph import (
    LangGraphApproval,
    _durable,
    compile_flow,
    fan_out,
    one_use,
    reducer,
    run_compiled,
    tool_node,
    tool_task,
)
from agent_adapter.integrations.langgraph._spent import CLAIMS
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

RAN: list[str] = []
registry = Registry()
APPROVE = ApprovalDecision(approved=True)


@registry.tool(confirm=True)
async def refuse(item: str) -> str:
    """Do something (needs approval), then refuse."""
    RAN.append(f"refuse {item}")
    raise ToolError(f"no {item}", tool="refuse")


@registry.tool(confirm=True)
async def crash(item: str) -> str:
    """Do something (needs approval), then crash."""
    RAN.append(f"crash {item}")
    raise RuntimeError(item)


@registry.tool(confirm=True)
async def stamp(text: str) -> str:
    """Stamp a text (needs approval)."""
    RAN.append(f"stamp {text}")
    return f"{text}#{RAN.count(f'stamp {text}')}"


@pytest.fixture(autouse=True)
def _clean():
    RAN.clear()
    CLAIMS.clear()


class Note(BaseModel):
    text: str = ""


class Fan(TypedDict, total=False):
    items: list[dict[str, Any]]
    results: Annotated[list[Any], reducer(Merge.append)]


async def test_a_send_task_whose_approved_tool_failed_visibly_is_not_run_again_on_replay():
    executor = Executor(registry, approval=LangGraphApproval())
    graph = StateGraph(Fan)
    graph.add_node(tool_task(registry.get("refuse"), executor=executor, into="results"))
    graph.add_conditional_edges(START, fan_out("refuse", "items"), ["refuse"])
    graph.add_edge("refuse", END)
    app, cfg = graph.compile(checkpointer=InMemorySaver()), config()
    out = await app.ainvoke({"items": [{"item": "x"}], "results": []}, cfg, version="v2")
    [interrupt] = out.interrupts
    paused = (await app.aget_state(cfg)).config
    out = await app.ainvoke(resume(interrupt, APPROVE), cfg, version="v2")
    assert out.value["results"][0]["error"]["kind"] == "tool-error"
    with pytest.raises(ApprovalAlreadyDecided):
        await app.ainvoke(resume(interrupt, APPROVE), paused)
    assert RAN == ["refuse x"]


async def test_an_approved_tool_that_failed_is_not_run_again_when_the_host_retries():
    flow = Flow("crashy", state=Note).node("go", Call("crash", args={"item": "x"})).end("go")
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver()), config()
    await run_compiled(app, {}, cfg)
    with pytest.raises(FlowError, match="crash"):
        await run_compiled(app, APPROVE, cfg)
    with pytest.raises(ApprovalAlreadyDecided):
        await app.ainvoke(None, cfg)  # the retry LangGraph offers after a failed task
    assert RAN == ["crash x"]


def _request(name: str) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"apr_{name}", run_id="user-run", call_id=f"call_{name}", tool=name
    )


def _two_approvals(counts: Counter[str]) -> Callable[..., Any]:
    port = LangGraphApproval()

    async def node(state: Any) -> dict[str, Any]:
        if (await port.request(_request("a"))).approved:
            counts["a"] += 1  # LangGraph re-runs your node: code after an answer runs again
        if (await port.request(_request("b"))).approved:
            counts["b"] += 1
        await asyncio.sleep(0.03)  # a duplicate overlaps this
        return {"messages": []}

    return node


async def test_a_users_own_node_wrapped_with_one_use_uses_each_answer_once():
    counts: Counter[str] = Counter()
    app, cfg = one_node(one_use(_two_approvals(counts)), MessagesState), config()
    [a] = (await app.ainvoke({"messages": []}, cfg, version="v2")).interrupts
    [b] = (await app.ainvoke(resume(a, APPROVE), cfg, version="v2")).interrupts
    paused = (await app.aget_state(cfg)).config
    command = resume(b, APPROVE)
    results = await asyncio.gather(
        app.ainvoke(command, cfg), app.ainvoke(command, cfg), return_exceptions=True
    )
    failed = [r for r in results if isinstance(r, BaseException)]
    assert [type(e) for e in failed] == [ApprovalAlreadyDecided]
    for replay in (resume(b, APPROVE), resume(a, APPROVE)):  # the paused checkpoint again
        with pytest.raises(ApprovalAlreadyDecided):
            await app.ainvoke(replay, paused)
    assert counts == {"a": 2, "b": 1}


async def test_langgraph_approval_in_an_unwrapped_node_is_refused_before_it_asks():
    counts: Counter[str] = Counter()
    app, cfg = one_node(_two_approvals(counts), MessagesState), config()
    with pytest.raises(ToolUsageError, match="one_use"):
        await app.ainvoke({"messages": []}, cfg)
    assert counts == {} and not (await app.aget_state(cfg)).interrupts


async def test_one_use_wraps_sync_nodes_and_nodes_taking_config():
    seen: list[str] = []

    def sync_node(state: Any) -> dict[str, Any]:
        seen.append("sync")
        return {"messages": []}

    async def with_config(state: Any, config: Any) -> dict[str, Any]:
        seen.append(config["configurable"]["thread_id"])
        return {"messages": []}

    await one_node(one_use(sync_node)).ainvoke({"messages": []}, config("cfg-a"))
    await one_node(one_use(with_config)).ainvoke({"messages": []}, config("cfg-b"))
    assert seen == ["sync", "cfg-b"] and one_use(sync_node).__name__ == "sync_node"


def _two_stamps() -> tuple[Flow[Note], Executor]:
    executor = Executor(registry, approval=LangGraphApproval())

    async def code(state: Note, ctx: ToolContext) -> dict[str, Any]:
        first = await executor.run("stamp", {"text": "first"}, ctx)
        second = await executor.run("stamp", {"text": "second"}, ctx)
        return {"text": f"{first} {second}"}

    return Flow("stamps", state=Note).node("go", code).end("go"), executor


async def test_a_renamed_saver_key_fails_closed_instead_of_re_running(monkeypatch):
    flow, executor = _two_stamps()
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver()), config()
    await run_compiled(app, {}, cfg)
    await run_compiled(app, APPROVE, cfg)
    monkeypatch.setattr(_durable, "CHECKPOINTER_KEY", "__renamed_by_a_new_langgraph__")
    with pytest.raises(ToolUsageError, match="unsupported LangGraph version"):
        await run_compiled(app, APPROVE, cfg)
    assert RAN == ["stamp first"]


class _Stripping(InMemorySaver):
    """A checkpointer whose saved interrupt values lost their sealed state."""

    strip = False

    async def aget_tuple(self, config: Any) -> Any:
        found = await super().aget_tuple(config)
        if found is None or not self.strip:
            return found
        writes = []
        for task, channel, value in found.pending_writes or ():
            if channel == _durable.INTERRUPT_CHANNEL:
                value = [
                    Interrupt(value={k: v for k, v in i.value.items() if k != "aa_state"}, id=i.id)
                    for i in value
                ]
            writes.append((task, channel, value))
        return found._replace(pending_writes=writes)


async def test_resume_values_without_sealed_state_fail_closed():
    flow, executor = _two_stamps()
    saver = _Stripping()
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=saver), config()
    await run_compiled(app, {}, cfg)
    await run_compiled(app, APPROVE, cfg)
    saver.strip = True
    with pytest.raises(FlowError, match="checkpoint-tampered"):
        await run_compiled(app, APPROVE, cfg)
    assert RAN == ["stamp first"]


def _same_names(extra: bool) -> Flow[Note]:
    flow = Flow("twin", state=Note).node("ask", Call("stamp", args={"text": "t"}, into="text"))
    if extra:
        flow = flow.node("after", lambda s: {}).edge("ask", "after").end("after")
        return flow
    return flow.end("ask")


async def test_an_answer_for_another_graph_on_the_same_thread_is_refused():
    executor = Executor(registry, approval=LangGraphApproval())  # the default process key
    apps = [
        compile_flow(_same_names(extra), executor=executor).compile(checkpointer=InMemorySaver())
        for extra in (False, True)
    ]
    cfg = config("twin-thread")
    for app in apps:
        await run_compiled(app, {}, cfg)
    [first] = (await apps[0].aget_state(cfg)).interrupts
    [second] = (await apps[1].aget_state(cfg)).interrupts
    assert first.value["request"] == second.value["request"] | {
        "approval_id": first.value["request"]["approval_id"],
        "created_at": first.value["request"]["created_at"],
    }
    foreign = resume(first, APPROVE).resume[first.id]
    with pytest.raises(ValidationError):
        await apps[1].ainvoke(Command(resume={second.id: foreign}), cfg)
    assert RAN == []


async def test_an_invalid_resume_value_keeps_the_earlier_answers_usable():
    flow, executor = _two_stamps()
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver()), config()
    await run_compiled(app, {}, cfg)
    await run_compiled(app, APPROVE, cfg)
    [second] = (await app.aget_state(cfg)).interrupts
    with pytest.raises(ValidationError):  # nothing is saved: the pause can be answered again
        await app.ainvoke(Command(resume={second.id: {"approved": True}}), cfg)
    done = await run_compiled(app, APPROVE, cfg)
    assert done.state.text == "first#1 second#1"
    assert RAN == ["stamp first", "stamp second"]


class _OnceFailing(Hook):
    """A transient failure in a hook that runs before the tool (only on its second call)."""

    def __init__(self) -> None:
        self.calls = 0

    async def before_tool(self, inv: Any) -> None:
        self.calls += 1
        if self.calls == 2:
            raise ConnectionError("hook backend down")


@pytest.mark.parametrize("backed", [False, True], ids=["langgraph-port", "store"])
async def test_a_transient_failure_before_the_approved_tool_runs_keeps_the_answer(backed):
    store = MemoryApprovalStore() if backed else None
    port = DeferredApproval(store) if store is not None else LangGraphApproval()
    executor = Executor(registry, approval=port, hooks=[_OnceFailing()])
    flow = Flow("hooked", state=Note).node("go", Call("stamp", args={"text": "h"})).end("go")
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver()), config()
    await run_compiled(app, {}, cfg)
    with pytest.raises(FlowError, match=r"hook backend down|ConnectionError"):
        await run_compiled(app, APPROVE, cfg)
    assert RAN == []
    done = await run_compiled(app, None, cfg)  # the host retries
    assert done.status == "done" and RAN == ["stamp h"]


class _OutageStore(MemoryApprovalStore):
    """A store whose first consume fails (an outage while the answer is settled)."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def consume(self, approval_id: str) -> bool:
        if not self.failed:
            self.failed = True
            raise ConnectionError("store down")
        return await super().consume(approval_id)


async def test_a_store_outage_while_settling_keeps_the_answer():
    store = _OutageStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    flow = Flow("outage", state=Note).node("go", Call("stamp", args={"text": "o"})).end("go")
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver()), config()
    await run_compiled(app, {}, cfg)
    with pytest.raises(ConnectionError):
        await run_compiled(app, APPROVE, cfg)
    done = await run_compiled(app, None, cfg)
    assert done.status == "done" and RAN == ["stamp o"]


async def test_first_pauses_do_not_depend_on_langgraphs_interrupt_counter(monkeypatch):
    import langgraph.pregel._algo as algo

    class Counter2:  # the same behaviour under another attribute name
        __slots__ = ("_next",)

        def __init__(self) -> None:
            self._next = 0

        def __call__(self) -> int:
            self._next += 1
            return self._next - 1

    monkeypatch.setattr(algo, "LazyAtomicCounter", Counter2)
    flow = Flow("first", state=Note).node("go", Call("stamp", args={"text": "f"})).end("go")
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver()), config()
    assert (await run_compiled(app, {}, cfg)).status == "paused"
    node = tool_node(registry, executor=executor)
    message = AIMessage(
        content="", tool_calls=[{"name": "stamp", "args": {"text": "t"}, "id": "c"}]
    )
    out = await one_node(node).ainvoke({"messages": [message]}, config(), version="v2")
    assert len(out.interrupts) == 1 and RAN == []
