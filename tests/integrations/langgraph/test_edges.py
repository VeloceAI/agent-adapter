"""Edge cases of the LangGraph package: memo bounds, confirm rules, odd inputs, port failures."""

import json
from typing import Annotated, Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalDenied,
    FlowError,
    ToolUsageError,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, ToolCall
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow, Merge
from agent_adapter.integrations.langgraph import (
    LangGraphApproval,
    compile_flow,
    run_compiled,
    tool_node,
    tool_task,
)
from agent_adapter.integrations.langgraph._context import call_context, task_namespace
from agent_adapter.integrations.langgraph._gate import GateAnswer, ask_gate, spend
from agent_adapter.integrations.langgraph._memo import TaskMemo
from agent_adapter.integrations.langgraph.approval import defer_to_node
from agent_adapter.integrations.langgraph.tools import _call_digest, _content, _needs_approval
from agent_adapter.runtime.current import use_context
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

registry = Registry()


@registry.tool(confirm=lambda call, ctx: call.arguments.get("item") == "risky")
async def change(item: str) -> str:
    """Change an item; risky ones need approval."""
    return f"changed {item}"


@registry.tool(confirm=lambda call, ctx: 1 / 0)
async def broken_rule(item: str) -> str:
    """A confirm rule that fails."""
    return item


@registry.tool(confirm=True)
async def send(text: str) -> str:
    """Send a text (needs approval)."""
    return f"sent {text}"


def test_task_memo_evicts_the_oldest_and_expires_by_age():
    now = [0.0]
    memo: TaskMemo[int] = TaskMemo(max_entries=2, ttl=10.0, clock=lambda: now[0])
    memo.put(("a",), 1)
    memo.put(("b",), 2)
    memo.put(("c",), 3)
    assert (memo.get(("a",)), memo.get(("c",)), len(memo)) == (None, 3, 2)
    now[0] = 11.0
    assert memo.get(("c",)) is None and len(memo) == 0


async def test_confirm_rules_order_calls():
    executor = Executor(registry, approval=LangGraphApproval())
    app = one_node(tool_node(registry, executor=executor))
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "change", "args": {"item": "risky"}, "id": "c1"},
            {"name": "change", "args": {"item": "safe"}, "id": "c2"},
        ],
    )
    cfg = config()
    out = await app.ainvoke({"messages": [message]}, cfg, version="v2")
    [first] = out.interrupts
    assert first.value["request"]["call_id"] == "c1"  # the safe call already ran
    out = await app.ainvoke(resume(first, ApprovalDecision(approved=True)), cfg, version="v2")
    assert [m.content for m in out.value["messages"][1:]] == ["changed risky", "changed safe"]


async def test_a_failing_confirm_rule_orders_the_call_last_and_the_executor_raises_it():
    spec = registry.get("broken_rule")
    ctx = ToolContext(run_id="r", call_id="c")
    assert await _needs_approval(spec, {"name": "broken_rule", "args": {}}, ctx) is True
    app = one_node(tool_node(registry, executor=Executor(registry)))
    message = AIMessage(
        content="", tool_calls=[{"name": "broken_rule", "args": {"item": "x"}, "id": "c3"}]
    )
    with pytest.raises(ZeroDivisionError):
        await app.ainvoke({"messages": [message]}, config())


async def test_calls_without_an_id_get_one_from_the_task():
    app = one_node(tool_node(registry, executor=Executor(registry)))
    message = AIMessage(
        content="", tool_calls=[{"name": "change", "args": {"item": "a"}, "id": ""}]
    )
    out = await app.ainvoke({"messages": [message]}, config())
    assert out["messages"][-1].tool_call_id.endswith(":0")


def test_odd_values_still_get_a_digest_and_content():
    assert _call_digest("x", {"raw": b"1"}).startswith("('x'")
    assert _content({"s": {1, 2}}).startswith("{'s'")
    assert json.loads(_content([1, "a"])) == [1, "a"]
    assert task_namespace("not a config") == ""
    node, task = tool_node(registry), tool_task(registry.get("send"), into="out")
    assert repr(node) == "ToolNode(tools=['change', 'broken_rule', 'send'])"
    assert repr(task) == "ToolTask('send', into='out')" and repr(LangGraphApproval())


async def test_a_call_inside_an_adapter_run_joins_that_run():
    outer = ToolContext(run_id="outer-run", call_id="parent", extras={"k": 1})
    with use_context(outer):
        ctx = call_context(config("ignored"), {"id": "c7"})
    assert (ctx.run_id, ctx.call_id, ctx.extras["k"]) == ("outer-run", "c7", 1)


async def test_deferring_outside_a_graph_still_refuses():
    request = ApprovalRequest(approval_id="a", run_id="r", call_id="c", tool="t")
    with defer_to_node("some:ns"), pytest.raises(ToolUsageError):
        await LangGraphApproval().request(request)


class _Raising:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        raise self.exc


async def test_a_gate_port_that_fails_is_a_flow_error_and_a_denial_is_a_denial():
    request = ApprovalRequest(
        approval_id="a", run_id="r", call_id="c", tool="node:f.n", kind="node"
    )
    with pytest.raises(FlowError, match="approval failed"):
        await ask_gate("f", "n", request, _Raising(RuntimeError("down")), peek=False)
    denied = await ask_gate("f", "n", request, _Raising(ApprovalDenied("t", reason="x")), peek=True)
    assert denied.decision is not None and not denied.approved and denied.decision.reason == "x"


async def test_spending_a_denial_twice_is_harmless():
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    request = ApprovalRequest(
        approval_id="a", run_id="r", call_id="c", tool="node:f.n", kind="node"
    )
    answer = await ask_gate("f", "n", request, port, peek=True)
    await store.decide(answer.pending.approval_id, ApprovalDecision(approved=False))
    peeked = await ask_gate("f", "n", request, port, peek=True)
    await spend(peeked, port)
    await spend(peeked, port)  # a concurrent spend already used it
    await spend(GateAnswer(), port)
    assert (await store.get(answer.pending.approval_id)).consumed


class Note(BaseModel):
    text: str = ""
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)


async def test_a_user_interrupt_before_a_plain_node_pauses_and_continues():
    flow = Flow("plain", state=Note).node("a", lambda s: {"log": ["a"]}).end("a")
    graph = compile_flow(flow, executor=Executor())
    app = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["a"])
    cfg = config()
    paused = await run_compiled(app, {}, cfg)
    assert (paused.status, paused.pending) == ("paused", None)
    done = await run_compiled(app, None, cfg)
    assert done.status == "done" and done.state.log == ["a"]


def _gated() -> Flow[Note]:
    return (
        Flow("gated", state=Note).node("gate", lambda s: {"log": ["gate"]}, pause=True).end("gate")
    )


async def test_a_gate_denied_out_of_band_is_denied_when_resumed_with_that_denial():
    store = MemoryApprovalStore()
    app = compile_flow(_gated(), executor=Executor(approval=DeferredApproval(store)))
    app = app.compile(checkpointer=InMemorySaver())
    cfg = config()
    paused = await run_compiled(app, {}, cfg)
    denial = ApprovalDecision(approved=False, reason="no")
    await store.decide(paused.pending.approval_id, denial)
    result = await run_compiled(app, denial, cfg)
    assert (result.status, result.denied_at) == ("denied", "gate")
    assert (await store.get(paused.pending.approval_id)).consumed


class _RacyStore(MemoryApprovalStore):
    """Someone decides the record between our look and our decision."""

    def __init__(self, other: ApprovalDecision) -> None:
        super().__init__()
        self.other = other

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> Any:
        record = await self.get(approval_id)
        if record.status == "pending":
            await super().decide(approval_id, self.other)
            raise ApprovalAlreadyDecided(approval_id, status="decided")
        return await super().decide(approval_id, decision)


@pytest.mark.parametrize("same", [True, False])
async def test_a_gate_decided_by_someone_else_meanwhile_must_agree(same):
    approve = ApprovalDecision(approved=True)
    store = _RacyStore(approve if same else ApprovalDecision(approved=False))
    app = compile_flow(_gated(), executor=Executor(approval=DeferredApproval(store)))
    app = app.compile(checkpointer=InMemorySaver())
    cfg = config()
    await run_compiled(app, {}, cfg)
    if same:
        assert (await run_compiled(app, approve, cfg)).status == "done"
    else:
        with pytest.raises(ApprovalAlreadyDecided):
            await run_compiled(app, approve, cfg)


async def test_a_function_node_s_other_pending_calls_are_withdrawn_as_superseded():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))

    async def both(state: Note, ctx: ToolContext) -> dict[str, Any]:
        calls = [ToolCall(name="send", arguments={"text": t}) for t in ("a", "b")]
        results = await executor.run_many(calls, ctx)
        return {"log": [r.unwrap() for r in results]}

    flow = Flow("both", state=Note).node("go", both).end("go")
    app = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver())
    cfg = config("both-thread")
    paused = await run_compiled(app, {}, cfg)
    assert paused.pending.arguments == {"text": "a"}
    records = await store.list(run_id="both-thread")
    other = next(r for r in records if r.request.arguments == {"text": "b"})
    assert other.consumed and other.decision.reason == "superseded"
    second = await run_compiled(app, ApprovalDecision(approved=True), cfg)
    assert second.pending.arguments == {"text": "b"}  # asked afresh under a new call id
    done = await run_compiled(app, ApprovalDecision(approved=True), cfg)
    assert done.state.log == ["sent a", "sent b"]
