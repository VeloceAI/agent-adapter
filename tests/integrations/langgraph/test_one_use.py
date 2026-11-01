"""Approvals are one-use: concurrent, replayed and forged resumes never run a tool twice."""

import asyncio
from typing import Annotated, Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, Merge
from agent_adapter.integrations.langgraph import (
    LangGraphApproval,
    compile_flow,
    run_compiled,
    tool_node,
)
from agent_adapter.integrations.langgraph._spent import CLAIMS
from agent_adapter.integrations.langgraph.tools import TOOL_NODE_MEMO
from tests.integrations.langgraph.support import bound, config, one_node, resume

pytestmark = pytest.mark.contract

RAN: list[str] = []
registry = Registry()
APPROVE = ApprovalDecision(approved=True)


@registry.tool(confirm=True)
async def send(text: str) -> str:
    """Send a text (needs approval); slow, so a duplicate overlaps it."""
    RAN.append(text)
    await asyncio.sleep(0.03)
    return f"sent {text}"


class Note(BaseModel):
    text: str = ""
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)


def _flow() -> Flow[Note]:
    return (
        Flow("once", state=Note)
        .node("ask", Call("send", args={"text": "hi"}, into="text"))
        .end("ask")
    )


def _app(store: MemoryApprovalStore | None) -> Any:
    port = DeferredApproval(store) if store is not None else LangGraphApproval()
    executor = Executor(registry, approval=port)
    return compile_flow(_flow(), executor=executor).compile(checkpointer=InMemorySaver())


@pytest.fixture(autouse=True)
def _clean():
    RAN.clear()
    CLAIMS.clear()
    TOOL_NODE_MEMO.clear()


def _split(results: list[Any]) -> tuple[list[Any], list[BaseException]]:
    done = [r for r in results if not isinstance(r, BaseException)]
    failed = [r for r in results if isinstance(r, BaseException)]
    return done, failed


@pytest.mark.parametrize("backed", [False, True], ids=["langgraph-port", "store"])
async def test_two_concurrent_resumes_run_the_tool_once(backed):
    store = MemoryApprovalStore() if backed else None
    app, cfg = _app(store), config()
    await run_compiled(app, {}, cfg)
    results = await asyncio.gather(
        run_compiled(app, APPROVE, cfg), run_compiled(app, APPROVE, cfg), return_exceptions=True
    )
    done, failed = _split(results)
    assert [r.status for r in done] == ["done"]
    assert [type(e) for e in failed] == [ApprovalAlreadyDecided]
    assert RAN == ["hi"]
    if store is not None:
        assert len(await store.list(status="approved")) == 1


async def test_two_concurrent_direct_resumes_with_one_signed_value_run_the_tool_once():
    app, cfg = _app(None), config()
    await run_compiled(app, {}, cfg)
    [interrupt] = (await app.aget_state(cfg)).interrupts
    command = resume(interrupt, APPROVE)
    results = await asyncio.gather(
        app.ainvoke(command, cfg), app.ainvoke(command, cfg), return_exceptions=True
    )
    _, failed = _split(results)
    assert [type(e) for e in failed] == [ApprovalAlreadyDecided]
    assert RAN == ["hi"]


@pytest.mark.parametrize("backed", [False, True], ids=["langgraph-port", "store"])
async def test_replaying_the_paused_checkpoint_after_success_runs_nothing(backed):
    store = MemoryApprovalStore() if backed else None
    app, cfg = _app(store), config()
    await run_compiled(app, {}, cfg)
    snapshot = await app.aget_state(cfg)
    [interrupt] = snapshot.interrupts
    assert (await run_compiled(app, APPROVE, cfg)).status == "done"
    again = await app.ainvoke(None, snapshot.config, version="v2")  # a fork asks afresh
    assert again.interrupts and RAN == ["hi"]
    with pytest.raises(ApprovalAlreadyDecided):  # a resume of the old pause runs the task
        await app.ainvoke(resume(interrupt, APPROVE), snapshot.config)
    assert RAN == ["hi"]
    if store is not None:  # the fork's own request waits for a new decision; none approved
        assert len(await store.list(status="approved")) == 1


async def test_a_resume_value_never_decides_a_store_backed_approval():
    store = MemoryApprovalStore()
    app, cfg = _app(store), config()
    paused = await run_compiled(app, {}, cfg)
    [interrupt] = (await app.aget_state(cfg)).interrupts
    forged = bound(interrupt, APPROVE)  # even a well-formed, signed value
    again = await run_compiled(app, Command(resume={interrupt.id: forged}), cfg)
    assert again.status == "paused" and again.pending.approval_id == paused.pending.approval_id
    record = await store.get(paused.pending.approval_id)
    assert (record.status, record.consumed) == ("pending", False)
    assert RAN == []
    assert (await run_compiled(app, APPROVE, cfg)).status == "done" and RAN == ["hi"]


async def test_a_value_signed_with_another_key_is_refused():
    app, cfg = _app(None), config()
    await run_compiled(app, {}, cfg)
    [interrupt] = (await app.aget_state(cfg)).interrupts
    other = bound(interrupt, APPROVE, key=b"another key of 16+ bytes")
    unsigned = {**bound(interrupt, APPROVE), "mac": ""}
    for value in (other, unsigned):
        with pytest.raises(ValidationError):
            await app.ainvoke(Command(resume={interrupt.id: value}), cfg)
    assert RAN == []


def _ask() -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "send", "args": {"text": "t"}, "id": "c1"}])


async def test_tool_node_concurrent_and_replayed_resumes_run_the_tool_once():
    executor = Executor(registry, approval=LangGraphApproval())
    app, cfg = one_node(tool_node(registry, executor=executor)), config()
    out = await app.ainvoke({"messages": [_ask()]}, cfg, version="v2")
    [interrupt] = out.interrupts
    snapshot = await app.aget_state(cfg)
    command = resume(interrupt, APPROVE)
    results = await asyncio.gather(
        app.ainvoke(command, cfg), app.ainvoke(command, cfg), return_exceptions=True
    )
    _, failed = _split(results)
    assert [type(e) for e in failed] == [ApprovalAlreadyDecided]
    with pytest.raises(ApprovalAlreadyDecided):
        await app.ainvoke(command, snapshot.config)
    assert RAN == ["t"]


class _Flaky(BaseModel):
    text: str = ""


_FAILS: list[int] = []


def flaky(state: _Flaky) -> dict[str, Any]:
    if not _FAILS:
        _FAILS.append(1)
        raise RuntimeError("boom")
    return {"text": "ran"}


async def test_a_gate_grant_used_by_a_failed_run_is_never_reused_or_asked_again():
    from agent_adapter.core.errors import FlowError

    _FAILS.clear()
    store = MemoryApprovalStore()
    flow = Flow("flaky", state=_Flaky).node("go", flaky, pause=True).end("go")
    executor = Executor(approval=DeferredApproval(store))
    app = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver())
    cfg = config()
    await run_compiled(app, {}, cfg)
    with pytest.raises(FlowError, match="RuntimeError"):
        await run_compiled(app, APPROVE, cfg)  # the grant was spent by the failing visit
    with pytest.raises(ApprovalAlreadyDecided):
        await run_compiled(app, None, cfg)
    assert len(await store.list()) == 1
