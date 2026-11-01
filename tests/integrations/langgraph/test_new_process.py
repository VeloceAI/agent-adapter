"""A resume in another process replays the paused node's finished calls from its sealed state."""

from collections import Counter
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import FlowError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.cost import Ledger
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow
from agent_adapter.integrations.langgraph import LangGraphApproval, compile_flow, run_compiled
from agent_adapter.integrations.langgraph._spent import CLAIMS
from tests.integrations.langgraph.support import config

pytestmark = pytest.mark.contract

KEY = b"a key every process shares"
RAN: Counter[str] = Counter()
registry = Registry()
APPROVE = ApprovalDecision(approved=True)


@registry.tool(once_per_run=True)
async def fetch(item: str) -> str:
    """Fetch an item (once per run)."""
    RAN[f"fetch {item}"] += 1
    return f"got {item}"


@registry.tool(confirm=True)
async def stamp(text: str) -> str:
    """Stamp a text (needs approval)."""
    RAN[f"stamp {text}"] += 1
    return f"stamped {text}"


class Note(BaseModel):
    text: str = ""


def _setup(backed: bool, key: bytes = KEY) -> tuple[Flow[Note], Executor, Ledger, Any]:
    store = MemoryApprovalStore() if backed else None
    port = DeferredApproval(store) if store is not None else LangGraphApproval(key)
    ledger = Ledger()
    executor = Executor(registry, approval=port, ledger=ledger)

    async def code(state: Note, ctx: ToolContext) -> dict[str, Any]:
        got = await executor.run("fetch", {"item": "x"}, ctx)  # own calls: fresh call ids
        first = await executor.run("stamp", {"text": "first"}, ctx)
        second = await executor.run("stamp", {"text": "second"}, ctx)
        return {"text": f"{got} / {first} / {second}"}

    flow = Flow("code", state=Note).node("go", code).end("go")
    return flow, executor, ledger, store


def _process(flow: Flow[Note], executor: Executor, saver: Any, key: bytes = KEY) -> Any:
    """A graph as another process compiles it: claims start empty, the checkpointer is shared."""
    CLAIMS.clear()
    return compile_flow(flow, executor=executor, checkpoint_key=key).compile(checkpointer=saver)


@pytest.fixture(autouse=True)
def _clean():
    RAN.clear()
    CLAIMS.clear()


@pytest.mark.parametrize("backed", [False, True], ids=["langgraph-port", "store"])
async def test_a_new_process_resume_replays_finished_calls_once(backed):
    flow, executor, ledger, store = _setup(backed)
    saver, cfg = InMemorySaver(), config("np-thread")
    first = await run_compiled(_process(flow, executor, saver), {}, cfg)
    assert first.pending.arguments == {"text": "first"}
    second = await run_compiled(_process(flow, executor, saver), APPROVE, cfg)
    assert second.pending.arguments == {"text": "second"}
    done = await run_compiled(_process(flow, executor, saver), APPROVE, cfg)
    assert done.state.text == "got x / stamped first / stamped second"
    assert RAN == {"fetch x": 1, "stamp first": 1, "stamp second": 1}
    assert len(ledger.entries("np-thread")) == 3  # one entry per real call, once_per_run held
    if store is not None:
        assert not await store.list(status="pending")  # no record left behind
        assert len(await store.list(status="approved")) == 2


@pytest.mark.parametrize("backed", [False, True], ids=["langgraph-port", "store"])
async def test_a_process_without_the_key_fails_closed_and_runs_nothing(backed):
    flow, executor, _, _ = _setup(backed)
    saver, cfg = InMemorySaver(), config()
    await run_compiled(_process(flow, executor, saver), {}, cfg)
    _, other_executor, _, other_store = _setup(backed, key=b"another process key")
    if other_store is not None:
        other_executor = executor  # the same store, another seal key
    app = _process(flow, other_executor, saver, key=b"another process key")
    with pytest.raises(FlowError, match="resume in the pausing process"):
        await run_compiled(app, APPROVE, cfg)
    assert RAN == {"fetch x": 1}


async def test_a_gated_node_that_then_asks_for_a_tool_resumes_in_a_new_process():
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    flow = (
        Flow("gate_then_tool", state=Note)
        .node("go", Call("stamp", args={"text": "x"}, into="text"), pause=True)
        .end("go")
    )
    saver, cfg = InMemorySaver(), config()
    gate = await run_compiled(_process(flow, executor, saver), {}, cfg)
    assert gate.pending.kind == "node"
    tool = await run_compiled(_process(flow, executor, saver), APPROVE, cfg)
    assert (tool.pending.kind, tool.pending.tool) == ("tool", "stamp")
    done = await run_compiled(_process(flow, executor, saver), APPROVE, cfg)
    assert done.status == "done" and done.state.text == "stamped x"
    assert RAN == {"stamp x": 1} and not await store.list(status="pending")
