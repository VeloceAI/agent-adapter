"""An answer the executor used is spent on failure, whatever context or sink the call ran in."""

import dataclasses
from collections import Counter
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalAlreadyDecided, FlowError
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.exec import Executor
from agent_adapter.integrations.langgraph import LangGraphApproval, compile_flow, run_compiled
from agent_adapter.integrations.langgraph._spent import CLAIMS
from tests.integrations.langgraph.support import config

pytestmark = pytest.mark.contract

WIRED: Counter[int] = Counter()
registry = Registry()
APPROVE = ApprovalDecision(approved=True)


@registry.tool(confirm=True)
async def wire(amount: int) -> str:
    """Wire money (needs approval)."""
    WIRED[amount] += 1
    return f"wired {amount}"


@pytest.fixture(autouse=True)
def _clean():
    WIRED.clear()
    CLAIMS.clear()


class Note(BaseModel):
    text: str = ""


def _own_context(executor: Executor, ctx: ToolContext) -> ToolContext:
    return executor.context(run_id=ctx.run_id)


def _no_sink(executor: Executor, ctx: ToolContext) -> ToolContext:
    return dataclasses.replace(ctx, sink=None)


def _app(backed: bool, context: Any) -> Any:
    from agent_adapter.flow import Flow

    port = DeferredApproval(MemoryApprovalStore()) if backed else LangGraphApproval()
    executor = Executor(registry, approval=port)

    async def code(state: Note, ctx: ToolContext) -> dict[str, Any]:
        await executor.run("wire", {"amount": 5}, context(executor, ctx))
        raise RuntimeError("the node fails after the wire")

    flow = Flow("wires", state=Note).node("go", code).end("go")
    return compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize("backed", [False, True], ids=["langgraph-port", "store"])
@pytest.mark.parametrize("context", [_own_context, _no_sink], ids=["own-context", "no-sink"])
async def test_an_approved_call_outside_the_node_context_is_not_run_again_on_retry(backed, context):
    app, cfg = _app(backed, context), config()
    await run_compiled(app, {}, cfg)
    with pytest.raises(FlowError, match="RuntimeError"):
        await run_compiled(app, APPROVE, cfg)
    assert WIRED == {5: 1}
    with pytest.raises(ApprovalAlreadyDecided):
        await run_compiled(app, None, cfg)  # the host retries the failed task
    assert WIRED == {5: 1}
