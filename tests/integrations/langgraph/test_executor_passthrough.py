"""`GraphInterrupt` passes through the executor and every batch method unwrapped."""

from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.graph import MessagesState

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.exec import Executor, Step
from agent_adapter.integrations.langgraph import LangGraphApproval, one_use
from tests.fakes.hooks import RecordingHook
from tests.fakes.sinks import RecordingLedger
from tests.integrations.langgraph.support import config, one_node, resume

pytestmark = pytest.mark.contract

registry = Registry()
RAN: list[str] = []


@registry.tool(confirm=True, parallel_safe=True)
async def wipe(item: str) -> str:
    """Wipe an item (needs approval)."""
    RAN.append(item)
    return f"wiped {item}"


@registry.tool(parallel_safe=True)
async def peek(item: str) -> str:
    """Look at an item."""
    return item


def _calls() -> list[ToolCall]:
    return [
        ToolCall(name="peek", arguments={"item": "p"}, id="c0"),
        ToolCall(name="wipe", arguments={"item": "w"}, id="c1"),
    ]


async def _run(executor: Executor) -> Any:
    return await executor.run("wipe", {"item": "w"}, executor.context(run_id="r"), call_id="c1")


async def _run_spec(executor: Executor) -> Any:
    spec = registry.get("wipe")
    return await executor.run_spec(spec, {"item": "w"}, executor.context(run_id="r"), call_id="c1")


async def _run_call(executor: Executor) -> Any:
    return await executor.run_call(_calls()[1], executor.context(run_id="r"))


async def _run_many_collect(executor: Executor) -> Any:
    return await executor.run_many(_calls(), executor.context(run_id="r"))


async def _run_many_fail_fast(executor: Executor) -> Any:
    return await executor.run_many(_calls(), executor.context(run_id="r"), on_error="fail_fast")


async def _as_completed(executor: Executor) -> Any:
    return [r async for r in executor.as_completed(_calls(), executor.context(run_id="r"))]


async def _run_plan(executor: Executor) -> Any:
    steps = [Step("a", "peek", {"item": "p"}), Step("b", "wipe", {"item": "w"}, depends_on=["a"])]
    return await executor.run_plan(steps, executor.context(run_id="r"))


async def _parallel(executor: Executor) -> Any:
    return await executor.parallel(lambda: _run(executor), lambda: _run_spec(executor))


async def _parallel_collect(executor: Executor) -> Any:
    return await executor.parallel(lambda: _run(executor), on_error="collect")


PATHS = [
    _run,
    _run_spec,
    _run_call,
    _run_many_collect,
    _run_many_fail_fast,
    _as_completed,
    _run_plan,
    _parallel,
    _parallel_collect,
]


@pytest.mark.parametrize("path", PATHS, ids=[p.__name__.strip("_") for p in PATHS])
async def test_graph_interrupt_escapes_every_executor_path_unwrapped(path):
    RAN.clear()
    hook, ledger = RecordingHook(), RecordingLedger()
    executor = Executor(registry, approval=LangGraphApproval(), hooks=[hook], ledger=ledger)
    escaped: list[BaseException] = []

    async def node(state):
        try:
            await path(executor)
        except BaseException as exc:
            escaped.append(exc)
            raise
        return {"messages": []}

    app = one_node(one_use(node), MessagesState)
    cfg = config()
    out = await app.ainvoke({"messages": []}, cfg, version="v2")
    assert len(out.interrupts) >= 1
    assert escaped and all(type(exc) is GraphInterrupt for exc in escaped), escaped  # unwrapped
    assert "on_tool_error" not in hook.phases()
    assert not [e for e in ledger.entries() if e.stage == "wipe"]  # an interrupt is not a failure
    for interrupt in out.interrupts:
        assert interrupt.value["request"]["tool"] == "wipe"
    assert RAN == []
    first = out.interrupts[0]
    out = await app.ainvoke(resume(first, ApprovalDecision(approved=True)), cfg, version="v2")
    assert "w" in RAN


def test_langgraph_bubble_ups_are_control_flow():
    assert is_control_flow(GraphInterrupt()) and issubclass(GraphInterrupt, GraphBubbleUp)
