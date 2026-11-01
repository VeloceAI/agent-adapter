"""`tool_node`, `tool_task` and `fan_out`: registry tools as LangGraph nodes and `Send` targets."""

import asyncio
import json
import operator
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolError, ToolFailed, ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.flow import Merge
from agent_adapter.integrations.langgraph import fan_out, reducer, tool_node, tool_task
from tests.fakes.sinks import RecordingLedger
from tests.integrations.langgraph.support import config, one_node

pytestmark = pytest.mark.contract

registry = Registry()
SEEN: list[ToolContext] = []


@registry.tool
async def rate(item: str, ctx: ToolContext) -> dict[str, Any]:
    """GST rate of an item."""
    SEEN.append(ctx)
    return {"item": item, "rate": 18}


@registry.tool
async def echo(text: str) -> str:
    """Echo a text."""
    return text


@registry.tool
async def refuse(item: str) -> str:
    """Always refuses."""
    raise ToolError(f"no rate for {item!r}", tool="refuse")


@registry.tool
async def crash(item: str) -> str:
    """A programming error."""
    raise KeyError(item)


@registry.tool(targets="adk")
async def elsewhere(item: str) -> str:
    """Exported to ADK only."""
    return item


def ai(*calls: tuple[str, dict[str, Any], str]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in calls])


async def run(node: Any, message: AIMessage, cfg: dict[str, Any] | None = None) -> list[Any]:
    out = await one_node(node).ainvoke({"messages": [message]}, cfg or config())
    return out["messages"][1:]


async def test_tool_node_answers_each_call_in_order_with_a_tool_message():
    SEEN.clear()
    node = tool_node(registry, executor=Executor(registry))
    cfg = config("thread-1")
    messages = await run(
        node, ai(("rate", {"item": "pen"}, "c1"), ("echo", {"text": "hi"}, "c2")), cfg
    )
    assert [type(m) for m in messages] == [ToolMessage, ToolMessage]
    first, second = messages
    assert (first.tool_call_id, first.name, first.status) == ("c1", "rate", "success")
    assert json.loads(first.content) == {"item": "pen", "rate": 18}
    assert (second.tool_call_id, second.content) == ("c2", "hi")
    [ctx] = SEEN
    assert (ctx.call_id, ctx.run_id, ctx.thread_id) == ("c1", "thread-1", "thread-1")
    assert ctx.extras["langgraph"]["id"] == "c1"


async def test_run_id_comes_from_configurable_run_id_before_the_thread():
    SEEN.clear()
    node = tool_node(registry, executor=Executor(registry))
    cfg = config("thread-2")
    cfg["configurable"]["run_id"] = "run-x"
    await run(node, ai(("rate", {"item": "pen"}, "c1")), cfg)
    assert (SEEN[0].run_id, SEEN[0].thread_id) == ("run-x", "thread-2")


async def test_model_visible_errors_become_error_messages_and_others_raise():
    node = tool_node(registry, executor=Executor(registry))
    refused, invalid, unknown, elsewhere_msg = await run(
        node,
        ai(
            ("refuse", {"item": "kite"}, "c1"),
            ("echo", {"wrong": 1}, "c2"),
            ("nope", {}, "c3"),
            ("elsewhere", {"item": "x"}, "c4"),
        ),
    )
    assert refused.status == "error"
    assert json.loads(refused.content) == {
        "error": {"kind": "tool-error", "message": "no rate for 'kite'"}
    }
    assert invalid.status == "error" and "tool-args-invalid" in invalid.content
    assert unknown.status == "error" and "tool-not-found" in unknown.content
    assert elsewhere_msg.status == "error"  # not exported to langgraph
    with pytest.raises(ToolFailed) as caught:
        await run(node, ai(("crash", {"item": "x"}, "c1")))
    assert isinstance(caught.value.__cause__, KeyError)


async def test_duplicate_call_ids_are_reported_not_run_twice():
    node = tool_node(registry, executor=Executor(registry))
    first, second = await run(
        node, ai(("echo", {"text": "a"}, "c1"), ("echo", {"text": "b"}, "c1"))
    )
    assert first.content == "a"
    assert second.status == "error" and "duplicate" in second.content


async def test_tool_node_needs_an_ai_message_with_tool_calls_and_accepts_a_message_list():
    node = tool_node(registry, executor=Executor(registry))
    with pytest.raises(ToolUsageError, match="AIMessage"):
        await node({"messages": [HumanMessage("hi")]}, config())
    with pytest.raises(ToolUsageError, match="AIMessage"):
        await node({"messages": []}, config())
    out = await node([ai(("echo", {"text": "x"}, "c1"))], config())
    assert [m.content for m in out] == ["x"]
    custom = tool_node(registry, executor=Executor(registry), messages_key="history")
    out = await custom({"history": [ai(("echo", {"text": "y"}, "c1"))]}, config())
    assert [m.content for m in out["history"]] == ["y"]


async def test_tool_node_exposes_the_tool_definitions_it_runs():
    node = tool_node(registry)
    names = [tool["function"]["name"] for tool in node.tools]
    assert names == ["rate", "echo", "refuse", "crash"]  # targets honoured
    assert node.tools[0]["function"]["parameters"] == registry.get("rate").json_schema()
    assert node.name == "tools"


async def test_concurrency_caps_parallel_calls():
    active, peak = 0, 0
    local = Registry()

    @local.tool(parallel_safe=True)
    async def slow(n: int) -> int:
        """Sleep briefly."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return n

    node = tool_node(local, executor=Executor(local), concurrency=2)
    calls = [("slow", {"n": i}, f"c{i}") for i in range(5)]
    messages = await run(node, ai(*calls))
    assert [m.content for m in messages] == ["0", "1", "2", "3", "4"]
    assert peak == 2
    with pytest.raises(ToolUsageError, match="concurrency"):
        tool_node(local, concurrency=0)


async def test_cancellation_propagates_out_of_the_node():
    local = Registry()

    @local.tool
    async def stop(x: int) -> int:
        """Raise CancelledError."""
        raise asyncio.CancelledError

    node = tool_node(local, executor=Executor(local))
    with pytest.raises(asyncio.CancelledError):
        await node({"messages": [ai(("stop", {"x": 1}, "c1"))]}, config())


class Fan(TypedDict, total=False):
    items: list[dict[str, Any]]
    results: Annotated[list[Any], reducer(Merge.append)]
    log: Annotated[list[str], operator.add]


async def test_fan_out_sends_each_item_to_a_tool_task():
    ledger = RecordingLedger()
    executor = Executor(registry, ledger=ledger)
    graph = StateGraph(Fan)
    graph.add_node(tool_task(registry.get("rate"), executor=executor, into="results"))
    graph.add_conditional_edges(START, fan_out("rate", "items"), ["rate"])
    graph.add_edge("rate", END)
    app = graph.compile()
    items = [{"item": "pen"}, {"item": "ink"}]
    out = await app.ainvoke({"items": items, "results": []}, config("fan"))
    assert sorted(r["item"] for r in out["results"]) == ["ink", "pen"]
    ids = {entry.call_id for entry in ledger.entries("fan")}
    assert len(ids) == 2 and all(i.startswith("rate:") for i in ids)  # one stable id per task


async def test_fan_out_takes_a_function_and_refuses_items_that_are_not_mappings():
    route = fan_out("echo", lambda state: [{"text": t} for t in state["texts"]])
    sends = route({"texts": ["a", "b"]})
    assert [(s.node, s.arg) for s in sends] == [("echo", {"text": "a"}), ("echo", {"text": "b"})]
    with pytest.raises(ToolUsageError, match="mapping"):
        fan_out("echo", "items")({"items": ["a"]})


async def test_tool_task_reports_model_visible_errors_in_its_field():
    graph = StateGraph(Fan)
    graph.add_node("refuse", tool_task(registry.get("refuse"), into="results"))
    graph.add_conditional_edges(START, fan_out("refuse", "items"), ["refuse"])
    graph.add_edge("refuse", END)
    out = await graph.compile().ainvoke({"items": [{"item": "kite"}], "results": []}, config())
    assert out["results"] == [{"error": {"kind": "tool-error", "message": "no rate for 'kite'"}}]
    with pytest.raises(ToolUsageError, match="mapping"):
        await tool_task(registry.get("echo"), into="results")("not a mapping", config())
    with pytest.raises(ToolUsageError, match="langgraph"):
        tool_task(registry.get("elsewhere"), into="results")


async def test_tool_node_with_messages_state_appends_its_messages():
    graph = StateGraph(MessagesState)
    graph.add_node(tool_node(registry, executor=Executor(registry)))  # named "tools"
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    out = await graph.compile().ainvoke({"messages": [ai(("echo", {"text": "z"}, "c9"))]})
    assert out["messages"][-1].content == "z"
