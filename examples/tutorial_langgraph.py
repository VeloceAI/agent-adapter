"""Tutorial: move a LangGraph app's tools and approvals onto Agent Adapter, keep the graph."""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    Flow,
    MemoryApprovalStore,
    Registry,
)
from agent_adapter.flow import Call
from agent_adapter.integrations.langchain import import_langchain
from agent_adapter.integrations.langgraph import (
    LangGraphApproval,
    compile_flow,
    one_use,
    run_compiled,
    tool_node,
)

CALLS = [
    {"name": "shipping_rate", "args": {"item": "laptop"}, "id": "c1"},
    {"name": "send_statement", "args": {"period": "2026-09"}, "id": "c2"},
]


def build(tools_node: Any, calls: list[dict[str, Any]]) -> Any:
    """START -> agent -> tools -> END, with a checkpointer (interrupts need one)."""

    def agent(state: MessagesState) -> dict[str, Any]:  # stands in for the model node
        return {"messages": [AIMessage("", tool_calls=calls)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
    return graph.compile(checkpointer=InMemorySaver())


# --8<-- [start:before]
@tool
def shipping_rate(item: str, region: str = "us") -> float:
    """Shipping rate for an item, in percent of the order value (by region)."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@tool
def send_statement_unsafe(period: str) -> str:
    """Send the account statement for a period, after asking a human."""
    if interrupt({"period": period}) != "yes":  # any resume value "yes" approves
        return "cancelled"
    return f"sent {period}"


# --8<-- [end:before]


async def before_demo() -> None:
    calls = [CALLS[0], {**CALLS[1], "name": "send_statement_unsafe"}]
    app = build(ToolNode([shipping_rate, send_statement_unsafe]), calls)
    config: Any = {"configurable": {"thread_id": "before-1"}}
    [pending] = (await app.ainvoke({"messages": []}, config, version="v2")).interrupts
    out = await app.ainvoke(Command(resume={pending.id: "yes"}), config, version="v2")
    print("before:", [m.content for m in out.value["messages"][1:]])


# --8<-- [start:registry]
registry = Registry()
import_langchain(registry, [shipping_rate], parallel_safe=True)  # keep existing LangChain tools


@registry.tool(confirm=True, effects="destructive")
async def send_statement(period: str) -> str:
    """Send the account statement for a period (needs approval)."""
    return f"sent {period}"


# --8<-- [end:registry]

# --8<-- [start:port]
KEY = b"load me from your secret store"  # >= 16 bytes; the same in every process
port = LangGraphApproval(key=KEY)
executor = Executor(registry, approval=port)  # hooks, budget, ledger as usual
# --8<-- [end:port]


async def tool_node_demo() -> None:
    # --8<-- [start:tool_node]
    app = build(tool_node(registry, executor=executor), CALLS)  # replaces ToolNode([...])
    config: Any = {"configurable": {"thread_id": "shop-1"}}
    [pending] = (await app.ainvoke({"messages": []}, config, version="v2")).interrupts
    request = pending.value["request"]  # show reviewers this, never the raw interrupt value
    print("asks", request["tool"], request["arguments"])
    # --8<-- [end:tool_node]

    # --8<-- [start:refused]
    answer = port.resume_value(pending, ApprovalDecision(approved=True, decided_by="ana"))
    forged = {**answer, "decision": {"approved": True, "edited_args": {"period": "2020-01"}}}
    for value in ("yes", forged):  # a bare answer; an edited decision the mac does not cover
        try:
            await app.ainvoke(Command(resume={pending.id: value}), config)
        except ValidationError:  # LangGraph's invalid resume: nothing saved, nothing runs
            print("refused", "bare" if value == "yes" else "forged")
    out = await app.ainvoke(Command(resume={pending.id: answer}), config, version="v2")
    print("after:", [m.content for m in out.value["messages"][1:]])
    # --8<-- [end:refused]


# --8<-- [start:one_use]
async def send_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """Your own node calling a confirm tool: wrap it in one_use."""
    thread = str(config["configurable"]["thread_id"])
    ctx = executor.context(run_id=thread)
    receipt = await executor.run("send_statement", {"period": "2026-10"}, ctx, call_id="send-1")
    return {"messages": [AIMessage(str(receipt))]}


# --8<-- [end:one_use]


async def one_use_demo() -> None:
    graph = StateGraph(MessagesState)
    graph.add_node("send", one_use(send_node))
    graph.add_edge(START, "send")
    graph.add_edge("send", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "own-1"}}
    [pending] = (await app.ainvoke({"messages": []}, config, version="v2")).interrupts
    answer = port.resume_value(pending, ApprovalDecision(approved=True))
    out = await app.ainvoke(Command(resume={pending.id: answer}), config, version="v2")
    print("one_use:", out.value["messages"][-1].content)


# --8<-- [start:flow]
class Statement(BaseModel):
    rate: float = 0.0
    receipt: str = ""


flow = (
    Flow("statement", state=Statement)
    .node("rate", Call("shipping_rate", args={"item": "laptop"}, into="rate"))
    .node("send", Call("send_statement", args={"period": "2026-11"}, into="receipt"))
    .edge("rate", "send")
    .end("send")
)


async def flow_demo() -> None:
    store = MemoryApprovalStore()  # a database store: one use across processes
    flows = Executor(registry, approval=DeferredApproval(store))
    app = compile_flow(flow, executor=flows).compile(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "statement-1"}}

    paused = await run_compiled(app, {}, config)
    assert paused.pending is not None
    print(paused.status, paused.pending.tool, paused.pending.arguments)
    # the reviewer's process decides the record; any worker then continues with None
    await store.decide(paused.pending.approval_id, ApprovalDecision(approved=True))
    done = await run_compiled(app, None, config)
    print(done.status, done.state.rate, done.state.receipt)


# --8<-- [end:flow]


async def main() -> None:
    await before_demo()
    await tool_node_demo()
    await one_use_demo()
    await flow_demo()


if __name__ == "__main__":
    asyncio.run(main())
