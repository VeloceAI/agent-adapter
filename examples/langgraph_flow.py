"""LangGraph: a tool node with interrupt approvals, and a flow compiled to a StateGraph."""

import asyncio
from typing import Annotated, Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent_adapter import (
    ApprovalDecision,
    DeferredApproval,
    Executor,
    Flow,
    MemoryApprovalStore,
    Registry,
)
from agent_adapter.flow import Map, Merge
from agent_adapter.integrations.langgraph import (
    LangGraphApproval,
    compile_flow,
    run_compiled,
    tool_node,
)

registry = Registry()


@registry.tool(parallel_safe=True)
async def shipping_rate(item: str, region: str = "us") -> float:
    """Shipping rate for an item, in percent of the order value (by region)."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


@registry.tool(confirm=True)
async def send_statement(period: str) -> str:
    """Send the account statement for a period (needs approval)."""
    return f"sent {period}"


async def tools_demo() -> None:
    # --8<-- [start:tool_node]
    port = LangGraphApproval(key=b"a shared secret of 16+ bytes")  # signs resume values
    executor = Executor(registry, approval=port)  # approvals pause the graph
    graph = StateGraph(MessagesState)
    graph.add_node(tool_node(registry, executor=executor))  # a node named "tools"
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    app = graph.compile(checkpointer=InMemorySaver())

    calls = [
        {"name": "shipping_rate", "args": {"item": "laptop"}, "id": "c1"},
        {"name": "send_statement", "args": {"period": "2026-09"}, "id": "c2"},
    ]
    config: Any = {"configurable": {"thread_id": "shop-1"}}
    out = await app.ainvoke({"messages": [AIMessage("", tool_calls=calls)]}, config, version="v2")
    [interrupt] = out.interrupts  # shipping_rate ran; send_statement waits for a reviewer
    request = interrupt.value["request"]
    print("asks", request["tool"], request["arguments"])

    answer = port.resume_value(interrupt, ApprovalDecision(approved=True))  # bound and signed
    out = await app.ainvoke(Command(resume={interrupt.id: answer}), config, version="v2")
    print([m.content for m in out.value["messages"][1:]])
    # --8<-- [end:tool_node]


# --8<-- [start:flow]
class Order(BaseModel):
    items: list[dict[str, Any]]
    rates: Annotated[list[float], Merge.append] = Field(default_factory=list)
    receipt: str = ""


def place(state: Order) -> dict[str, Any]:
    return {"receipt": f"placed with {len(state.rates)} rates"}


flow = (
    Flow("order", state=Order)
    .node("rate", Map("items", tool="shipping_rate", into="rates"))
    .node("place", place, pause=lambda s: {"rates": s.rates})
    .edge("rate", "place")
    .end("place")
)
# --8<-- [end:flow]


async def flow_demo() -> None:
    # --8<-- [start:compile]
    store = MemoryApprovalStore()
    executor = Executor(registry, approval=DeferredApproval(store))
    app = compile_flow(flow, executor=executor).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "order-1"}}

    paused = await run_compiled(app, {"items": [{"item": "a"}, {"item": "b"}]}, config)
    assert paused.pending is not None
    print(paused.status, paused.pending.tool, paused.pending.arguments)

    done = await run_compiled(app, ApprovalDecision(approved=True), config)
    print(done.status, done.steps, dict(done.visits), done.state.receipt)
    # --8<-- [end:compile]


async def main() -> None:
    await tools_demo()
    await flow_demo()


if __name__ == "__main__":
    asyncio.run(main())
