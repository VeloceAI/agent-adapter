"""Starter LangGraph graph: a tool node runs registry tools. Run: python graph.py"""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_adapter import Executor, Registry
from agent_adapter.integrations.langgraph import tool_node

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return {"eu": 5.0, "us": 18.0}.get(region, 12.0)


def agent(state: MessagesState) -> dict[str, Any]:
    """Stands in for your model node: it asks for one tool call."""
    call = {"name": "shipping_rate", "args": {"region": "us"}, "id": "c1"}
    return {"messages": [AIMessage("", tool_calls=[call])]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node(registry, executor=Executor(registry)))
graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", END)
app = graph.compile()  # add a checkpointer before you give a tool confirm=True

if __name__ == "__main__":
    print(asyncio.run(app.ainvoke({"messages": []}))["messages"][-1].content)
