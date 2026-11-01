"""LangGraph: compress `state["messages"]` in your agent node, before the model."""

import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_adapter.context import CompressionPolicy, RemoveOldMessages
from agent_adapter.integrations.langgraph import langgraph_compressor

OLD = "an old turn with a lot of detail " * 30
policy = CompressionPolicy(
    start_when_tokens_over=1_000, shrink_to_tokens=600, steps=[RemoveOldMessages(keep_last=2)]
)
llm = FakeListChatModel(responses=["The rate is 30%."])  # your chat model
sent: list[int] = []  # how many messages each model call got


class State(TypedDict):
    messages: Annotated[list[Any], add_messages]


# --8<-- [start:node]
compressor = langgraph_compressor(policy)


async def agent(state: State) -> dict[str, Any]:
    messages = await compressor(state["messages"])  # or compressor.sync(...) in a sync node
    sent.append(len(messages))
    return {"messages": [await llm.ainvoke(messages)]}  # the state keeps the full history


# --8<-- [end:node]


def history() -> list[Any]:
    turns: list[Any] = [SystemMessage(content="Be brief.")]
    for n in range(3):
        turns += [HumanMessage(content=f"q{n} {OLD}"), AIMessage(content=f"a{n} {OLD}")]
    return [*turns, HumanMessage(content="What is the rate?")]


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    result = asyncio.run(graph.compile().ainvoke({"messages": history()}))
    print("langgraph:", len(result["messages"]), "in state,", sent[0], "sent")


if __name__ == "__main__":
    main()
