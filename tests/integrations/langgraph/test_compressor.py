"""langgraph_compressor and the langchain format on real langchain_core messages."""

import asyncio
import copy
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent_adapter.context import CompressionPolicy, CutLongMessages, MyOwnStep, RemoveOldMessages
from agent_adapter.context.formats import compress_for_sync, convert, detect_format
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import Message
from agent_adapter.integrations.langgraph import LangGraphCompressor, langgraph_compressor

pytestmark = pytest.mark.contract
PAD = "x" * 400
DROP = CompressionPolicy(
    start_when_tokens_over=50,
    shrink_to_tokens=10,
    steps=[RemoveOldMessages(keep_first=0, keep_last=1)],
)
CUT = CompressionPolicy(
    start_when_tokens_over=50,
    shrink_to_tokens=10,
    steps=[CutLongMessages(max_tokens_per_message=20)],
)


def _history() -> list[Any]:
    call = {"name": "lookup", "args": {"q": "a"}, "id": "c1", "type": "tool_call"}
    return [
        SystemMessage(content="Be brief."),
        HumanMessage(content=f"old {PAD}", id="h1", additional_kwargs={"keep": True}),
        AIMessage(content=[{"type": "text", "text": f"look {PAD}"}], tool_calls=[call], id="a1"),
        ToolMessage(content=f"rows {PAD}", tool_call_id="c1", name="lookup", id="t1"),
        AIMessage(content=f"done {PAD}", id="a2"),
        HumanMessage(content="new", id="h2"),
    ]


def test_langchain_round_trip_is_lossless():
    items = _history()
    snapshot = copy.deepcopy(items)
    conversion = convert(items)
    assert detect_format(items) == "langchain"
    back = conversion.back(conversion.messages)
    assert back == snapshot and all(a is b for a, b in zip(back, items, strict=True))
    calls = [c.id for m in conversion.messages for c in m.tool_calls]
    assert calls == [m.tool_call_id for m in conversion.messages if m.role == "tool"] == ["c1"]


def test_dropping_keeps_rounds_whole_and_returns_the_same_objects():
    items = _history()
    smaller, report = compress_for_sync(items, DROP)
    assert report.tokens_after < report.tokens_before
    assert smaller == [items[0], items[-1]]
    assert all(any(s is i for i in items) for s in smaller)


def test_cutting_keeps_ids_tool_calls_and_kwargs():
    items = _history()
    smaller, _ = compress_for_sync(items, CUT)
    assert [type(m) for m in smaller] == [type(m) for m in items]
    human, ai, tool = smaller[1], smaller[2], smaller[3]
    assert human.id == "h1" and human.additional_kwargs == {"keep": True}
    assert "characters cut" in human.content
    assert ai.tool_calls == items[2].tool_calls and ai.content[0]["type"] == "text"
    assert tool.tool_call_id == "c1" and tool.name == "lookup" and tool.id == "t1"
    assert smaller[-1] is items[-1]


def test_pinned_messages_stay_verbatim():
    items = _history()
    compressor = langgraph_compressor(DROP, pinned=lambda m: bool(m.additional_kwargs.get("keep")))
    smaller = compressor.sync(items)
    assert smaller == [items[0], items[1], items[-1]] and smaller[1] is items[1]


def test_a_summary_step_makes_real_langchain_messages():
    def summarise(messages: list[Message]) -> list[Message]:
        return [messages[0], Message(role="user", content="earlier: a lookup"), messages[-1]]

    policy = CompressionPolicy(
        start_when_tokens_over=50, shrink_to_tokens=10, steps=[MyOwnStep(summarise)]
    )
    smaller, _ = compress_for_sync(_history(), policy)
    assert isinstance(smaller[1], HumanMessage) and smaller[1].content == "earlier: a lookup"


def test_async_compressor_inside_a_langgraph_node():
    compressor = langgraph_compressor(DROP)
    seen: list[list[Any]] = []

    class State(TypedDict):
        messages: list[Any]

    async def agent(state: State) -> dict[str, Any]:
        sent = await compressor(state["messages"])
        seen.append(sent)
        return {"messages": [*state["messages"], AIMessage(content=f"saw {len(sent)}")]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    result = asyncio.run(graph.compile().ainvoke({"messages": _history()}))
    assert len(seen[0]) == 2
    assert len(result["messages"]) == 7  # the state keeps the full history
    assert result["messages"][-1].content == "saw 2"


def test_compressor_checks_its_policy_and_items():
    with pytest.raises(ConfigError, match="CompressionPolicy"):
        LangGraphCompressor("policy")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="not a langchain message"):
        langgraph_compressor(DROP).sync([{"role": "user", "content": "x"}])
