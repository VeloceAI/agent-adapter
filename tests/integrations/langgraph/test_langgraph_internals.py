"""Version guards: the LangGraph 1.2 details this package relies on still hold."""

import operator
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.runnables import RunnableConfig
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_adapter.integrations.langgraph._context import NS_KEY
from agent_adapter.integrations.langgraph._durable import CHECKPOINTER_KEY, INTERRUPT_CHANNEL
from tests.integrations.langgraph.support import config

pytestmark = pytest.mark.contract


class S(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


SEEN: list[dict[str, Any]] = []


async def ask(state: S, config: RunnableConfig) -> dict[str, Any]:
    SEEN.append(config)
    interrupt({"q": 1})
    return {"log": ["ask"]}


def side(state: S) -> dict[str, Any]:
    return {"log": ["side"]}


def test_the_empty_channel_sentinel_is_langgraphs():
    from langgraph._internal._typing import MISSING

    assert LastValue(object).checkpoint() is MISSING


async def test_task_namespaces_saver_and_saved_interrupts_have_the_shapes_we_read():
    SEEN.clear()
    graph = StateGraph(S)
    graph.add_node("ask", ask)
    graph.add_node("side", side)
    graph.add_edge(START, "ask")
    graph.add_edge(START, "side")
    graph.add_edge("ask", END)
    graph.add_edge("side", END)
    saver = InMemorySaver()
    app = graph.compile(checkpointer=saver)
    cfg = config()
    await app.ainvoke({"log": []}, cfg)
    snapshot = await app.aget_state(cfg)
    [task] = [t for t in snapshot.tasks if t.name == "ask"]
    [seen] = SEEN
    namespace = seen["metadata"][NS_KEY]
    assert namespace == f"ask:{task.id}" == seen["configurable"]["checkpoint_ns"]
    assert seen["configurable"][CHECKPOINTER_KEY] is saver
    assert "" in seen["configurable"]["checkpoint_map"]
    found = await saver.aget_tuple(snapshot.config)
    writes = [(w[0], w[1]) for w in found.pending_writes]
    assert (task.id, INTERRUPT_CHANNEL) in writes
    assert found.checkpoint["channel_values"].get("log") == []  # the sibling's write is pending
    assert snapshot.values["log"] == ["side"]  # while the snapshot shows it applied


PADS: list[list[Any]] = []


async def twice(state: S, config: RunnableConfig) -> dict[str, Any]:
    from agent_adapter.integrations.langgraph.approval import SCRATCHPAD_KEY, resume_count

    pad = config["configurable"][SCRATCHPAD_KEY]
    PADS.append(list(pad.resume))
    assert resume_count(config) == len(pad.resume)
    interrupt({"q": 1})
    interrupt({"q": 2})
    return {"log": ["twice"]}


async def test_the_resume_scratchpad_has_the_shape_we_read():
    from langgraph.types import Command

    PADS.clear()
    graph = StateGraph(S)
    graph.add_node("twice", twice)
    graph.add_edge(START, "twice")
    graph.add_edge("twice", END)
    app, cfg = graph.compile(checkpointer=InMemorySaver()), config()
    [first] = (await app.ainvoke({"log": []}, cfg, version="v2")).interrupts
    [second] = (await app.ainvoke(Command(resume={first.id: "a"}), cfg, version="v2")).interrupts
    await app.ainvoke(Command(resume={second.id: "b"}), cfg, version="v2")
    assert PADS == [[], ["a"], ["a", "b"]]  # earlier answers by position, this one appended
