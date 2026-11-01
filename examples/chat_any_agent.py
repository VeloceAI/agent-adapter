"""Chat with any agent: a plain async function that runs a small flow and streams its answer.

Run it and open the printed link; Ctrl+C stops it. To chat with LangGraph, ADK, CrewAI or an
HTTP backend instead, pass a factory from `examples/chat_wrappers.py` to `serve(agent=...)`.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agent_adapter import Flow, FlowRunner, Registry
from agent_adapter.flow import Map, Merge
from agent_adapter.viewer import ChatMessage, ChatSession, serve

registry = Registry()


@registry.tool(parallel_safe=True)
async def opening_hours(place: str) -> str:
    """When a place is open."""
    await asyncio.sleep(0.05)
    return f"{place}: 09:00-18:00"


@registry.tool(parallel_safe=True)
async def ticket_price(place: str) -> float:
    """An adult ticket, in EUR."""
    await asyncio.sleep(0.05)
    return 12.5


class Visit(BaseModel):
    question: str
    lookups: list[dict[str, Any]] = Field(default_factory=list)
    facts: Annotated[list[Any], Merge.append] = Field(default_factory=list)  # one per lookup
    answer: str = ""


def plan(state: Visit) -> dict[str, Any]:
    place = state.question.rstrip("?").split()[-1]
    tools = ("opening_hours", "ticket_price")
    return {"lookups": [{"tool": t, "args": {"place": place}} for t in tools]}


def answer(state: Visit) -> dict[str, Any]:
    hours = next(f for f in state.facts if isinstance(f, str))
    price = next(f for f in state.facts if isinstance(f, float))
    return {"answer": f"**{hours}**, and a ticket costs **{price:.2f} EUR**."}


visit_flow = (
    Flow("visit", state=Visit)
    .node("plan", plan)
    .node("lookup", Map("lookups", tool_from="tool", into="facts"), concurrency=2)
    .node("answer", answer)
    .edge("plan", "lookup")
    .edge("lookup", "answer")
    .end("answer")
)


# --8<-- [start:plain]
async def chat(message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
    """Runs the flow in this turn's run, then streams the answer word by word."""
    session.state["questions"] = session.state.get("questions", 0) + 1  # editable in the page
    runner = FlowRunner(session.executor, trace_state=True)  # trace_state fills the State tab
    result = await runner.run(visit_flow, {"question": message.text}, ctx=session.context())
    for word in result.state.answer.split(" "):
        yield word + " "
        await asyncio.sleep(0.02)


# --8<-- [end:plain]


if __name__ == "__main__":
    # --8<-- [start:serve]
    serve(agent=chat, registry=registry, port=8780, open_browser=True)  # blocks until Ctrl+C
    # --8<-- [end:serve]
