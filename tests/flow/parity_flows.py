"""The flows, tools and executor behind the parity cases (see `tests/flow/parity.py`).

Every target runs these flows with `executor`: its registry holds the tools, and its
approval port (`DeferredApproval` over `store`) makes confirm tools pend.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field

from agent_adapter.approval import DeferredApproval, MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.flow import Call, Flow, Join, Map, Merge, Route, keyed

registry = Registry()


@registry.tool(parallel_safe=True)
async def search(query: str) -> list[dict[str, str]]:
    """Return evidence for a query: its own hit and one hit every query shares."""
    return [
        {"source_id": f"src-{query}", "text": f"evidence for {query}"},
        {"source_id": "shared", "text": "shared evidence"},
    ]


@registry.tool(parallel_safe=True)
async def bump(n: int) -> int:
    """Add one to n."""
    return n + 1


@registry.tool(parallel_safe=True)
async def broken(query: str) -> list[dict[str, str]]:
    """A search backend that is down."""
    raise ConnectionError(query)


@registry.tool(parallel_safe=True, confirm=True)
async def send(text: str) -> str:
    """Send a message (needs approval)."""
    return f"sent {text}"


_STAMPS: dict[tuple[str, str], int] = {}


@registry.tool(parallel_safe=True, confirm=True)
async def stamp(text: str, ctx: ToolContext | None = None) -> str:
    """Stamp a text (needs approval); the number counts real runs of it in this run."""
    key = (ctx.run_id if ctx is not None else "", text)
    _STAMPS[key] = _STAMPS.get(key, 0) + 1
    return f"{text}#{_STAMPS[key]}"


store = MemoryApprovalStore()
executor = Executor(registry, approval=DeferredApproval(store))


# --- case 1: fan-out Map + Join + keyed merge ----------------------------------------------


class ResearchState(BaseModel):
    queries: list[str]
    plan: list[dict[str, Any]] = Field(default_factory=list)
    evidence: Annotated[list[dict[str, Any]], keyed(by="source_id")] = Field(default_factory=list)
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)
    calls: Annotated[int, Merge.sum] = 0
    summary: str = ""


def plan(state: ResearchState) -> dict[str, Any]:
    steps = [{"tool": "search", "args": {"query": q}} for q in state.queries]
    return {"plan": steps, "log": ["plan"]}


def count(state: ResearchState) -> dict[str, Any]:
    return {"calls": len(state.plan), "log": ["count"]}


def audit(state: ResearchState) -> dict[str, Any]:
    return {"calls": 1, "log": ["audit"]}


def summarise(state: ResearchState) -> dict[str, Any]:
    ids = sorted(item["source_id"] for item in state.evidence)
    return {"summary": ",".join(ids), "log": ["merge"]}


fanout_flow = (
    Flow("fanout", state=ResearchState)
    .node("plan", plan)
    .node("retrieve", Map("plan", tool_from="tool", into="evidence"), concurrency=2)
    .node("count", count)
    .node("audit", audit)
    .node("merge", Join(["retrieve", "count", "audit"], fn=summarise))
    .edge("plan", ["retrieve", "count", "audit"])
    .edge("retrieve", "merge")
    .edge("count", "merge")
    .edge("audit", "merge")
    .end("merge")
)

# --- case 2: route loop with max_loops and on_exhausted -------------------------------------


class LoopState(BaseModel):
    target: int
    n: int = 0
    trail: Annotated[list[str], Merge.append] = Field(default_factory=list)
    finished: bool = False


def note(state: LoopState) -> dict[str, Any]:
    return {"trail": [f"n={state.n}"]}


def finish(state: LoopState) -> dict[str, Any]:
    return {"finished": True}


def again_or_stop(state: LoopState) -> str:
    return "again" if state.n < state.target else "stop"


loop_flow = (
    Flow("loop", state=LoopState)
    .node("work", Call("bump", args=lambda s: {"n": s.n}, into="n"))
    .node("note", note)
    .node("finish", finish)
    .edge("work", "note")
    .route(
        "note",
        Route(again_or_stop, targets={"again": "work", "stop": "finish"}),
        max_loops=3,
        on_exhausted="finish",
    )
    .end("finish")
)

# --- case 3: a pause node, approved, denied and edited --------------------------------------


class ReviewState(BaseModel):
    topic: str
    draft: str = ""
    published: str = ""
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)


def write(state: ReviewState) -> dict[str, Any]:
    return {"draft": f"draft about {state.topic}", "log": ["write"]}


def publish(state: ReviewState) -> dict[str, Any]:
    return {"published": state.draft, "log": ["publish"]}


review_flow = (
    Flow("review", state=ReviewState)
    .node("write", write)
    .node("publish", publish, pause=lambda s: {"draft": s.draft})
    .edge("write", "publish")
    .end("publish")
)


# --- cases ----------------------------------------------------------------------------------


# --- small flows for the cross-target edge cases --------------------------------------------


class Board(BaseModel):
    flag: str = ""
    note: str = ""
    sent: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    hits: Annotated[list[dict[str, Any]], keyed(by="source_id")] = Field(default_factory=list)
    log: Annotated[list[str], Merge.append] = Field(default_factory=list)


def mark(name: str, **update: Any) -> Any:
    """A node that logs its name and writes `update`."""

    def node(state: Board) -> dict[str, Any]:
        return {"log": [name], **update}

    node.__name__ = name
    return node


# a pause gate beside a sibling: nothing in the superstep runs until the gate is answered
gate_flow = (
    Flow("gate", state=Board)
    .node("start", mark("start"))
    .node("gated", mark("gated"), pause=True)
    .node("side", mark("side"))
    .edge("start", ["gated", "side"])
    .end("gated", "side")
)

# a tool-level pause in a Call beside a sibling: the sibling's writes are kept, it runs once
tool_pause_flow = (
    Flow("tool_pause", state=Board)
    .node("start", mark("start"))
    .node("ask", Call("send", args={"text": "hello"}, into="sent"))
    .node("side", mark("side"))
    .edge("start", ["ask", "side"])
    .end("ask", "side")
)


async def send_from_code(state: Board, ctx: ToolContext) -> dict[str, Any]:
    """A function node calling a confirm tool itself (fresh call ids on every visit)."""
    return {"sent": await executor.run("send", {"text": "from code"}, ctx), "log": ["code"]}


code_tool_flow = Flow("code_tool", state=Board).node("go", send_from_code).end("go")

# a route sees its own node's writes, not a sibling's from the same superstep
route_view_flow = (
    Flow("route_view", state=Board)
    .node("start", mark("start"))
    .node("setter", mark("setter", flag="set"))
    .node("chooser", mark("chooser"))
    .node("saw_flag", mark("saw_flag"))
    .node("no_flag", mark("no_flag"))
    .edge("start", ["setter", "chooser"])
    .route(
        "chooser", Route(lambda s: "yes" if s.flag else "no", {"yes": "saw_flag", "no": "no_flag"})
    )
    .end("setter", "saw_flag", "no_flag")
)

# two nodes write the same key in one superstep: node-name order, so "zeta" wins over "alpha"
collision_flow = (
    Flow("collision", state=Board)
    .node("start", mark("start"))
    .node("zeta", mark("zeta", hits=[{"source_id": "x", "text": "zeta"}]))
    .node("alpha", mark("alpha", hits=[{"source_id": "x", "text": "alpha"}]))
    .edge("start", ["zeta", "alpha"])
    .end("zeta", "alpha")
)

# equal replace writes from two nodes in one superstep are not a conflict
equal_flow = (
    Flow("equal", state=Board)
    .node("start", mark("start"))
    .node("left", mark("left", note="same"))
    .node("right", mark("right", note="same"))
    .edge("start", ["left", "right"])
    .end("left", "right")
)

# a diamond: the shared successor runs once
diamond_flow = (
    Flow("diamond", state=Board)
    .node("top", mark("top"))
    .node("left", mark("left"))
    .node("right", mark("right"))
    .node("bottom", mark("bottom"))
    .edge("top", ["left", "right"])
    .edge("left", "bottom")
    .edge("right", "bottom")
    .end("bottom")
)

# a join whose second source is never taken still runs once the taken one is done
untaken_join_flow = (
    Flow("untaken_join", state=Board)
    .node("pick", mark("pick"))
    .node("taken", mark("taken"))
    .node("skipped", mark("skipped"))
    .node("joined", Join(["taken", "skipped"], fn=mark("joined")))
    .route("pick", Route(lambda s: "taken", targets=["taken", "skipped"]))
    .edge("taken", "joined")
    .edge("skipped", "joined")
    .end("joined")
)

# a Map item that fails is dropped with on_error="skip"
map_skip_flow = (
    Flow("map_skip", state=Board)
    .node("fan", Map("items", tool_from="tool", into="hits", on_error="skip"))
    .end("fan")
)

# one node needing two approvals: two confirm items in one Map, resumed twice
two_approvals_flow = (
    Flow("two_approvals", state=Board).node("fan", Map("items", tool="send", into="log")).end("fan")
)


async def stamp_twice(state: Board, ctx: ToolContext) -> dict[str, Any]:
    """Two confirm calls in one function node: each pauses once and runs once."""
    first = await executor.run("stamp", {"text": "first"}, ctx)
    second = await executor.run("stamp", {"text": "second"}, ctx)
    return {"note": f"{first} {second}"}


# the local runner replays a resumed node's finished calls; LangGraph re-runs them
replay_flow = Flow("replay", state=Board).node("go", stamp_twice).end("go")
