"""`analyse_run`: joins and waits, repeated calls, savings, claims and cost per agent."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_adapter import Registry, ToolCall, ToolContext
from agent_adapter.board import board_of
from agent_adapter.core.types import Event
from agent_adapter.exec import Executor, agent_tool
from agent_adapter.viewer import analyse_run, build_spans


def ev(type: str, ts: float, call: str = "", parent: str = "", **data: Any) -> dict[str, Any]:
    return {"type": type, "ts": ts, "call_id": call, "parent_id": parent, "data": data}


def node(name: str, start: float, end: float, call: str) -> list[dict[str, Any]]:
    ids = {"flow": "trip", "node": name, "visit": 1}
    return [ev("node.started", start, call, **ids), ev("node.finished", end, call, **ids)]


def llm(call: str, parent: str, start: float, end: float, cost: float) -> list[dict[str, Any]]:
    ids = {"provider": "p", "model": "m", "stage": "s"}
    return [
        ev("llm.started", start, call, parent, stream=False, **ids),
        ev(
            "llm.finished",
            end,
            call,
            parent,
            latency_ms=1,
            usage={},
            cost_usd=cost,
            finish_reason="stop",
            **ids,
        ),
    ]


def agent(name: str, call: str, start: float, end: float) -> list[dict[str, Any]]:
    return [
        ev("tool.started", start, call, tool=name, kind="agent"),
        ev("tool.finished", end, call, tool=name, latency_ms=1, attempts=1, cached=False),
    ]


def test_supersteps_report_each_branch_wait_and_the_slowest():
    events = [*node("look", 0, 1, "n1"), *node("price", 0, 3, "n2"), *node("plan", 3, 4, "n3")]
    report = analyse_run(events)
    [join] = report["joins"]
    assert (join["kind"], join["where"], join["slowest"]) == ("flow", "trip", "price")
    assert join["wall_ms"] == 3000
    waits = {b["name"]: b["waited_ms"] for b in join["branches"]}
    assert waits == {"look": 2000, "price": 0}
    assert report["waited_ms"] == 2000 and report["critical_path"] == ["price"]
    assert report["parallel_ms"] == 3000


def test_cost_and_calls_roll_up_to_the_agent_around_them():
    events = [
        *agent("scout", "a1", 0, 2),
        *agent("writer", "a2", 0, 1),
        *llm("l1", "a1", 0.1, 0.5, 0.01),
        *llm("l2", "a1", 0.6, 0.9, 0.02),
        *llm("l3", "a2", 0.1, 0.4, 0.05),
        *llm("l4", "", 2.1, 2.2, 0.5),
        ev("board.posted", 0.3, "x", "", entry_id="e", kind="finding", key="k", by="scout", seq=1),
        ev("board.claimed", 0.4, "x", "", key="job", by="scout", holder="scout", won=True),
        ev("board.claimed", 0.5, "y", "", key="job", by="writer", holder="scout", won=False),
    ]
    report = analyse_run(events)
    rows = {r["agent"]: r for r in report["agents"]}
    assert rows["scout"]["cost_usd"] == 0.03 and rows["scout"]["llm_calls"] == 2
    assert rows["scout"]["busy_ms"] == 2000 and rows["scout"]["posts"] == 1
    assert rows["writer"]["cost_usd"] == 0.05 and rows["writer"]["claims_lost"] == 1
    assert rows["main"]["cost_usd"] == 0.5 and rows["main"]["tool_calls"] == 2
    assert report["claims"] == {
        "total": 2,
        "contested": [{"key": "job", "holder": "scout", "lost": ["writer"]}],
    }
    [join] = report["joins"]
    assert join["kind"] == "agents" and join["slowest"] == "scout"
    assert [p["type"] for p in report["board"]] == [
        "board.posted",
        "board.claimed",
        "board.claimed",
    ]


def test_savings_split_the_cache_from_calls_shared_in_flight():
    events = [
        ev("tool.cached", 1, "c1", tool="t"),
        ev("tool.cached", 2, "c2", tool="t"),
        ev("llm.cached", 3, "c3", provider="p", model="m", stage="s"),
        ev("inflight.shared", 2, "c2", kind="tool", name="t", waited_ms=120.5),
    ]
    assert analyse_run(events)["saved"] == {
        "cache": 2,
        "inflight": 1,
        "inflight_waited_ms": 120.5,
    }


def test_repeated_calls_need_recorded_arguments():
    finished = {"tool": "lookup", "latency_ms": 1, "attempts": 1, "cached": False}
    bare = [ev("tool.finished", 1, "c1", **finished), ev("tool.finished", 2, "c2", **finished)]
    report = analyse_run(bare)
    assert report["duplicates"] == [] and report["arguments_recorded"] is False
    args = [
        ev("tool.finished", 1, "c1", arguments={"city": "Oslo"}, **finished),
        ev("tool.finished", 2, "c2", arguments={"city": "Oslo"}, **finished),
        ev("tool.finished", 3, "c3", arguments={"city": "Rome"}, **finished),
    ]
    report = analyse_run(args)
    assert report["duplicate_calls"] == 1
    assert report["duplicates"][0]["call_ids"] == ["c1", "c2"]


class ContentSink:
    """Records events with their captured content, as the run recorder can."""

    wants_content = True

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


async def test_three_agents_fanned_out_live():
    registry = Registry()

    @registry.tool
    async def lookup(city: str) -> str:
        """Look a city up."""
        return f"{city}: ok"

    sink = ContentSink()
    executor = Executor(registry, sink=sink)

    def researcher(pause: float):
        async def run(task: str, ctx: ToolContext) -> str:
            board = board_of(ctx)
            claim = await board.claim("lookup:oslo")
            if claim.won:
                await executor.run("lookup", {"city": "Oslo"}, ctx)
            await executor.run("lookup", {"city": "Oslo"}, ctx)  # a repeat the board missed
            await asyncio.sleep(pause)
            await board.post("finding", f"{task} done")
            return task

        return run

    for name, pause in (("fast", 0.01), ("mid", 0.05), ("slow", 0.15)):
        registry.add(agent_tool(researcher(pause), name=name, description=name))
    ctx = executor.context()
    calls = [
        ToolCall(name=n, arguments={"task": n}, id=f"call_{n}") for n in ("fast", "mid", "slow")
    ]
    await executor.run_many(calls, ctx)

    report = analyse_run(sink.events)
    [join] = report["joins"]
    assert join["kind"] == "agents" and join["slowest"] == "slow"
    waits = {b["name"]: b["waited_ms"] for b in join["branches"]}
    assert waits["slow"] == 0 and waits["fast"] > waits["mid"] > 0
    assert report["duplicates"][0]["tool"] == "lookup"
    assert report["duplicates"][0]["count"] == 4  # 1 claimed + 3 repeats
    assert len(report["claims"]["contested"][0]["lost"]) == 2
    rows = {r["agent"]: r for r in report["agents"]}
    assert {rows[n]["posts"] for n in ("fast", "mid", "slow")} == {1}
    points = [s for s in build_spans([e.to_dict() for e in sink.events]) if s["family"] == "board"]
    assert len(points) == 6 and all(p["call_id"] == "" for p in points)
    assert any(
        p["name"].endswith("(held by " + report["claims"]["contested"][0]["holder"] + ")")
        for p in points
    )
