"""The board stays cheap: capped news-only notes, short summaries and short tool output."""

from __future__ import annotations

from typing import Any

from agent_adapter import Registry, ToolCall, ToolContext
from agent_adapter.board import Board, board_of, board_tools, posting_as
from agent_adapter.board.tools import board_read
from agent_adapter.exec import Executor, LoopAgent, agent_tool
from agent_adapter.viewer import analyse_run
from tests.fakes.models import ScriptedModel, calls, text

LONG = "x" * 500


def notes_of(model: ScriptedModel) -> list[str]:
    return [
        m.content
        for request in model.requests
        for m in request.messages
        if m.role == "user" and "<shared-board>" in m.content
    ]


async def three_agents(board_chars: int | None = None) -> tuple[list[ScriptedModel], Board]:
    registry = Registry()

    @registry.tool
    async def work(step: int, ctx: ToolContext) -> str:
        """Do a step of work and post it."""
        board = board_of(ctx)
        await board.post("status", f"step {step}")
        await board.post("finding", f"result {step}: {LONG}", key=f"r{step}")
        return "done"

    team = Registry()
    executor = Executor(team)
    models = []
    for name in ("alpha", "beta", "gamma"):
        steps = [
            calls(ToolCall(name="work", arguments={"step": n}, id=f"{name}{n}")) for n in (1, 2)
        ]
        model = ScriptedModel([*steps, text(f"{name} done")])
        models.append(model)
        options: dict[str, Any] = {} if board_chars is None else {"board_chars": board_chars}
        agent = LoopAgent(model, registry, board="auto", **options)
        team.add(agent_tool(agent, name=name, description=name, executor=executor))
    ctx = executor.context()
    board = board_of(ctx)
    for n in range(50):
        await board.post("finding", f"old fact {n}: {LONG}", key=f"k{n}", by=f"seed{n % 4}")
    asks = [
        ToolCall(name=n, arguments={"task": "go"}, id=f"c_{n}") for n in ("alpha", "beta", "gamma")
    ]
    await executor.run_many(asks, ctx)
    return models, board


async def test_notes_stay_under_their_cap_with_three_agents_and_fifty_entries():
    models, board = await three_agents()
    assert len(await board.read()) > 50
    for model, name in zip(models, ("alpha", "beta", "gamma"), strict=True):
        notes = notes_of(model)
        assert notes, name
        assert all(len(n) <= 600 for n in notes), [len(n) for n in notes]
        assert all(f" {name} [" not in n for n in notes)  # never its own entries
        assert "[cut: board_read(entry=" in notes[0]
        assert "older: board_read(since=" in notes[0]
    raw = sum(len(e.text) for e in await board.read())
    sent = sum(len(n) for m in models for n in notes_of(m))
    assert sent < raw / 5  # 9 model calls got under a fifth of the board's text


async def test_board_chars_sets_the_cap():
    models, _ = await three_agents(board_chars=300)
    assert all(len(n) <= 300 for m in models for n in notes_of(m))


async def test_summary_drops_own_entries_merges_statuses_and_cuts_long_text():
    board = Board(scope="s")
    await board.post("status", "reading", by="scout")
    await board.post("status", "writing", by="scout")
    await board.post("finding", "mine", by="writer")
    await board.post("finding", LONG * 2, by="scout", key="long")
    with posting_as("writer"):
        text_ = await board.summary()
    assert "mine" not in text_ and "reading" not in text_ and "writing" in text_
    assert "[cut: board_read(entry=4)]" in text_ and len(text_) < 600
    assert "mine" in await board.summary(include_own=True)


async def test_board_read_is_short_newest_first_and_filters():
    registry = Registry()
    registry.add_many(board_tools(include=("read",)))
    executor = Executor(registry)
    ctx = executor.context()
    board = board_of(ctx)
    for n in range(30):
        await board.post("finding" if n % 2 else "status", f"note {n}", by="a" if n < 20 else "b")
    shown = await board_read(ctx)
    lines = [line for line in shown.splitlines() if line.startswith("#")]
    assert len(lines) == 10 and lines[0].startswith("#30 ")
    by_a = await board_read(ctx, by="a", kind="finding", since=10, limit=3)
    assert [line.split()[0] for line in by_a.splitlines() if line.startswith("#")] == [
        "#20",
        "#18",
        "#16",
    ]
    await board.post("finding", LONG * 3, by="b")
    whole = await executor.run("board_read", {"entry": 31}, ctx)
    assert whole.count("x") == 1500
    assert await board_read(ctx, entry=99) == "No entry #99 on the board."


def ev(type: str, call: str, parent: str = "", **data: Any) -> dict[str, Any]:
    return {"type": type, "ts": 1.0, "call_id": call, "parent_id": parent, "data": data}


def test_analyse_run_counts_tokens_per_agent_and_saved():
    ids = {"provider": "p", "model": "m", "stage": "s"}
    used = {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 30}
    events = [
        ev("tool.started", "a1", tool="scout", kind="agent"),
        ev("llm.started", "l1", "a1", stream=False, **ids),
        ev("llm.finished", "l1", "a1", usage=used, cost_usd=0.1, latency_ms=1, **ids),
        ev("tool.finished", "a1", tool="scout", latency_ms=1, attempts=1, cached=False),
        ev("llm.cached", "l2", usage={"input_tokens": 50, "output_tokens": 5}, **ids),
        ev("llm.cached", "l3", usage={"input_tokens": 40, "output_tokens": 4}, **ids),
        ev("inflight.shared", "l3", kind="llm", name="m", waited_ms=3),
    ]
    report = analyse_run(events)
    assert report["tokens"] == {
        "input": 120,
        "output": 30,
        "saved_by_cache": 55,
        "saved_by_inflight": 44,
    }
    scout = next(r for r in report["agents"] if r["agent"] == "scout")
    assert (scout["input_tokens"], scout["output_tokens"]) == (120, 30)
