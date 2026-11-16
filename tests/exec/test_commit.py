import asyncio

import pytest

from agent_adapter import Registry, ToolCall
from agent_adapter.errors import ToolUsageError
from agent_adapter.exec import listen_progress, progress
from agent_adapter.exec._commit import REPEAT_NOTE
from agent_adapter.hooks import Hook
from tests.exec.helpers import check_events, make

registry = Registry()
gate = {"open": asyncio.Event, "count": 0}
booked: list[str] = []


def reset() -> asyncio.Event:
    booked.clear()
    event = asyncio.Event()
    gate["open"] = event  # type: ignore[assignment]
    return event


@registry.tool(effects="destructive")
async def book(seat: str) -> str:
    """Book a seat."""
    progress("Booking that now")
    await gate["open"].wait()  # type: ignore[attr-defined]
    booked.append(seat)
    return f"booked {seat}"


@registry.tool(effects="read_only")
async def look(seat: str) -> str:
    """Look a seat up."""
    await gate["open"].wait()  # type: ignore[attr-defined]
    booked.append(f"looked {seat}")
    return "free"


@registry.tool(effects="destructive", idempotent=True)
async def set_name(name: str) -> str:
    """Set a name (safe to repeat)."""
    booked.append(name)
    return name


async def _cancel_while_running(ex, call, ctx, release):
    task = asyncio.create_task(ex.run_call(call, ctx))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() or call.name == "look"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cancelled_call_that_changes_things_finishes_once_and_is_recorded():
    release = reset()
    ex, sink, ledger, _ = make(registry)
    ctx = ex.context(run_id="r1")
    await _cancel_while_running(
        ex, ToolCall(name="book", arguments={"seat": "4A"}, id="c1"), ctx, release
    )
    assert booked == ["4A"]
    assert check_events(sink) == [
        "tool.started",
        "tool.progress",
        "tool.finished",
        "tool.interrupted",
    ]
    [entry] = ledger.entries("r1")
    assert entry.ok and entry.call_id == "c1"


async def test_a_cancelled_read_only_call_stops_at_once():
    release = reset()
    ex, sink, _, _ = make(registry)
    task = asyncio.create_task(
        ex.run_call(ToolCall(name="look", arguments={"seat": "4A"}), ex.context())
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0)
    assert booked == []
    assert "tool.interrupted" not in sink.types()


async def test_a_repeat_with_a_new_call_id_returns_the_first_result():
    release = reset()
    release.set()
    ex, sink, _, _ = make(registry, dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    first = await ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}, id="c1"), ctx)
    again = await ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}, id="c2"), ctx)
    other = await ex.run_call(ToolCall(name="book", arguments={"seat": "5B"}, id="c3"), ctx)
    assert booked == ["4A", "5B"]
    assert first.value == "booked 4A" and again.cached and again.call_id == "c2"
    assert again.value == f"booked 4A ({REPEAT_NOTE})"
    assert other.value == "booked 5B"
    assert sink.types().count("tool.cached") == 1


async def test_repeats_running_at_the_same_time_run_once():
    release = reset()
    ex, _, _, _ = make(registry, dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    calls = [
        ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}, id=f"c{i}"), ctx)
        for i in range(3)
    ]
    tasks = [asyncio.create_task(c) for c in calls]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks)
    assert booked == ["4A"]
    assert [r.value for r in results] == ["booked 4A"] + [f"booked 4A ({REPEAT_NOTE})"] * 2


async def test_the_window_expires_and_is_per_run():
    release = reset()
    release.set()
    ex, _, _, clock = make(registry, dedupe_window_s=10)
    call = ToolCall(name="book", arguments={"seat": "4A"})
    await ex.run_call(call, ex.context(run_id="r1"))
    await ex.run_call(call, ex.context(run_id="r2"))
    clock.advance(11)
    await ex.run_call(call, ex.context(run_id="r1"))
    assert booked == ["4A", "4A", "4A"]


async def test_idempotent_and_read_only_tools_are_not_deduped():
    release = reset()
    release.set()
    ex, _, _, _ = make(registry, dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    for _ in range(2):
        await ex.run_call(ToolCall(name="set_name", arguments={"name": "Ann"}), ctx)
        await ex.run_call(ToolCall(name="look", arguments={"seat": "1"}), ctx)
    assert booked == ["Ann", "looked 1", "Ann", "looked 1"]


async def test_a_failed_call_is_not_remembered():
    reset().set()
    calls = {"n": 0}
    reg = Registry()

    @reg.tool(effects="additive")
    async def flaky() -> str:
        """Fail once."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("down")
        return "ok"

    ex, _, _, _ = make(reg, dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    assert not (await ex.run_call(ToolCall(name="flaky"), ctx)).ok
    assert (await ex.run_call(ToolCall(name="flaky"), ctx)).value == "ok"


def test_dedupe_window_must_be_positive():
    with pytest.raises(ToolUsageError, match="dedupe_window_s"):
        make(registry, dedupe_window_s=0)


async def test_progress_goes_to_the_listener_and_never_into_the_event():
    reset().set()
    heard: list[str] = []
    ex, sink, _, _ = make(registry)
    with listen_progress(heard.append):
        await ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}), ex.context())
    assert heard == ["Booking that now"]
    [event] = [e for e in sink.events if e.type == "tool.progress"]
    assert event.data == {}


def test_progress_outside_a_call_and_without_a_listener_is_a_no_op():
    progress("nothing listens")


async def test_a_cancel_after_the_tool_but_before_the_record_still_records_once():
    reset().set()
    in_hook, leave = asyncio.Event(), asyncio.Event()

    class SlowAfter(Hook):
        async def after_tool(self, inv, result):
            in_hook.set()
            await leave.wait()
            return None

    ex, _, ledger, _ = make(registry, hooks=[SlowAfter()], dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    task = asyncio.create_task(
        ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}, id="c1"), ctx)
    )
    await in_hook.wait()
    task.cancel()
    await asyncio.sleep(0)
    leave.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    again = await ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}, id="c2"), ctx)
    assert booked == ["4A"] and again.cached
    assert [e.call_id for e in ledger.entries("r1")] == ["c1", "c2"]


async def test_a_cancel_before_the_tool_starts_stops_it():
    reset().set()
    waiting = asyncio.Event()

    class SlowBefore(Hook):
        async def before_tool(self, inv):
            waiting.set()
            await asyncio.Event().wait()

    ex, _, _, _ = make(registry, hooks=[SlowBefore()])
    task = asyncio.create_task(
        ex.run_call(ToolCall(name="book", arguments={"seat": "4A"}), ex.context())
    )
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert booked == []
