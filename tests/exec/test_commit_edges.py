import asyncio

import pytest

from agent_adapter import CallbackApproval, Registry, ToolCall, ToolResult
from agent_adapter.exec import changes_things
from agent_adapter.exec._commit import RepeatWindow, finish_once_started
from agent_adapter.exec._loop_steps import _tool_parts
from tests.exec.helpers import check_events, make

refunds: list[float] = []
gate: dict[str, asyncio.Event] = {}
registry = Registry()


@registry.tool(confirm=True)
async def refund(amount: float) -> str:
    """Refund an amount."""
    await gate["open"].wait()
    refunds.append(amount)
    return f"refunded {amount}"


@registry.tool(confirm=True, effects="read_only")
async def peek(amount: float) -> str:
    """Approved, but declared read-only."""
    await gate["open"].wait()
    refunds.append(-amount)
    return "seen"


@registry.tool(effects="additive")
async def charge(amount: float) -> str:
    """Charge an amount."""
    refunds.append(amount)
    return f"charged {amount}"


@registry.tool(once_per_run=True)
def welcome() -> str:
    """Send the welcome note once."""
    return "sent"


def reset() -> asyncio.Event:
    refunds.clear()
    gate["open"] = asyncio.Event()
    return gate["open"]


def test_confirm_or_once_per_run_without_effects_changes_things():
    assert changes_things(registry.get("refund"))
    assert changes_things(registry.get("welcome"))
    assert not changes_things(registry.get("peek"))  # an explicit effects value wins


async def test_an_approved_call_without_effects_finishes_when_cancelled():
    release = reset()
    ex, sink, ledger, _ = make(registry, approval=CallbackApproval(lambda r: True))
    task = asyncio.create_task(
        ex.run_call(
            ToolCall(name="refund", arguments={"amount": 5}, id="c1"), ex.context(run_id="r1")
        )
    )
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert refunds == [5.0]
    assert check_events(sink)[-1] == "tool.interrupted"
    [entry] = ledger.entries("r1")
    assert entry.ok


async def test_an_explicit_read_only_effect_is_still_cancelled():
    release = reset()
    ex, sink, _, _ = make(registry, approval=CallbackApproval(lambda r: True))
    task = asyncio.create_task(
        ex.run_call(ToolCall(name="peek", arguments={"amount": 5}), ex.context(run_id="r1"))
    )
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0)
    assert refunds == []
    assert "tool.interrupted" not in sink.types()


@pytest.mark.parametrize("again", [10.0, "10"])
async def test_repeats_match_on_validated_arguments(again):
    reset()
    ex, _, _, _ = make(registry, dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    await ex.run_call(ToolCall(name="charge", arguments={"amount": 10}), ctx)
    repeat = await ex.run_call(ToolCall(name="charge", arguments={"amount": again}), ctx)
    assert refunds == [10.0]
    assert repeat.cached


async def test_a_dropped_repeat_tells_the_model_it_was_not_run_again():
    reset()
    ex, _, _, _ = make(registry, dedupe_window_s=30)
    ctx = ex.context(run_id="r1")
    first = await ex.run_call(ToolCall(name="charge", arguments={"amount": 1}), ctx)
    repeat = await ex.run_call(ToolCall(name="charge", arguments={"amount": 1}), ctx)
    assert first.value == "charged 1.0"
    text, _ = _tool_parts(repeat)
    assert text.startswith("charged 1.0") and "not run again" in text


def _ok(value: object) -> ToolResult:
    return ToolResult(call_id="c", name="t", status="ok", value=value)


async def test_after_a_failed_first_only_one_waiter_runs():
    window = RepeatWindow(30, lambda: 0.0)
    key = ("r", "t", "d")
    runs: list[int] = []
    fail = asyncio.Event()

    async def first():
        await fail.wait()
        raise RuntimeError("down")

    async def work():
        runs.append(1)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return _ok("done"), None

    head = asyncio.create_task(window.run(key, first))
    await asyncio.sleep(0)
    waiters = [asyncio.create_task(window.run(key, work)) for _ in range(3)]
    await asyncio.sleep(0)
    fail.set()
    with pytest.raises(RuntimeError):
        await head
    results = await asyncio.gather(*waiters)
    assert runs == [1]
    assert [r[0].value for r in results] == ["done"] * 3
    assert not window._running


async def test_work_that_finishes_as_the_caller_is_cancelled_is_reported_done():
    ready = asyncio.Event()

    async def work() -> str:
        await ready.wait()
        return "done"

    outer = asyncio.create_task(finish_once_started(work(), lambda: True))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    ready.set()
    outer.cancel()
    assert await outer == ("done", True)


async def test_work_cancelled_itself_still_raises():
    async def work() -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await finish_once_started(work(), lambda: True)
