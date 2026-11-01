import asyncio
import logging

import pytest

from agent_adapter.core.errors import ConfigError, RunLimitExceeded
from agent_adapter.exec.guards import (
    RunLimits,
    _call_identity,
    _run_slot,
    _RunLock,
    _RunTracker,
)


def test_run_limits_defaults_and_frozen_caps():
    limits = RunLimits(per_kind={"search": 2})
    assert limits.max_calls is None and limits.max_concurrency == 8
    with pytest.raises(TypeError):
        limits.per_kind["search"] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "fields",
    [
        {"max_calls": 0},
        {"max_calls": True},
        {"max_concurrency": 0},
        {"max_concurrency": 1.5},
        {"per_kind": {"search": 0}},
        {"per_kind": {"": 1}},
    ],
)
def test_run_limits_reject_bad_values(fields):
    with pytest.raises(ConfigError):
        RunLimits(**fields)


def test_tracker_rejects_bad_size():
    with pytest.raises(ConfigError):
        _RunTracker(0)


def _admit(tracker, call_id, *, tool="t", once=False, max_calls=None, run="r", args=None):
    """Admit and commit: a call that reaches execution."""
    identity = _call_identity(call_id, tool, args or {})
    guards = {"once_per_run": once, "max_calls": max_calls}
    state = tracker.admit(run, identity, **guards)
    tracker.commit(state, identity, **guards)
    return state


def test_admit_only_checks_and_commit_counts():
    tracker = _RunTracker(10)
    first = _call_identity("c1", "t", {})
    state = tracker.admit("r", first, once_per_run=True, max_calls=1)
    tracker.admit("r", _call_identity("c2", "t", {}), once_per_run=True, max_calls=1)
    tracker.commit(state, first, once_per_run=True, max_calls=1)
    with pytest.raises(RunLimitExceeded):  # a concurrent call lost the race at commit time
        tracker.commit(state, _call_identity("c2", "t", {}), once_per_run=True, max_calls=1)


def test_reusing_a_call_id_for_another_tool_or_arguments_is_a_new_call():
    tracker = _RunTracker(10)
    _admit(tracker, "c1", tool="a", once=True, max_calls=2)
    _admit(tracker, "c1", tool="a", once=True, max_calls=2)  # exact re-entry: free
    _admit(tracker, "c1", tool="b", once=True, max_calls=2)  # same id, other tool: counted
    with pytest.raises(RunLimitExceeded) as info:  # same id, other args: a new call
        _admit(tracker, "c1", tool="a", once=True, max_calls=3, args={"x": 1})
    assert info.value.limit == "once_per_run"
    with pytest.raises(RunLimitExceeded) as info:
        _admit(tracker, "c1", tool="c", max_calls=2)
    assert info.value.limit == "max_calls"
    _admit(tracker, "c1", tool="b", once=True, max_calls=2)  # b's own re-entry still free


def test_call_identity_survives_unencodable_arguments():
    first = _call_identity("c", "t", {"x": object()})
    assert first[:2] == ("c", "t") and len(first[2]) == 32


def test_pending_approval_ids_are_remembered_per_call():
    tracker = _RunTracker(10)
    state = _admit(tracker, "c1")
    assert tracker.pending_approval(state, "c1") == ""
    tracker.set_pending_approval(state, "c1", "apr_1")
    assert tracker.pending_approval(state, "c1") == "apr_1"
    tracker.set_pending_approval(state, "c1", "")
    assert tracker.pending_approval(state, "c1") == ""


def test_once_per_run_rejects_a_second_distinct_call():
    tracker = _RunTracker(10)
    _admit(tracker, "c1", once=True)
    with pytest.raises(RunLimitExceeded) as info:
        _admit(tracker, "c2", once=True)
    assert info.value.limit == "once_per_run" and info.value.run_id == "r"
    _admit(tracker, "c3", once=True, run="other")  # another run is unaffected


def test_max_calls_counts_distinct_call_ids():
    tracker = _RunTracker(10)
    _admit(tracker, "c1", max_calls=2)
    _admit(tracker, "c2", max_calls=2)
    _admit(tracker, "c1", max_calls=2)  # re-entry (approval resume, replay) is not a new call
    with pytest.raises(RunLimitExceeded) as info:
        _admit(tracker, "c3", max_calls=2)
    assert info.value.limit == "max_calls"


def test_reentry_of_a_once_per_run_call_is_allowed():
    tracker = _RunTracker(10)
    _admit(tracker, "c1", once=True)
    _admit(tracker, "c1", once=True)


def test_end_forgets_the_run():
    tracker = _RunTracker(10)
    _admit(tracker, "c1", once=True)
    tracker.end("r")
    tracker.end("missing")
    _admit(tracker, "c2", once=True)


def test_eviction_skips_active_runs(caplog):
    tracker = _RunTracker(2)
    busy = _admit(tracker, "c", run="busy")
    idle = _admit(tracker, "c", run="idle")
    tracker.leave(idle)
    with caplog.at_level(logging.INFO, logger="agent_adapter.exec.guards"):
        _admit(tracker, "c", run="new")
    assert len(tracker) == 2 and "idle evicted" in caplog.text
    _admit(tracker, "c", run="busy")  # still known: re-entry, no new count
    assert busy.active == 2


def test_eviction_warns_when_every_run_is_active(caplog):
    tracker = _RunTracker(1)
    _admit(tracker, "c", run="a")
    with caplog.at_level(logging.WARNING, logger="agent_adapter.exec.guards"):
        _admit(tracker, "c", run="b")
    assert len(tracker) == 2 and "every tracked run is active" in caplog.text


async def test_run_lock_shared_holders_overlap_and_exclusive_waits():
    lock = _RunLock()
    await lock.acquire(False)
    await lock.acquire(False)
    writer = asyncio.create_task(lock.acquire(True))
    await asyncio.sleep(0)
    assert not writer.done() and lock.readers == 2
    reader = asyncio.create_task(lock.acquire(False))  # queued behind the writer (FIFO)
    await asyncio.sleep(0)
    assert not reader.done()
    lock.release(False)
    lock.release(False)
    await writer
    assert lock.writer and not reader.done()
    lock.release(True)
    await reader
    assert lock.readers == 1 and not lock.writer


async def test_run_lock_cancelled_waiter_leaves_the_queue():
    lock = _RunLock()
    await lock.acquire(True)
    first = asyncio.create_task(lock.acquire(True))
    second = asyncio.create_task(lock.acquire(False))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    lock.release(True)
    await second
    assert lock.readers == 1 and not lock.writer


async def test_run_lock_cancel_after_grant_releases():
    lock = _RunLock()
    await lock.acquire(True)
    waiter = asyncio.create_task(lock.acquire(True))
    await asyncio.sleep(0)
    lock.release(True)  # grants the waiter...
    waiter.cancel()  # ...which is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not lock.writer and lock.readers == 0


async def test_run_lock_cancelled_head_waiter_wakes_the_next():
    lock = _RunLock()
    await lock.acquire(False)
    writer = asyncio.create_task(lock.acquire(True))
    reader = asyncio.create_task(lock.acquire(False))
    await asyncio.sleep(0)
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer
    await reader  # no writer ahead any more
    assert lock.readers == 2


async def _probe(state, call_id, parent, *, exclusive, log, kind="k", cap=None, hold=0.01):
    async with _run_slot(state, call_id, parent, exclusive=exclusive, kind=kind, cap=cap):
        log.append(("in", call_id))
        await asyncio.sleep(hold)
        log.append(("out", call_id))


async def test_run_slot_exclusive_calls_never_overlap():
    tracker = _RunTracker(10)
    state = _admit(tracker, "x")
    log: list[tuple[str, str]] = []
    await asyncio.gather(*(_probe(state, f"c{i}", "", exclusive=True, log=log) for i in range(3)))
    assert log == [(e, f"c{i}") for i in range(3) for e in ("in", "out")]


async def test_run_slot_kind_cap_limits_concurrency():
    tracker = _RunTracker(10)
    state = _admit(tracker, "x")
    running, peak = 0, 0

    async def one(i):
        nonlocal running, peak
        async with _run_slot(state, f"c{i}", "", exclusive=False, kind="search", cap=2):
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(one(i) for i in range(5)))
    assert peak == 2


async def test_run_slot_nested_calls_run_inside_the_parent_slot():
    tracker = _RunTracker(10)
    state = _admit(tracker, "x")
    seen: list[set[str]] = []

    async def nested(call_id, parent, depth):
        async with _run_slot(state, call_id, parent, exclusive=True, kind="k", cap=1):
            seen.append(set(state.loops.get().active))
            if depth:
                await nested(f"{call_id}.child", call_id, depth - 1)

    await asyncio.wait_for(nested("root", "", 2), timeout=1)
    assert seen == [
        {"root"},
        {"root", "root.child"},
        {"root", "root.child", "root.child.child"},
    ]
    assert not state.loops.get().active


async def test_nested_calls_still_wait_for_kinds_their_ancestors_do_not_hold():
    tracker = _RunTracker(10)
    state = _admit(tracker, "x")
    running = peak = 0

    async def http(call_id, parent):
        nonlocal running, peak
        async with _run_slot(state, call_id, parent, exclusive=False, kind="http", cap=1):
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.005)
            if call_id == "h0":  # http nested in http: the ancestor holds the slot
                await asyncio.wait_for(http("h0.inner", "h0"), timeout=1)
            running -= 1

    async with _run_slot(state, "orch", "", exclusive=True, kind="orch", cap=None):
        await asyncio.gather(*(http(f"h{i}", "orch") for i in range(5)))
    assert peak == 2  # h0 and its own nested child; never two siblings


async def test_active_counts_duplicate_call_ids():
    tracker = _RunTracker(10)
    state = _admit(tracker, "x")
    gate = asyncio.Event()

    async def hold():
        async with _run_slot(state, "dup", "", exclusive=False, kind="k", cap=None):
            await gate.wait()

    first = asyncio.create_task(hold())
    second = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    assert state.loops.get().active["dup"] == 2
    gate.set()
    await asyncio.gather(first, second)
    assert "dup" not in state.loops.get().active
