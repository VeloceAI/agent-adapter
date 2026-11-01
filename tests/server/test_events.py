import asyncio
import threading
from collections.abc import AsyncIterator

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.ports import EventSink
from agent_adapter.core.types import Event
from agent_adapter.server.events import EventStream, MemoryEventStream


def ev(run_id: str, type: str = "tool.started", **data: object) -> Event:
    return Event(type, run_id=run_id, data=data)


async def take(it: AsyncIterator[tuple[int, Event]], n: int) -> list[tuple[int, str]]:
    out = []
    for _ in range(n):
        seq, event = await asyncio.wait_for(anext(it), 1)
        out.append((seq, event.type))
    return out


async def following(stream: MemoryEventStream, *run_ids: str) -> None:
    """Yield to the loop until each run has a subscriber (3.11's wait_for adds a task hop)."""
    for _ in range(100):
        if all(run_id in stream._waiters for run_id in run_ids):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"no subscriber for {run_ids}")


def test_memory_stream_is_an_event_stream_and_sink():
    stream = MemoryEventStream()
    assert isinstance(stream, EventStream)
    assert isinstance(stream, EventSink)


@pytest.mark.parametrize("kwargs", [{"max_events_per_run": 0}, {"max_runs": 0}])
def test_limits_must_be_positive(kwargs: dict[str, int]):
    with pytest.raises(ConfigError):
        MemoryEventStream(**kwargs)


async def test_seq_starts_at_one_per_run_and_replays_after():
    stream = MemoryEventStream()
    for t in ("run.started", "tool.started", "tool.finished"):
        stream.emit(ev("r1", t))
    stream.emit(ev("r2"))
    it = stream.subscribe("r1", after=1)
    assert await take(it, 2) == [(2, "tool.started"), (3, "tool.finished")]
    other = stream.subscribe("r2")
    assert await take(other, 1) == [(1, "tool.started")]
    await it.aclose()
    await other.aclose()


async def test_subscribers_follow_new_events():
    stream = MemoryEventStream()
    it = stream.subscribe("r1")
    pending = asyncio.ensure_future(take(it, 2))
    await following(stream, "r1")
    stream.emit(ev("r1", "tool.started"))
    stream.emit(ev("r1", "tool.finished"))
    assert await pending == [(1, "tool.started"), (2, "tool.finished")]
    await it.aclose()


async def test_events_from_another_thread_wake_the_subscriber():
    stream = MemoryEventStream()
    it = stream.subscribe("r1")
    pending = asyncio.ensure_future(take(it, 1))
    await following(stream, "r1")
    thread = threading.Thread(target=stream.emit, args=(ev("r1"),))
    thread.start()
    thread.join()
    assert await pending == [(1, "tool.started")]
    await it.aclose()


async def test_old_events_are_dropped_per_run():
    stream = MemoryEventStream(max_events_per_run=2)
    for i in range(4):
        stream.emit(ev("r1", "custom.n", i=i))
    it = stream.subscribe("r1")
    assert [seq for seq, _ in await take(it, 2)] == [3, 4]
    await it.aclose()


async def test_least_recent_runs_are_evicted_but_not_followed_ones():
    stream = MemoryEventStream(max_runs=2)
    followed = stream.subscribe("a")
    pending = asyncio.ensure_future(take(followed, 2))
    await following(stream, "a")
    stream.emit(ev("a"))
    stream.emit(ev("b"))
    stream.emit(ev("c"))  # evicts "b", the least recent run nobody follows
    assert stream._since("b", 0) == []
    stream.emit(ev("a", "tool.finished"))
    assert await pending == [(1, "tool.started"), (2, "tool.finished")]
    await followed.aclose()
    stream.emit(ev("d"))
    stream.emit(ev("e"))
    assert set(stream._runs) == {"d", "e"}


async def test_eviction_falls_back_to_the_oldest_followed_run():
    stream = MemoryEventStream(max_runs=1)
    a, b = stream.subscribe("a"), stream.subscribe("b")
    pa, pb = asyncio.ensure_future(take(a, 1)), asyncio.ensure_future(take(b, 1))
    await following(stream, "a", "b")
    stream.emit(ev("a"))
    assert await pa == [(1, "tool.started")]
    stream.emit(ev("b"))  # both followed: the oldest goes
    assert await pb == [(1, "tool.started")]
    assert set(stream._runs) == {"b"}
    stream.emit(ev("a", "tool.finished"))  # "a" keeps counting for its subscriber
    assert stream._since("a", 0)[0][0] == 2
    await a.aclose()
    await b.aclose()


async def test_events_without_a_run_are_dropped_and_subscribing_creates_nothing():
    stream = MemoryEventStream()
    stream.emit(ev(""))
    it = stream.subscribe("ghost")
    task = asyncio.ensure_future(anext(it))
    await asyncio.sleep(0.01)
    assert not task.done() and stream._runs == {}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream._waiters == {}


def test_emit_never_raises(caplog: pytest.LogCaptureFixture):
    stream = MemoryEventStream()
    stream._runs = None  # type: ignore[assignment]
    stream.emit(ev("r1"))
    assert "event stream failed" in caplog.text


async def test_a_closed_loop_subscriber_is_ignored(caplog: pytest.LogCaptureFixture):
    stream = MemoryEventStream()
    loop = asyncio.new_event_loop()
    from agent_adapter.server.events import _Waiter

    waiter = _Waiter(asyncio.get_running_loop())
    waiter.loop = loop
    loop.close()
    stream._waiters["r1"] = {waiter}
    stream.emit(ev("r1"))  # no exception
    assert stream._since("r1", 0)


async def test_a_new_run_is_never_its_own_eviction_victim(caplog: pytest.LogCaptureFixture):
    stream = MemoryEventStream(max_runs=2)
    a, b = stream.subscribe("a"), stream.subscribe("b")
    pa, pb = asyncio.ensure_future(take(a, 1)), asyncio.ensure_future(take(b, 1))
    await following(stream, "a", "b")
    stream.emit(ev("a"))
    stream.emit(ev("b"))
    await pa, await pb
    stream.emit(ev("c"))  # every other run is followed: the oldest of them goes, never "c"
    assert "event stream failed" not in caplog.text
    assert [seq for seq, _ in stream._since("c", 0)] == [1]
    assert set(stream._runs) == {"b", "c"}
    await a.aclose()
    await b.aclose()
