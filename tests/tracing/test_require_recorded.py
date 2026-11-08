"""require_recorded: a required sink anywhere under the executor's sink can stop a call."""

from __future__ import annotations

import threading

import pytest

from agent_adapter.core.types import Event
from agent_adapter.tracing.events import RequiredSink, require_recorded
from agent_adapter.tracing.sinks import EventLog, FanoutSink, SinkHub


class Broken:
    """A required sink that cannot record."""

    def __init__(self, settled: bool = True) -> None:
        self._settled = settled
        self.threads: list[str] = []

    def emit(self, event: Event) -> None:
        return None

    def settled(self) -> bool:
        return self._settled

    def require(self) -> None:
        self.threads.append(threading.current_thread().name)
        raise RuntimeError("audit down")


async def test_no_sink_and_plain_sinks_pass() -> None:
    await require_recorded(None)
    await require_recorded(SinkHub(EventLog(), (FanoutSink(EventLog()),)))


async def test_a_broken_required_sink_raises_through_hubs_and_fan_outs() -> None:
    broken = Broken()
    assert isinstance(broken, RequiredSink)
    hub = SinkHub(EventLog(), (FanoutSink(EventLog(), broken),))
    with pytest.raises(RuntimeError, match="audit down"):
        await require_recorded(hub)
    assert broken.threads == [threading.current_thread().name]  # settled: no thread hop


async def test_an_unsettled_sink_is_waited_for_off_the_event_loop() -> None:
    busy = Broken(settled=False)
    with pytest.raises(RuntimeError):
        await require_recorded(busy)
    assert busy.threads and busy.threads[0] != threading.current_thread().name


async def test_a_sink_reached_twice_is_checked_once() -> None:
    calls: list[int] = []

    class Counting(Broken):
        def require(self) -> None:
            calls.append(1)

    shared = Counting()
    await require_recorded(FanoutSink(shared, FanoutSink(shared)))
    assert calls == [1]
