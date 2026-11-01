"""Sink routing around the executor hub: fall-through, no double delivery, stop and fork."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

import agent_adapter as aa
from agent_adapter import CallableModel, Executor, Flow, FlowRunner, LLMRequest, Message, Registry
from agent_adapter.exec.defaults import (
    default_executor,
    reset_default_executor,
    set_default_executor,
)
from agent_adapter.tracing import EventLog, FanoutSink, SinkHub
from agent_adapter.tracing.sinks import first_sink
from agent_adapter.viewer import record
from agent_adapter.viewer import recorder as recorder_module
from tests.viewer.helpers import event, lines, segments


def registry() -> Registry:
    found = Registry()

    @found.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return found


class State(BaseModel):
    n: int = 0


def tiny() -> Flow[State]:
    return Flow("tiny", state=State).node("only", lambda s: {"n": 1}).end("only")


def test_first_sink_treats_an_empty_hub_as_no_sink() -> None:
    log = EventLog()
    assert first_sink(SinkHub(), log) is log
    assert first_sink(None, log) is log
    full = SinkHub(log)
    assert first_sink(full, None) is full


async def test_a_context_of_a_sinkless_executor_still_reaches_the_running_executor() -> None:
    log = EventLog()
    quiet, loud = Executor(registry()), Executor(registry(), sink=log)
    assert await loud.run("add", {"a": 1, "b": 2}, quiet.context()) == 3
    assert [e.type for e in log.events] == ["tool.started", "tool.finished"]


async def test_a_model_of_a_sinkless_executor_reports_to_the_calls_sink() -> None:
    log = EventLog()
    quiet, loud = Executor(registry()), Executor(registry(), sink=log)
    model = quiet.model(CallableModel(lambda request: "hi", provider="p", model="m"))
    async with loud.run_scope() as ctx:
        request = LLMRequest(messages=(Message(role="user", content="x"),))
        assert (await model.bind(ctx).generate(request)).text == "hi"
    assert [e.type for e in log.of_type("llm.")] == ["llm.started", "llm.finished"]


async def test_a_flow_on_a_sinkless_context_still_reaches_the_executor() -> None:
    log = EventLog()
    quiet, loud = Executor(Registry()), Executor(Registry(), sink=log)
    await FlowRunner(loud).run(tiny(), {}, quiet.context())
    assert len(log.of_type("flow.started")) == 1


async def test_a_runner_sink_already_in_the_run_is_not_delivered_twice() -> None:
    log, other = EventLog(), EventLog()
    executor = Executor(Registry(), sink=log)
    await FlowRunner(executor, sink=log).run(tiny(), {})
    assert len(log.of_type("flow.started")) == 1
    await FlowRunner(executor, sink=FanoutSink(log, other)).run(tiny(), {})
    assert len(log.of_type("flow.started")) == 2  # once more, not twice more
    assert len(other.of_type("flow.started")) == 1


def test_stop_detaches_the_recorder_and_ignores_later_events(tmp_path: Path) -> None:
    executor = Executor(registry())
    recorder = record(executor, folder=tmp_path)
    executor.run_sync("add", {"a": 1, "b": 1})
    assert recorder.stop()
    assert executor.sink.sinks == () and recorder.stopped
    executor.run_sync("add", {"a": 2, "b": 2})
    recorder.emit(event("run.started", 1.0, run_id="run_late"))
    written = [line.get("type") for s in segments(tmp_path) for line in lines(s.path)]
    assert written.count("tool.finished") == 1
    assert record(executor, folder=tmp_path) is not recorder  # a fresh one can start again


def test_remove_sink_keeps_the_base_and_reports_misses() -> None:
    log, extra = EventLog(), EventLog()
    executor = Executor(Registry(), sink=log)
    executor.add_sink(extra)
    assert executor.remove_sink(extra) and not executor.remove_sink(extra)
    assert not executor.remove_sink(log)  # the constructor's sink stays
    assert executor.sink.sinks == (log,)


def test_the_default_recorder_moves_with_the_default_and_stops(tmp_path: Path) -> None:
    reset_default_executor()
    try:
        recorder = record(folder=tmp_path)
        first = default_executor()
        second = Executor(registry())
        set_default_executor(second)
        assert recorder in second.sink.sinks and recorder not in first.sink.sinks
        assert recorder.stop()
        assert recorder not in second.sink.sinks
        assert recorder not in aa.configure(registry()).sink.sinks  # no longer follows
    finally:
        reset_default_executor()


def test_reset_detaches_default_sinks_from_the_dropped_default(tmp_path: Path) -> None:
    reset_default_executor()
    recorder = record(folder=tmp_path)
    old = default_executor()
    reset_default_executor()
    assert recorder not in old.sink.sinks
    assert recorder not in default_executor().sink.sinks
    reset_default_executor()


def test_a_forked_child_never_waits_on_the_parents_queue(tmp_path: Path) -> None:
    recorder = recorder_module.Recorder(tmp_path)
    recorder.emit(event("run.started", 1.0))
    assert recorder.close()
    # The state a fork copies: a queue with work only the parent's (absent) writer would do.
    recorder._pid = os.getpid()
    parent_queue = recorder._queue
    parent_queue.put("left by the parent")
    assert recorder.flush(timeout=0.05) is False  # without the fork hook it would wait
    recorder_module._after_fork_in_child()  # what os.register_at_fork runs in a child
    assert recorder._queue is not parent_queue
    started = time.monotonic()
    assert recorder.flush(timeout=5.0)
    assert time.monotonic() - started < 1.0
    recorder.emit(event("run.finished", 2.0, ok=True))  # the child starts its own writer
    assert recorder.flush()
    written = [line.get("type") for s in segments(tmp_path) for line in lines(s.path)]
    assert written.count("run.finished") == 1
    parent_queue.get()
    parent_queue.task_done()
    assert recorder.close()


def test_stop_wins_against_concurrent_emits_and_no_writer_restarts(tmp_path: Path) -> None:
    for trial in range(20):
        recorder = recorder_module.Recorder(tmp_path / str(trial))
        name = f"agent-adapter-recorder-{id(recorder):x}"
        go = threading.Event()

        def hammer(recorder: recorder_module.Recorder = recorder, go: threading.Event = go) -> None:
            go.wait()
            for n in range(300):
                recorder.emit(event("custom.tick", float(n)))

        threads = [threading.Thread(target=hammer) for _ in range(6)]
        for thread in threads:
            thread.start()
        go.set()
        assert recorder.stop()
        for thread in threads:
            thread.join(timeout=10)
        assert recorder._writer() is None
        time.sleep(0.02)  # a writer started after stop() would be alive now
        assert not [t for t in threading.enumerate() if t.name == name and t.is_alive()]
        before = sum(len(lines(s.path)) for s in segments(tmp_path / str(trial)))
        recorder.emit(event("custom.late", 1.0))
        assert sum(len(lines(s.path)) for s in segments(tmp_path / str(trial))) == before


def test_a_hub_base_is_read_only() -> None:
    hub = SinkHub(EventLog())
    with pytest.raises(AttributeError):
        hub.base = EventLog()  # type: ignore[misc]
