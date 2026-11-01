"""The recorder across processes and attach paths: liveness, plain calls, the sink hub, queueing."""

from __future__ import annotations

import logging
import os
import queue
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import agent_adapter as aa
from agent_adapter import CallableModel, Executor, Flow, FlowRunner, LLMRequest, Message, Registry
from agent_adapter.core.errors import ConfigError, ToolFailed
from agent_adapter.exec._sink_hub import offer_flow_graph
from agent_adapter.exec.defaults import default_executor, reset_default_executor
from agent_adapter.tracing import EventLog
from agent_adapter.viewer import Recorder, record
from agent_adapter.viewer import _segments as segments_module
from agent_adapter.viewer._files import ended, rotate
from tests.viewer.helpers import event, lines, refreshed, segments


def adder() -> Registry:
    registry = Registry()

    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @registry.tool
    def boom() -> int:
        """Always fails."""
        raise RuntimeError("no")

    return registry


class State(BaseModel):
    n: int = 0


def one_node_flow() -> Flow[State]:
    return Flow("tiny", state=State).node("only", lambda s: {"n": 1}).end("only")


def test_rotation_keeps_another_process_open_run(tmp_path: Path) -> None:
    other = Recorder(tmp_path)  # another worker: its open runs are not in our active set
    other.emit(event("run.started", 1.0, run_id="run_live"))
    mine = Recorder(tmp_path)
    for n in range(3):
        mine.emit(event("run.started", float(n + 2), run_id=f"run_{n}"))
        mine.emit(event("run.finished", float(n + 2.5), run_id=f"run_{n}", ok=True))
    assert mine.flush() and other.flush()
    assert rotate(tmp_path, 1, set()) > 0
    assert "run_live" in {s.key for s in segments(tmp_path)}
    stale = time.time() - 3600  # silent for an hour: the worker died, so it may go
    for segment in segments(tmp_path):
        os.utime(segment.path, (stale, stale))
    mine.emit(event("run.started", 9.0, run_id="run_new"))
    mine.emit(event("run.finished", 9.5, run_id="run_new", ok=True))
    assert mine.flush()
    rotate(tmp_path, 1, set())
    assert "run_live" not in {s.key for s in segments(tmp_path)}


def test_ended_reads_the_tail_and_a_missing_file_counts_as_ended(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"kind":"event","type":"run.started"}\n', encoding="utf-8")
    assert not ended(path)
    path.write_text('{"kind":"end","ts":1}\n', encoding="utf-8")
    assert ended(path)
    assert ended(tmp_path / "missing.jsonl")


async def test_plain_calls_end_their_run_and_rotate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(segments_module, "ROTATE_EVERY_S", 0.0)
    executor = Executor(adder())
    recorder = Recorder(tmp_path, keep_runs=3, registry=executor.registry)
    executor.add_sink(recorder)
    for n in range(6):
        assert await executor.run("add", {"a": n, "b": 1}) == n + 1
        assert recorder.flush()
    with pytest.raises(ToolFailed):
        await executor.run("boom")
    assert recorder.flush()  # the writer thread may not have written the failed run yet
    runs = refreshed(tmp_path).summaries()
    assert len(runs) <= 4  # keep_runs holds: plain runs end, so rotation may delete them
    assert runs[0]["status"] == "failed"
    assert {s["status"] for s in runs[1:]} == {"ok"}
    (newest,) = [s for s in segments(tmp_path) if s.key == runs[0]["key"]]
    assert lines(newest.path)[-1] == {
        "kind": "end",
        "ts": lines(newest.path)[-2]["ts"],
        "inferred": True,
    }


def test_contexts_and_models_made_before_record_still_reach_it(tmp_path: Path) -> None:
    executor = Executor(adder())
    ctx = executor.context()
    model = executor.model(CallableModel(lambda request: "hi", provider="p", model="m"))
    recorder = record(executor, folder=tmp_path)
    executor.run_sync("add", {"a": 1, "b": 2}, ctx)
    reply = aa.run_sync(model.generate(LLMRequest(messages=(Message(role="user", content="x"),))))
    assert reply.text == "hi"
    assert recorder.flush()
    kinds = [line.get("type") for s in segments(tmp_path) for line in lines(s.path)]
    assert "tool.finished" in kinds and "llm.finished" in kinds


async def test_a_flow_runner_sink_adds_to_the_recorder_instead_of_hiding_it(
    tmp_path: Path,
) -> None:
    executor = Executor(Registry())
    recorder = record(executor, folder=tmp_path)
    own = EventLog()
    result = await FlowRunner(executor, sink=own).run(one_node_flow(), {})
    assert result.status == "done"
    assert own.of_type("flow.started")
    detail = refreshed(tmp_path).summaries()[0]
    assert detail["flows"] == ["tiny"]
    assert recorder.flush()
    (segment,) = segments(tmp_path)
    graph_lines = [line for line in lines(segment.path) if line["kind"] == "graph"]
    assert [line["flow"] for line in graph_lines] == ["tiny"]
    started = own.of_type("flow.started")[0]
    assert "graph" not in started.data  # generic sinks never get the graph


def test_replace_keeps_added_sinks_unless_the_sink_is_replaced(tmp_path: Path) -> None:
    executor = Executor(Registry())
    recorder = record(executor, folder=tmp_path)
    assert executor.replace(namespace="x").sink.sinks == (recorder,)
    other = EventLog()
    assert executor.replace(sink=other).sink.sinks == (other,)
    assert executor.sink.sinks == (recorder,)  # the original is never changed


def test_record_follows_the_default_executor_through_configure(tmp_path: Path) -> None:
    reset_default_executor()
    try:
        recorder = record(folder=tmp_path)
        replaced = aa.configure(adder())
        assert recorder in replaced.sink.sinks
        assert default_executor() is replaced
        assert aa.call("add", a=2, b=3) == 5
        assert recorder.registry() is replaced.registry  # the catalogue follows too
        assert recorder.flush()
        assert any(
            line.get("type") == "tool.finished"
            for s in segments(tmp_path)
            for line in lines(s.path)
        )
    finally:
        reset_default_executor()
    assert recorder not in default_executor().sink.sinks
    reset_default_executor()


def test_record_refuses_other_options_for_the_same_folder(tmp_path: Path) -> None:
    executor = Executor(Registry())
    record(executor, folder=tmp_path)
    with pytest.raises(ConfigError, match="already recorded"):
        record(executor, folder=tmp_path, capture_content=True)


def test_a_full_queue_drops_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(tmp_path)
    full: queue.Queue[Any] = queue.Queue(1)
    full.put("busy")
    monkeypatch.setattr(recorder, "_writer", lambda: full)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.viewer.recorder"):
        for n in range(3):
            recorder.emit(event("custom.x", float(n)))
    assert recorder.dropped == 3
    assert len([r for r in caplog.records if "queue full" in r.getMessage()]) == 1


def test_flush_times_out_and_close_restarts_the_writer(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("run.started", 1.0))
    assert recorder.close()
    assert recorder.close()  # already stopped
    recorder.emit(event("run.finished", 2.0, ok=True))  # a new writer starts
    assert recorder.flush()
    assert [line["kind"] for line in lines(segments(tmp_path)[0].path)][-1] == "event"
    stuck: queue.Queue[Any] = queue.Queue()
    stuck.put("never done")
    recorder._queue = stuck
    assert recorder.flush(timeout=0.05) is False
    stuck.get()
    stuck.task_done()


def test_a_bad_queued_item_is_logged_and_the_rest_written(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(tmp_path)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.viewer.recorder"):
        recorder._put(("graph", "run_1"))  # malformed: cannot unpack
        recorder.emit(event("run.started", 1.0))
        assert recorder.flush()
    assert recorder.failures == 1
    assert len(segments(tmp_path)) == 1


def test_a_failing_graph_taker_never_breaks_the_flow(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def emit(self, event: Any) -> None:
            return None

        def take_flow_graph(self, run_id: str, flow: str, graph: dict[str, Any]) -> None:
            raise RuntimeError("full disk")

    built: list[int] = []
    with caplog.at_level(logging.WARNING):
        offer_flow_graph(Broken(), "r", "f", lambda: built.append(1) or {})  # type: ignore[func-returns-value]
        offer_flow_graph(EventLog(), "r", "f", lambda: built.append(2) or {})  # type: ignore[func-returns-value]
    assert built == [1]  # no taker, no graph built
    assert any("graph sink failed" in r.getMessage() for r in caplog.records)
