"""The recorder: what it writes, what it leaves out, several processes, rotation and attach."""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent_adapter import Executor, Registry
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.exec.defaults import default_executor, reset_default_executor
from agent_adapter.tracing import EventLog
from agent_adapter.viewer import Recorder, record
from agent_adapter.viewer import _segments as segments_module
from agent_adapter.viewer import recorder as recorder_module
from agent_adapter.viewer._files import run_key
from tests.viewer.helpers import event, lines, refreshed, segments

ROOT = Path(__file__).resolve().parents[2]


def only_file(folder: Path) -> Path:
    files = segments(folder)
    assert len(files) == 1
    return files[0].path


def test_recorder_writes_meta_then_events_in_one_file_per_run(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, source="tests")
    recorder.emit(event("run.started", 10.0))
    recorder.emit(event("tool.started", 10.1, "c1", tool="t", kind="tool"))
    recorder.emit(event("run.finished", 10.2, ok=True))
    found = lines(only_file(tmp_path))
    assert [line["kind"] for line in found] == ["meta", "event", "event", "event"]
    assert found[0]["run_id"] == "run_1"
    assert found[0]["source"] == "tests"
    assert found[0]["capture_content"] is False
    assert [line["type"] for line in found[1:]] == ["run.started", "tool.started", "run.finished"]


def test_recorder_leaves_content_out_unless_capture_content(tmp_path: Path) -> None:
    quiet = Recorder(tmp_path / "quiet")
    loud = Recorder(tmp_path / "loud", capture_content=True)
    for sink in (quiet, loud):
        sink.emit(
            event("custom.io", 1.0, arguments={"q": "hi"}, result="answer", output_messages=["x"])
        )
    quiet_line = lines(only_file(tmp_path / "quiet"))[-1]
    assert quiet_line["data"] == {}
    assert quiet_line["omitted"] == ["arguments", "output_messages", "result"]
    loud_line = lines(only_file(tmp_path / "loud"))[-1]
    assert loud_line["data"]["arguments"] == {"q": "hi"}
    assert "omitted" not in loud_line


def test_recorder_masks_secret_named_values_even_with_capture_content(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, capture_content=True)
    recorder.emit(
        event(
            "custom.login",
            1.0,
            api_key="sk-live-123456",
            arguments={"password": "hunter22", "max_tokens": 10},
        )
    )
    data = lines(only_file(tmp_path))[-1]["data"]
    assert data["api_key"] == "***"
    assert data["arguments"] == {"password": "***", "max_tokens": 10}


def test_recorder_clips_long_strings_and_lists(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("custom.big", 1.0, note="x" * 5000, items=list(range(500))))
    data = lines(only_file(tmp_path))[-1]["data"]
    assert data["note"].endswith("... [+3000 chars]")
    assert len(data["items"]) == 201
    assert data["items"][-1] == "... +300 items"


def test_recorder_writes_the_tool_catalogue_without_secret_examples(tmp_path: Path) -> None:
    registry = Registry()

    @registry.tool(confirm=True, cache_ttl=60, effects="destructive", timeout=5.0, retries=2)
    def wipe(path: str, api_key: str = "sk-default-secret") -> str:
        """Delete a path."""
        return path

    Recorder(tmp_path, registry=registry).emit(event("run.started", 1.0))
    tools = lines(only_file(tmp_path))[1]
    assert tools["kind"] == "tools"
    (tool,) = tools["tools"]
    assert tool["name"] == "wipe"
    assert tool["confirm"] == "always"
    assert (tool["cache_ttl"], tool["effects"], tool["timeout"], tool["retries"]) == (
        60,
        "destructive",
        5.0,
        2,
    )
    assert "default" not in tool["schema"]["properties"]["api_key"]
    assert "sk-default-secret" not in only_file(tmp_path).read_text(encoding="utf-8")


def test_recorder_writes_a_flow_graph_once_per_flow(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    graph = {"flow": "f", "nodes": [], "edges": []}
    recorder.take_flow_graph("run_1", "f", graph)
    recorder.emit(event("flow.started", 1.0, flow="f"))
    recorder.take_flow_graph("run_1", "f", graph)
    found = lines(only_file(tmp_path))
    assert [line["kind"] for line in found] == ["meta", "graph", "event"]
    assert found[1]["graph"] == graph


def test_recorder_stops_a_run_at_max_events_with_one_marker(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path, max_events_per_run=2)
    for n in range(5):
        recorder.emit(event("custom.tick", float(n)))
    recorder.emit(event("run.finished", 9.0, ok=True))  # past the cap, a run still ends
    found = lines(only_file(tmp_path))
    assert [line["kind"] for line in found] == ["meta", "event", "event", "truncated", "event"]
    assert found[-1]["type"] == "run.finished"
    assert refreshed(tmp_path).summaries()[0]["status"] == "ok"


def test_recorder_keys_unusual_run_ids_by_hash_and_empty_ones_as_no_run(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("custom.a", 1.0, run_id="../../etc/passwd"))
    recorder.emit(event("custom.b", 2.0, run_id=""))
    keys = {segment.key for segment in segments(tmp_path)}
    assert keys == {run_key("../../etc/passwd"), "no-run"}
    assert all(key.replace("-", "").replace("_", "").isalnum() for key in keys)


def test_a_run_continued_after_run_finished_starts_a_new_segment(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("run.finished", 1.0, ok=True))
    recorder.emit(event("tool.started", 2.0, "c", tool="t", kind=""))
    assert len(segments(tmp_path)) == 2
    (summary,) = refreshed(tmp_path).summaries()
    assert summary["processes"] == 1 and summary["events"] == 2


def test_several_processes_append_to_one_folder_without_tearing_lines(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        from agent_adapter.core.types import Event
        from agent_adapter.viewer import Recorder
        recorder = Recorder(sys.argv[1])
        for n in range(150):
            data = {"n": n, "pad": "x" * 300}
            recorder.emit(Event("custom.tick", run_id="run_shared", data=data))
        """
    )
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT) for _ in range(3)
    ]
    assert [worker.wait(timeout=120) for worker in workers] == [0, 0, 0]
    found = segments(tmp_path)  # nothing flushed by hand: exit wrote every queued event
    assert len(found) == 3
    assert len({segment.pid for segment in found}) == 3
    (summary,) = refreshed(tmp_path).summaries()
    assert summary["events"] == 450
    assert summary["processes"] == 3
    for segment in found:
        assert len(lines(segment.path)) == 151  # every line parses


def test_rotation_keeps_the_newest_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(segments_module, "ROTATE_EVERY_S", 0.0)
    recorder = Recorder(tmp_path, keep_runs=2)
    for n in range(5):
        recorder.emit(event("run.started", float(n + 1), run_id=f"run_{n}"))
        recorder.emit(event("run.finished", float(n + 1) + 0.5, run_id=f"run_{n}", ok=True))
        recorder.flush()
    keys = sorted(segment.key for segment in segments(tmp_path))
    # Rotation runs before the new run's file exists, so one extra run may remain.
    assert keys[-2:] == ["run_3", "run_4"]
    assert len(keys) <= 3


def test_rotation_never_deletes_an_open_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(segments_module, "ROTATE_EVERY_S", 0.0)
    recorder = Recorder(tmp_path, keep_runs=1)
    recorder.emit(event("run.started", 1.0, run_id="run_open"))
    for n in range(3):
        recorder.emit(event("run.started", float(n + 2), run_id=f"run_{n}"))
        recorder.flush()
    assert "run_open" in {segment.key for segment in segments(tmp_path)}


def test_recorder_failures_are_logged_never_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(tmp_path)

    def broken(path: Path, lines: list[bytes]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(recorder_module, "append", broken)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.viewer.recorder"):
        recorder.emit(event("run.started", 1.0))
        recorder.emit(event("run.started", 2.0, run_id="run_2"))
        assert recorder.flush()
    assert recorder.failures == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # rate-limited
    assert "1 failure(s)" in warnings[0].getMessage()


@pytest.mark.parametrize("bad", [0, -1, True, "2"])
def test_recorder_refuses_bad_limits(tmp_path: Path, bad: object) -> None:
    with pytest.raises(ConfigError, match="keep_runs"):
        Recorder(tmp_path, keep_runs=bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="max_events_per_run"):
        Recorder(tmp_path, max_events_per_run=bad)  # type: ignore[arg-type]


def test_recorder_refuses_a_folder_it_cannot_make(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot create the runs folder"):
        Recorder(blocker / "runs")


def test_record_attaches_next_to_the_existing_sink(tmp_path: Path) -> None:
    log = EventLog()
    executor = Executor(Registry(), sink=log)
    recorder = record(executor, folder=tmp_path)
    assert executor.sink.sinks == (log, recorder)
    assert record(executor, folder=tmp_path) is recorder  # attached once
    executor.sink.emit(event("run.started", 1.0))
    assert len(log.events) == 1
    assert len(segments(tmp_path)) == 1


def test_record_without_executor_uses_the_default_one(tmp_path: Path) -> None:
    reset_default_executor()
    try:
        recorder = record(folder=tmp_path, keep_runs=5)
        assert default_executor().sink.sinks == (recorder,)
        assert recorder.keep_runs == 5
    finally:
        reset_default_executor()


def test_add_sink_refuses_a_non_sink() -> None:
    with pytest.raises(ToolUsageError, match="EventSink"):
        Executor(Registry()).add_sink(object())  # type: ignore[arg-type]


async def test_a_recorded_executor_run_lands_in_the_folder(tmp_path: Path) -> None:
    registry = Registry()

    @registry.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    executor = Executor(registry)
    record(executor, folder=tmp_path)
    async with executor.run_scope() as ctx:
        assert await executor.run(add, {"a": 1, "b": 2}, ctx) == 3
    folder = refreshed(tmp_path)
    (summary,) = folder.summaries()
    assert summary["status"] == "ok"
    detail = folder.detail(summary["key"])
    assert detail is not None
    assert [tool["name"] for tool in detail["tools"]] == ["add"]
    assert [span["name"] for span in detail["spans"]] == ["add"]
