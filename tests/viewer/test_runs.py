"""Reading a runs folder: tailing, status, cost, framework, single files and rotation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.viewer import Recorder, RunFolder
from agent_adapter.viewer._files import list_segments, rotate, segment_name
from agent_adapter.viewer.recorder import flush_all
from tests.viewer.helpers import event


def list_segments_flushed(folder: Path) -> list[Any]:
    assert flush_all()
    return list_segments(folder)


def summary(folder: Path) -> dict[str, Any]:
    runs = RunFolder(folder)
    assert flush_all()
    runs.refresh()
    (found,) = runs.summaries()
    return found


def test_status_follows_run_finished_and_open_approvals(tmp_path: Path) -> None:
    ok, failed, paused = (tmp_path / name for name in ("ok", "failed", "paused"))
    Recorder(ok).emit(event("run.finished", 1.0, ok=True))
    Recorder(failed).emit(event("run.finished", 1.0, ok=False))
    recorder = Recorder(paused)
    recorder.emit(event("tool.pending", 1.0, "c", tool="t", approval_id="a"))
    recorder.emit(event("run.finished", 2.0, ok=True))
    assert summary(ok)["status"] == "ok"
    assert summary(failed)["status"] == "failed"
    assert summary(paused)["status"] == "paused"
    recorder.emit(event("tool.started", 3.0, "c", tool="t", kind=""))
    assert summary(paused)["status"] == "ok"  # the resumed call settles it


def test_a_run_without_run_finished_is_running_then_unfinished(tmp_path: Path) -> None:
    Recorder(tmp_path).emit(event("tool.started", time.time(), "c", tool="t", kind=""))
    assert summary(tmp_path)["status"] == "running"
    (segment,) = list_segments_flushed(tmp_path)
    old = time.time() - 3600
    os.utime(segment.path, (old, old))
    assert summary(tmp_path)["status"] == "unfinished"


def test_cost_sums_priced_events_and_is_none_when_nothing_was_priced(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "priced")
    recorder.emit(event("llm.finished", 1.0, "a", cost_usd=0.5, provider="p"))
    recorder.emit(event("llm.finished", 2.0, "b", cost_usd=None))
    recorder.emit(event("context.compressed", 3.0, "c", summary_cost_usd=0.25))
    assert summary(tmp_path / "priced")["cost_usd"] == 0.75
    Recorder(tmp_path / "free").emit(event("tool.started", 1.0, "c", tool="t", kind=""))
    assert summary(tmp_path / "free")["cost_usd"] is None


def test_framework_comes_from_source_then_events_then_providers(tmp_path: Path) -> None:
    Recorder(tmp_path / "a", source="celery").emit(event("custom.x", 1.0, framework="lg"))
    Recorder(tmp_path / "b").emit(event("custom.x", 1.0, backend="fastapi"))
    Recorder(tmp_path / "c").emit(
        event("llm.started", 1.0, "m", provider="openai", model="x", stage="s", stream=False)
    )
    assert summary(tmp_path / "a")["framework"] == "celery"
    assert summary(tmp_path / "b")["framework"] == "fastapi"
    assert summary(tmp_path / "c")["framework"] == "openai"


def test_refresh_reads_only_new_complete_lines(tmp_path: Path) -> None:
    path = tmp_path / segment_name(1000, "run_x", 1)
    good = json.dumps({"kind": "event", "type": "run.started", "ts": 1.0, "data": {}})
    path.write_text(good + "\nnot json\n[1]\n" + good[:20], encoding="utf-8")
    runs = RunFolder(tmp_path)
    assert flush_all()
    runs.refresh()
    assert runs.summaries()[0]["events"] == 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(good[20:] + "\n")
    assert flush_all()
    runs.refresh()
    assert runs.summaries()[0]["events"] == 2


def test_events_without_a_type_or_time_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / segment_name(1000, "run_x", 1)
    rows = [
        {"kind": "event", "type": "run.started", "ts": True},
        {"kind": "event", "ts": 1.0},
        {"kind": "event", "type": "run.started", "ts": 2.0, "data": "oops"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    runs = RunFolder(tmp_path)
    assert flush_all()
    runs.refresh()
    detail = runs.detail("run_x")
    assert detail is not None
    assert [e["data"] for e in detail["events"]] == [{}]


def test_detail_has_spans_graphs_and_tools_and_none_for_unknown(tmp_path: Path) -> None:
    graph = {"flow": "f", "start": "a", "nodes": [{"id": "a"}], "edges": []}
    recorder = Recorder(tmp_path)
    recorder.take_flow_graph("run_1", "f", graph)
    recorder.emit(event("flow.started", 1.0, flow="f"))
    recorder.emit(event("flow.finished", 2.0, flow="f", status="done", steps=1))
    runs = RunFolder(tmp_path)
    assert flush_all()
    runs.refresh()
    detail = runs.detail("run_1")
    assert detail is not None
    assert detail["graphs"]["f"]["nodes"] == [{"id": "a", "layer": 0, "order": 0, "x": 0.0}]
    assert detail["tools"] is None and detail["truncated"] is False
    assert detail["summary"]["flows"] == ["f"]
    assert runs.detail("run_nope") is None


def test_a_single_run_file_can_be_the_target(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("run.started", 1.0))
    recorder.emit(event("run.started", 1.0, run_id="run_2"))
    first = list_segments_flushed(tmp_path)[0].path
    runs = RunFolder(first)
    assert flush_all()
    runs.refresh()
    assert [s["key"] for s in runs.summaries()] == ["run_1"]
    first.unlink()
    assert flush_all()
    runs.refresh()
    assert runs.summaries() == []


def test_a_file_that_is_not_a_run_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a run file"):
        RunFolder(other)


def test_a_missing_folder_has_no_runs_and_rotated_runs_disappear(tmp_path: Path) -> None:
    runs = RunFolder(tmp_path / "missing")
    assert flush_all()
    runs.refresh()
    assert runs.summaries() == []
    recorder = Recorder(tmp_path / "live")
    for n in range(3):
        recorder.emit(event("run.started", float(n + 1), run_id=f"run_{n}"))
    runs = RunFolder(tmp_path / "live")
    assert flush_all()
    runs.refresh()
    assert len(runs.summaries()) == 3
    assert rotate(tmp_path / "live", 1, set(), grace_s=0) == 2
    assert flush_all()
    runs.refresh()
    assert [s["key"] for s in runs.summaries()] == ["run_2"]


def test_rotation_logs_a_file_it_cannot_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(tmp_path)
    for n in range(2):
        recorder.emit(event("run.started", float(n + 1), run_id=f"run_{n}"))

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "unlink", refuse)
    assert rotate(tmp_path, 1, set(), grace_s=0) == 0


def test_rotation_skips_a_file_another_process_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(tmp_path)
    for n in range(2):
        recorder.emit(event("run.started", float(n + 1), run_id=f"run_{n}"))

    def gone(self: Path, missing_ok: bool = False) -> None:
        raise FileNotFoundError(self)

    monkeypatch.setattr(Path, "unlink", gone)
    assert rotate(tmp_path, 1, set(), grace_s=0) == 0


def test_summaries_are_cached_until_a_file_grows(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.emit(event("run.started", 1.0))
    runs = RunFolder(tmp_path)
    assert flush_all()
    runs.refresh()
    first = runs.summaries()[0]
    assert runs.summaries()[0] is first
    recorder.emit(event("run.finished", 2.0, ok=True))
    assert flush_all()
    runs.refresh()
    assert runs.summaries()[0]["events"] == 2


def test_a_malformed_graph_is_skipped_and_the_run_still_reads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(tmp_path)
    recorder.take_flow_graph("run_1", "bad", {"nodes": [{"name": "no id"}], "edges": []})
    good = {"flow": "f", "start": "a", "nodes": [{"id": "a"}], "edges": []}
    recorder.take_flow_graph("run_1", "f", good)
    recorder.emit(event("flow.started", 1.0, flow="f"))
    runs = RunFolder(tmp_path)
    assert flush_all()
    runs.refresh()
    detail = runs.detail("run_1")
    assert detail is not None and list(detail["graphs"]) == ["f"]
    assert "malformed graph" in caplog.text


def test_a_malformed_graph_is_reported_once_while_the_run_grows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = Recorder(tmp_path)
    recorder.take_flow_graph("run_1", "bad", {"nodes": [{"name": "no id"}], "edges": []})
    runs = RunFolder(tmp_path)
    for ts in (1.0, 2.0, 3.0):
        recorder.emit(event("custom.tick", ts))
        assert flush_all()
        runs.refresh()
        detail = runs.detail("run_1")
        assert detail is not None and detail["events_total"] == int(ts)
    assert caplog.text.count("malformed graph") == 1
