"""The evals copy of the run file format agrees with the viewer's; status and tag edge cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.evals import _run_files as files
from agent_adapter.viewer import _files as viewer_files
from agent_adapter.viewer import _tags as viewer_tags
from tests.viewer.helpers import refreshed


def write(folder: Path, name: str, *records: dict[str, Any], tail: bytes = b"") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    body = b"".join(json.dumps(r).encode() + b"\n" for r in records)
    path.write_bytes(body + tail)
    return path


def ev(kind: str, call: str = "", parent: str = "", **data: Any) -> dict[str, Any]:
    return {"kind": "event", "type": kind, "call_id": call, "parent_id": parent, "ts": 1.0,
            "data": data}  # fmt: skip


def test_the_format_constants_match_the_viewer() -> None:
    assert files.SEGMENT.pattern == viewer_files._NAME.pattern
    assert (files.TAG_PREFIX, files.TAG_SUFFIX) == (viewer_tags.PREFIX, viewer_tags.SUFFIX)
    assert files.MAX_TAG_FILE == viewer_tags.MAX_FILE and files.MAX_TAGS == viewer_tags.MAX_TAGS


@pytest.mark.parametrize(
    ("events", "ended", "status"),
    [
        ([ev("run.finished", ok=True)], False, "ok"),
        ([ev("run.finished", ok=True), ev("run.finished", ok=False)], False, "failed"),
        ([ev("tool.pending", "c1")], False, "paused"),
        ([ev("tool.pending", "c1"), ev("tool.finished", "c1")], True, "ok"),
        ([ev("tool.failed", "c1")], True, "failed"),
        ([ev("tool.failed", "c1", parent="n1")], True, "ok"),
        ([ev("tool.started", "c1")], False, "unfinished"),
        ([ev("llm.started", "l1"), ev("llm.finished", "l1")], False, "ok"),
        ([ev("flow.started", flow="f")], False, "unfinished"),
        (
            [ev("run.started"), ev("llm.started", "l1"), ev("llm.finished", "l1")],
            True,
            "unfinished",
        ),
    ],
)
def test_status_matches_the_viewer(
    tmp_path: Path, events: list[dict[str, Any]], ended: bool, status: str
) -> None:
    end = [{"kind": "end", "ts": 2.0, "inferred": True}] if ended else []
    meta = {"kind": "meta", "run_id": "r", "capture_content": True}
    write(tmp_path, "0000000000001.r.1.jsonl", meta, *events, *end)
    (run,) = files.read_runs(tmp_path)
    assert run.status == status
    if status != "unfinished":  # the viewer says "running" for a fresh unfinished file
        assert refreshed(tmp_path).summaries()[0]["status"] == status


def test_segments_join_and_bad_lines_are_skipped(tmp_path: Path) -> None:
    meta = {"kind": "meta", "run_id": "full id", "capture_content": True}
    write(tmp_path, "0000000000001.r.1.jsonl", meta, ev("run.started"), tail=b"{bad\n[1]\n")
    write(tmp_path, "0000000000002.r.2.jsonl", ev("run.finished", ok=True), tail=b'{"half":')
    write(tmp_path, "0000000000003.s.1.jsonl", {"kind": "event", "type": 3})
    (tmp_path / "notes.jsonl").write_text("{}\n", encoding="utf-8")
    first, second = files.read_runs(tmp_path)
    assert (first.key, first.run_id, first.status, first.capture_content) == (
        "r",
        "full id",
        "ok",
        True,
    )
    assert [e["type"] for e in first.events] == ["run.started", "run.finished"]
    assert (second.run_id, second.events, second.capture_content) == ("s", (), False)


def test_tag_sidecars_bad_huge_or_odd_give_no_tags(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write(tmp_path, "0000000000001.a.1.jsonl", ev("run.finished", ok=True))
    for key in ("b", "c", "d", "e"):
        write(tmp_path, f"0000000000001.{key}.1.jsonl", ev("run.finished", ok=True))
    (tmp_path / "tags.a.json").write_text('{"tags": ["vip", "vip", 3, ""]}', encoding="utf-8")
    (tmp_path / "tags.b.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "tags.c.json").write_text(" " * (files.MAX_TAG_FILE + 1), encoding="utf-8")
    (tmp_path / "tags.d.json").write_text('{"tags": "vip"}', encoding="utf-8")
    (tmp_path / "tags.e.json").mkdir()
    with caplog.at_level(logging.WARNING):
        tags = {run.key: run.tags for run in files.read_runs(tmp_path)}
    assert tags == {"a": ("vip",), "b": (), "c": (), "d": (), "e": ()}
    assert "JSONDecodeError" in caplog.text and "{bad" not in caplog.text


def test_a_segment_deleted_while_reading_is_skipped(tmp_path: Path) -> None:
    gone = tmp_path / "0000000000001.r.1.jsonl"
    assert list(files._records(gone)) == []
