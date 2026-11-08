"""Search across runs and run tags: sidecar files, the query, and the two routes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.viewer import Recorder
from agent_adapter.viewer._search import MAX_TEXT, SearchFold, matches, parse_query
from agent_adapter.viewer._tags import MAX_FILE, MAX_TAGS, TagStore, clean_tag
from tests.viewer.helpers import decide_headers, event, refreshed, request, served


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = tmp_path / "runs"
    recorder = Recorder(folder, capture_content=True)
    recorder.emit(event("run.started", 1.0, run_id="order_1"))
    recorder.emit(event("tool.started", 1.1, "c1", run_id="order_1", tool="find_ticket", kind=""))
    recorder.emit(
        event(
            "tool.finished", 1.2, "c1", run_id="order_1", tool="find_ticket", latency_ms=1.0,
            attempts=1, cached=False, result={"seat": "Row K"},
        )
    )  # fmt: skip
    recorder.emit(event("run.finished", 2.0, run_id="order_1", ok=True))
    recorder.emit(event("run.started", 3.0, run_id="order_2"))
    recorder.emit(event("flow.started", 3.1, run_id="order_2", flow="refunds"))
    recorder.emit(event("run.finished", 4.0, run_id="order_2", ok=False))
    assert recorder.flush()
    recorder.stop()
    return folder


def _keys(body: dict[str, Any]) -> list[str]:
    return [run["key"] for run in body["runs"]]


# --- tags ---------------------------------------------------------------------------------


def test_clean_tag_normalises_and_refuses_bad_tags() -> None:
    assert clean_tag("  Needs   Review ") == "needs review"
    assert clean_tag("team:shop/v2+1") == "team:shop/v2+1"
    for bad in ("", " ", "-x", "<b>", "a" * 41, 3, "a\nb\x00"):
        with pytest.raises(ToolUsageError):
            clean_tag(bad)


def test_tag_store_adds_removes_and_deletes_an_empty_sidecar(tmp_path: Path) -> None:
    store = TagStore(tmp_path)
    assert store.get("run_1") == [] and store.all() == {}
    assert store.update("run_1", add=["Bug", "bug", "slow"]) == ["bug", "slow"]
    assert json.loads((tmp_path / "tags.run_1.json").read_text("utf-8")) == {
        "v": 1,
        "tags": ["bug", "slow"],
    }
    assert store.update("run_1", add=["vip"], remove=["slow"]) == ["bug", "vip"]
    assert store.all() == {"run_1": ["bug", "vip"]}
    assert store.update("run_1", remove=["bug", "vip"]) == []
    assert not (tmp_path / "tags.run_1.json").exists()
    assert store.all() == {}


def test_tag_store_limits(tmp_path: Path) -> None:
    store = TagStore(tmp_path)
    with pytest.raises(ToolUsageError, match="at most"):
        store.update("run_1", add=[f"t{n}" for n in range(MAX_TAGS + 1)])
    with pytest.raises(ToolUsageError, match="run key"):
        store.update("../x", add=["a"])
    assert store.all() == {}


def test_a_bad_or_huge_sidecar_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "tags.a.json").write_text("{not json", "utf-8")
    (tmp_path / "tags.b.json").write_text(json.dumps({"tags": ["<x>"]}), "utf-8")
    (tmp_path / "tags.c.json").write_bytes(b" " * (MAX_FILE + 1))
    (tmp_path / "tags.bad key.json").write_text(json.dumps({"tags": ["ok"]}), "utf-8")
    with caplog.at_level(logging.WARNING):
        assert TagStore(tmp_path).all() == {}
    assert "<x>" not in caplog.text
    assert TagStore(tmp_path / "missing").all() == {}


def test_tags_written_by_another_process_are_read_again(tmp_path: Path) -> None:
    store = TagStore(tmp_path)
    store.update("run_1", add=["a"])
    path = tmp_path / "tags.run_1.json"
    path.write_text(json.dumps({"v": 1, "tags": ["b"]}), "utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert store.get("run_1") == ["b"]


# --- search -------------------------------------------------------------------------------


def test_parse_query_words_and_filters() -> None:
    query = parse_query("Find tag:VIP status:failed  ticket tag:", tag=" Needs  Review ")
    assert query.words == ("find", "ticket", "tag:")
    assert query.tags == ("needs review", "vip")
    assert query.statuses == ("failed",)
    assert parse_query("").empty and parse_query("x" * 500).words == ("x" * 200,)


def test_search_fold_keeps_names_and_capped_content() -> None:
    fold = SearchFold()
    fold.add({"kind": "meta", "run_id": "Run_X"})
    fold.add(
        {"kind": "event", "type": "tool.started", "data": {"tool": "Book", "arguments": "A" * 10}}
    )
    fold.add({"kind": "event", "type": "llm.started", "data": "not a mapping"})
    fold.add({"kind": "graph"})
    fold.add({"kind": "event", "type": "x", "data": {"result": "b" * (MAX_TEXT * 2)}})
    assert {"run_x", "tool.started", "book", "llm.started"} <= fold.names
    assert fold.text_chars == MAX_TEXT and fold.text[0] == "a" * 10
    fold.add({"kind": "event", "type": "y", "data": {"result": "more"}})
    assert fold.text_chars == MAX_TEXT


def test_matches_needs_every_word_and_filter() -> None:
    fold = SearchFold(names={"find_ticket"}, text=['{"seat": "row k"}'])
    summary = {"key": "order_1", "run_id": "order_1", "status": "ok", "flows": ["refunds"]}
    assert matches(summary, [fold], ["vip"], parse_query("find row"))
    assert matches(summary, [fold], ["vip"], parse_query("refunds tag:vip status:ok"))
    assert not matches(summary, [fold], ["vip"], parse_query("find missing"))
    assert not matches(summary, [fold], [], parse_query("tag:vip"))
    assert not matches(summary, [fold], ["vip"], parse_query("status:failed"))


# --- routes -------------------------------------------------------------------------------


def test_runs_route_lists_tags_and_searches(runs: Path) -> None:
    with served(runs) as server:
        status, body, _ = request(server, "GET", "/api/runs")
        assert status == 200 and _keys(body) == ["order_2", "order_1"]
        assert body["total"] == 2 and body["tags"] == []
        assert all(run["tags"] == [] for run in body["runs"])
        assert _keys(request(server, "GET", "/api/runs?q=find_ticket")[1]) == ["order_1"]
        assert _keys(request(server, "GET", "/api/runs?q=row+k")[1]) == ["order_1"]  # content
        assert _keys(request(server, "GET", "/api/runs?q=refunds")[1]) == ["order_2"]
        assert _keys(request(server, "GET", "/api/runs?q=status:failed")[1]) == ["order_2"]
        assert _keys(request(server, "GET", "/api/runs?q=nothing")[1]) == []


def test_content_is_searched_only_when_it_was_captured(tmp_path: Path) -> None:
    folder = tmp_path / "runs"
    recorder = Recorder(folder)  # capture_content off: arguments never reach disk
    recorder.emit(event("tool.started", 1.0, "c1", tool="find", kind="", arguments={"q": "secret"}))
    recorder.emit(
        event("tool.finished", 1.1, "c1", tool="find", latency_ms=1, attempts=1, cached=False)
    )
    assert recorder.flush()
    recorder.stop()
    with served(folder) as server:
        assert _keys(request(server, "GET", "/api/runs?q=secret")[1]) == []
        assert _keys(request(server, "GET", "/api/runs?q=find")[1]) == ["run_1"]


def test_tags_route_sets_tags_and_filters(runs: Path) -> None:
    with served(runs) as server:
        headers = decide_headers(server)
        body = json.dumps({"add": ["VIP", "slow"]}).encode()
        status, reply, _ = request(
            server, "POST", "/api/runs/order_1/tags", body=body, headers=headers
        )
        assert (status, reply) == (200, {"key": "order_1", "tags": ["vip", "slow"]})
        assert (runs / "tags.order_1.json").is_file()
        listing = request(server, "GET", "/api/runs?tag=vip")[1]
        assert _keys(listing) == ["order_1"] and listing["tags"] == ["slow", "vip"]
        assert listing["runs"][0]["tags"] == ["vip", "slow"]
        assert _keys(request(server, "GET", "/api/runs?q=tag:slow")[1]) == ["order_1"]
        assert _keys(request(server, "GET", "/api/runs?q=vip")[1]) == ["order_1"]
        body = json.dumps({"remove": ["slow"]}).encode()
        reply = request(server, "POST", "/api/runs/order_1/tags", body=body, headers=headers)[1]
        assert reply["tags"] == ["vip"]


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b"{}",
        b'{"add": "vip"}',
        b'{"set": ["vip"]}',
        json.dumps({"add": ["x"] * (MAX_TAGS + 1)}).encode(),
        b"x" * 5000,
    ],
)
def test_tags_route_refuses_bad_bodies(runs: Path, body: bytes) -> None:
    with served(runs) as server:
        status, reply, _ = request(
            server, "POST", "/api/runs/order_1/tags", body=body, headers=decide_headers(server)
        )
        assert status == 400 and "add" in reply["error"]


def test_tags_route_refuses_bad_tags_and_unknown_runs(runs: Path) -> None:
    with served(runs) as server:
        headers = decide_headers(server)
        status, reply, _ = request(
            server, "POST", "/api/runs/order_1/tags", body=b'{"add": ["<b>"]}', headers=headers
        )
        assert status == 400 and "letters" in reply["error"]
        status, reply, _ = request(
            server, "POST", "/api/runs/nope/tags", body=b'{"add": ["a"]}', headers=headers
        )
        assert status == 404
    assert not list(runs.glob("tags.*.json"))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"Origin": "http://evil.example"}, "Origin"),
        ({"X-AA-CSRF": "wrong"}, "CSRF"),
        ({"Content-Type": "text/plain"}, "Content-Type"),
    ],
)
def test_tags_route_checks_origin_csrf_and_type(
    runs: Path, change: dict[str, str], error: str
) -> None:
    with served(runs) as server:
        status, reply, _ = request(
            server,
            "POST",
            "/api/runs/order_1/tags",
            body=b'{"add": ["a"]}',
            headers=decide_headers(server, **change),
        )
        assert status == 403 and error in reply["error"]
        assert request(server, "POST", "/api/runs/order_1/tags", auth=False)[0] == 401
    assert not list(runs.glob("tags.*.json"))


def test_a_failed_tag_write_is_a_500(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> list[str]:
        raise PermissionError("read-only")

    with served(runs) as server:
        monkeypatch.setattr(server.tags, "update", broken)
        status, reply, _ = request(
            server, "POST", "/api/runs/order_1/tags", body=b'{"add": ["a"]}',
            headers=decide_headers(server),
        )  # fmt: skip
        assert status == 500 and reply == {"error": "the tags were not saved"}


def test_a_single_run_file_keeps_tags_next_to_it(runs: Path) -> None:
    (segment,) = [p for p in runs.glob("*.jsonl") if ".order_1." in p.name]
    folder = refreshed(segment)
    assert folder.tags_folder == runs


def test_a_failed_replace_removes_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        TagStore(tmp_path).update("run_1", add=["a"])
    assert list(tmp_path.iterdir()) == []
