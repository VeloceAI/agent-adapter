"""Tags and search edge cases: device-name keys, snapshots, the cache, route guards."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.viewer import Recorder
from agent_adapter.viewer import _runs_routes as routes
from agent_adapter.viewer._runs_routes import SearchCache, runs_listing
from agent_adapter.viewer._tags import TagStore
from tests.viewer.helpers import decide_headers, event, refreshed, request, served


def _within(seconds: float, work: Any) -> Any:
    """`work()` run in a thread; fails instead of hanging when it blocks."""
    box: dict[str, Any] = {}
    thread = threading.Thread(target=lambda: box.update(value=work()), daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "blocked"
    return box["value"]


@pytest.mark.parametrize("key", ["aux", "con", "nul", "prn", "com1", "lpt1"])
def test_windows_device_names_are_plain_run_keys(tmp_path: Path, key: str) -> None:
    store = TagStore(tmp_path)
    assert _within(10, lambda: store.update(key, add=["vip"])) == ["vip"]
    assert (tmp_path / f"tags.{key}.json").is_file()
    assert _within(10, lambda: store.get(key)) == ["vip"]
    assert _within(10, store.all) == {key: ["vip"]}


def test_a_sidecar_that_is_not_a_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "tags.run_1.json").mkdir()
    assert TagStore(tmp_path).get("run_1") == []


def test_get_refuses_a_bad_key(tmp_path: Path) -> None:
    with pytest.raises(ToolUsageError):
        TagStore(tmp_path).get("../x")


def test_a_failed_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(self: Path, *args: object, **kwargs: object) -> int:
        self.write_bytes(b"{")  # half written, then the disk is full
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", full)
    with pytest.raises(OSError):
        TagStore(tmp_path).update("run_1", add=["a"])
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = tmp_path / "runs"
    recorder = Recorder(folder)
    recorder.emit(event("tool.started", 1.0, "c1", tool="find", kind=""))
    recorder.emit(
        event("tool.finished", 1.1, "c1", tool="find", latency_ms=1, attempts=1, cached=False)
    )
    assert recorder.flush()
    recorder.stop()
    return folder


def test_search_folds_are_snapshots(runs: Path) -> None:
    folder = refreshed(runs)
    (snapshot,) = folder.search_folds()["run_1"]
    snapshot.names.add("changed")
    (again,) = folder.search_folds()["run_1"]
    assert "changed" not in again.names and "find" in again.names


def test_the_search_cache_skips_a_repeated_search(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, tags, cache = refreshed(runs), TagStore(runs), SearchCache()
    calls: list[int] = []
    real = folder.search_folds

    def counted() -> Any:
        calls.append(1)
        return real()

    monkeypatch.setattr(folder, "search_folds", counted)
    query = {"q": ["find"]}
    first = runs_listing(folder, tags, query, cache)
    assert [r["key"] for r in first["runs"]] == ["run_1"]
    assert runs_listing(folder, tags, query, cache) == first
    assert len(calls) == 1
    tags.update("run_1", add=["vip"])  # a tag change searches again
    assert runs_listing(folder, tags, query, cache)["runs"][0]["tags"] == ["vip"]
    assert len(calls) == 2
    assert runs_listing(folder, tags, {"q": ["nope"]}, cache)["runs"] == []
    assert routes.runs_listing(folder, tags, query)["runs"]  # no cache: searches directly


@pytest.mark.parametrize(
    "path", ["/api/runs/..%2F..%2Fx/tags", "/api/runs/a.b/tags", "/api/runs//tags"]
)
def test_tag_paths_outside_the_key_pattern_are_not_found(runs: Path, path: str) -> None:
    with served(runs) as server:
        status, _, _ = request(
            server, "POST", path, body=b'{"add": ["a"]}', headers=decide_headers(server)
        )
        assert status == 404
    assert not list(runs.parent.rglob("tags.*.json"))


def test_tags_need_an_origin_and_this_host(runs: Path) -> None:
    with served(runs) as server:
        body = json.dumps({"add": ["a"]}).encode()
        status, reply, _ = request(
            server,
            "POST",
            "/api/runs/run_1/tags",
            body=body,
            headers=decide_headers(server, Origin=""),
        )
        assert status == 403 and "Origin" in reply["error"]
        status, _, _ = request(
            server,
            "POST",
            "/api/runs/run_1/tags",
            body=body,
            headers={**decide_headers(server), "Host": "evil.example"},
        )
        assert status == 403
    assert not list(runs.glob("tags.*.json"))
