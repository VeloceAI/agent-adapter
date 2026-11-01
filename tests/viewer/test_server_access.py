"""The viewer's access token, exclusive port, timeouts, paged details and one-run export."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest

from agent_adapter.viewer import Recorder, export, make_server
from agent_adapter.viewer import runs as runs_module
from agent_adapter.viewer.server import _Handler
from tests.viewer.helpers import decide_headers, event, refreshed, request, served

export_module = importlib.import_module("agent_adapter.viewer.export")  # `export` is the function


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = tmp_path / "runs"
    recorder = Recorder(folder)
    recorder.emit(event("run.started", 1.0))
    for n in range(7):
        recorder.emit(event("tool.started", 1.0 + n, f"c{n}", tool="t", kind=""))
    recorder.emit(event("run.finished", 9.0, ok=True))
    recorder.emit(event("run.started", 10.0, run_id="run_2"))
    assert recorder.flush()
    return folder


@pytest.mark.parametrize("path", ["/api/runs", "/api/runs/run_1", "/api/approvals"])
def test_every_api_route_needs_the_token_header(runs: Path, path: str) -> None:
    with served(runs) as server:
        status, body, _ = request(server, "GET", path, auth=False)
        assert status == 401 and "access token" in body["error"]
        for headers in (
            {"X-AA-Token": "guess"},
            {"Cookie": f"aa_viewer_{server.port}={server.token}"},  # cookies never count
            {"Authorization": f"Bearer {server.token}"},
        ):
            assert request(server, "GET", path, auth=False, headers=headers)[0] == 401
        assert request(server, "GET", f"{path}?token={server.token}", auth=False)[0] == 401


def test_the_page_shell_is_public_and_no_cookie_is_ever_set(runs: Path) -> None:
    with served(runs) as server:
        assert server.url == f"http://127.0.0.1:{server.port}/#token={server.token}"
        for path in ("/", "/static/app.js", "/static/viewer.css"):
            status, body, headers = request(server, "GET", path, auth=False)
            assert status == 200 and "set-cookie" not in headers
            assert server.token not in body  # the shell holds no secret
        api = request(server, "GET", "/api/runs")
        assert api[0] == 200 and "set-cookie" not in api[2]


def test_a_decision_needs_the_token_even_with_csrf_and_origin(runs: Path, tmp_path: Path) -> None:
    from tests.viewer.test_server import YES, park

    db = tmp_path / "a.db"
    park(db, "apr_1")
    with served(runs, approvals=db, reviewer="ana") as server:
        headers = decide_headers(server)
        path = "/api/approvals/apr_1/decide"
        assert request(server, "POST", path, body=YES, headers=headers, auth=False)[0] == 401
        assert request(server, "POST", path, body=YES, headers=headers)[0] == 200


def test_refused_requests_warn_once_per_start_then_log_at_debug(
    runs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        caplog.at_level(logging.DEBUG, logger="agent_adapter.viewer.server"),
        served(runs) as server,
    ):
        for _ in range(3):
            assert request(server, "GET", "/api/runs", auth=False)[0] == 401
        token = server.token
    refused = [r for r in caplog.records if "without the access token" in r.getMessage()]
    assert [r.levelno for r in refused] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert all(token not in r.getMessage() for r in caplog.records)


def test_a_second_viewer_cannot_bind_the_same_port(runs: Path) -> None:
    with served(runs) as first, pytest.raises(OSError):
        make_server(runs, port=first.port)


def test_handlers_time_out_idle_clients() -> None:
    assert _Handler.timeout == 10


def test_details_are_paged_with_since(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runs_module, "EVENT_PAGE", 3)
    with served(runs) as server:
        status, first, _ = request(server, "GET", "/api/runs/run_1")
        assert status == 200 and len(first["events"]) == 3 and first["events_total"] == 9
        rest = request(server, "GET", "/api/runs/run_1?since=3")[1]
        assert rest["events_from"] == 3 and rest["events"][0]["call_id"] == "c2"
        assert request(server, "GET", "/api/runs/run_1?since=x")[1]["events_from"] == 0


def test_spans_are_capped_keeping_flows_and_the_latest_calls(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_module, "MAX_SPANS", 3)
    detail = refreshed(runs).detail("run_1")
    assert detail is not None
    assert detail["spans_total"] == 7
    assert [s["call_id"] for s in detail["spans"]] == ["c4", "c5", "c6"]


def test_a_detail_cache_rebuilds_after_rotation_and_stays_bounded(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs_module, "DETAILS_KEPT", 1)
    folder = refreshed(runs)
    first = folder.detail("run_1")
    assert first is not None and folder.detail("run_2") is not None
    assert list(folder._details) == ["run_2"]
    again = folder.detail("run_1")
    assert again is not None and again["events_total"] == first["events_total"]


def test_export_one_run_and_warn_when_large(
    runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(export_module, "LARGE_BYTES", 10)
    with caplog.at_level(logging.WARNING, logger="agent_adapter.viewer.export"):
        out = export(runs, tmp_path / "one.html", run="run_2")
    html = out.read_text(encoding="utf-8")
    assert '"run_2"' in html and '"key": "run_1"' not in html
    assert any("--run KEY" in r.getMessage() for r in caplog.records)
    with pytest.raises(Exception, match="no run"):
        export(runs, tmp_path / "none.html", run="run_missing")
