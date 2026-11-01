"""The viewer server: loopback only, Host/Origin/CSRF checks, run data and the approval desk."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.approval import ApprovalRequest, SQLiteApprovalStore
from agent_adapter.core.errors import ApprovalAlreadyDecided, ConfigError, ToolUsageError
from agent_adapter.viewer import ApprovalDesk, Recorder, ViewerServer, check_host
from tests.viewer.helpers import decide_headers, event, request, served

YES = json.dumps({"approved": True, "reason": "fine"}).encode()


def park(db: Path, approval_id: str, **request: Any) -> None:
    fields = {"run_id": "run_1", "call_id": approval_id, "tool": "refund", **request}
    store = SQLiteApprovalStore(db)
    asyncio.run(store.create(ApprovalRequest(approval_id=approval_id, **fields)))


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = tmp_path / "runs"
    recorder = Recorder(folder)
    recorder.emit(event("run.started", 1.0))
    recorder.emit(event("tool.started", 1.1, "c1", tool="t", kind=""))
    recorder.emit(event("run.finished", 1.2, ok=True))
    return folder


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "approvals.db"
    park(path, "apr_1", arguments={"order": "o-1", "api_key": "sk-secret-value"})
    park(path, "apr_node", kind="node", tool="node:f.n")
    park(path, "apr_server", metadata={"principal": "bob"})
    return path


@pytest.fixture
def desk_server(runs: Path, db: Path) -> Iterator[ViewerServer]:
    with served(runs, approvals=db, reviewer="ana") as server:
        yield server


def test_only_the_ipv4_loopback_may_be_bound(runs: Path) -> None:
    check_host("127.0.0.1")
    for host in ("0.0.0.0", "localhost", "::1", "192.168.1.5", ""):
        with pytest.raises(ConfigError, match=r"binds only to 127\.0\.0\.1"):
            check_host(host)
    with pytest.raises(ConfigError), served(runs, host="0.0.0.0"):
        pass  # pragma: no cover - never reached


def test_index_carries_a_fresh_csrf_token_and_security_headers(runs: Path) -> None:
    with served(runs) as first, served(runs) as second:
        status, body, headers = request(first, "GET", "/")
        assert status == 200
        assert first.csrf in body and "__AA_CSRF__" not in body
        assert '<meta name="aa-approvals" content="">' in body
        assert first.csrf != second.csrf
        assert "default-src 'none'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"


def test_static_files_and_unknown_paths(runs: Path) -> None:
    with served(runs) as server:
        status, _, headers = request(server, "GET", "/static/app.js")
        assert status == 200 and headers["content-type"].startswith("text/javascript")
        css = request(server, "GET", "/static/viewer.css")[2]
        assert css["content-type"].startswith("text/css")
        icon = request(server, "GET", "/static/favicon.svg")[2]
        assert icon["content-type"] == "image/svg+xml"
        for path in ("/static/../server.py", "/static/index.html", "/nope", "/api/approvals"):
            assert request(server, "GET", path)[0] == 404


def test_runs_and_run_detail(runs: Path) -> None:
    with served(runs) as server:
        status, body, _ = request(server, "GET", "/api/runs")
        assert status == 200
        (run,) = body["runs"]
        assert run["status"] == "ok"
        status, detail, _ = request(server, "GET", f"/api/runs/{run['key']}?x=1")
        assert status == 200 and [s["name"] for s in detail["spans"]] == ["t"]
        assert request(server, "GET", "/api/runs/run_missing")[0] == 404


def test_new_runs_appear_on_the_next_poll(runs: Path) -> None:
    with served(runs) as server:
        assert len(request(server, "GET", "/api/runs")[1]["runs"]) == 1
        Recorder(runs).emit(event("run.started", 5.0, run_id="run_2"))
        assert len(request(server, "GET", "/api/runs")[1]["runs"]) == 2


def test_a_foreign_host_header_is_refused(runs: Path) -> None:
    with served(runs) as server:
        for host in ("evil.example", f"evil.example:{server.port}", "127.0.0.1:1"):
            status, body, _ = request(server, "GET", "/api/runs", headers={"Host": host})
            assert status == 403 and body["error"] == "unknown Host header"
        local = request(server, "GET", "/", headers={"Host": f"localhost:{server.port}"})
        assert local[0] == 200


def test_reading_errors_answer_500_and_are_logged(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with served(runs) as server:

        def broken() -> None:
            raise OSError("disk gone")

        monkeypatch.setattr(server.folder, "refresh", broken)
        status, body, _ = request(server, "GET", "/api/runs")
        assert status == 500 and "could not read" in body["error"]


def test_approvals_list_masks_secrets_and_says_what_is_not_decidable(
    desk_server: ViewerServer,
) -> None:
    status, body, _ = request(desk_server, "GET", "/api/approvals")
    assert status == 200
    found = {a["approval_id"]: a for a in body["approvals"]}
    assert found["apr_1"]["arguments"] == {"order": "o-1", "api_key": "***"}
    assert found["apr_1"]["decidable"] is True
    assert "FlowRunner.resume" in found["apr_node"]["why_not"]
    assert "HTTP tool server" in found["apr_server"]["why_not"]
    index = request(desk_server, "GET", "/")[1]
    assert '<meta name="aa-approvals" content="on">' in index


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"Origin": ""}, "Origin"),
        ({"Origin": "http://evil.example"}, "Origin"),
        ({"Origin": "http://127.0.0.1:1"}, "Origin"),
        ({"X-AA-CSRF": ""}, "CSRF"),
        ({"X-AA-CSRF": "guess"}, "CSRF"),
        ({"Content-Type": "text/plain"}, "Content-Type"),
    ],
)
def test_decide_refuses_bad_origin_csrf_or_content_type(
    desk_server: ViewerServer, db: Path, change: dict[str, str], reason: str
) -> None:
    headers = decide_headers(desk_server, **change)
    status, body, _ = request(
        desk_server, "POST", "/api/approvals/apr_1/decide", body=YES, headers=headers
    )
    assert status == 403 and reason in body["error"]
    record = asyncio.run(SQLiteApprovalStore(db).get("apr_1"))
    assert record.status == "pending"


def test_decide_records_the_reviewer_once_then_refuses(desk_server: ViewerServer, db: Path) -> None:
    headers = decide_headers(desk_server)
    path = "/api/approvals/apr_1/decide"
    status, body, _ = request(desk_server, "POST", path, body=YES, headers=headers)
    assert status == 200 and body["approval"]["status"] == "approved"
    record = asyncio.run(SQLiteApprovalStore(db).get("apr_1"))
    assert record.decision is not None
    assert (record.decision.decided_by, record.decision.reason) == ("ana", "fine")
    assert record.consumed is False  # the viewer never runs the tool
    again = request(desk_server, "POST", path, body=YES, headers=headers)
    assert again[0] == 409 and again[1]["error"] == "already approved"


def test_decide_refuses_node_and_server_records_and_unknown_ids(
    desk_server: ViewerServer,
) -> None:
    headers = decide_headers(desk_server)
    for approval_id, status in (("apr_node", 409), ("apr_server", 409), ("apr_nope", 404)):
        path = f"/api/approvals/{approval_id}/decide"
        assert request(desk_server, "POST", path, body=YES, headers=headers)[0] == status


@pytest.mark.parametrize(
    "body", [b"", b"{", b"[]", b'{"approved": "yes"}', b"x" * 5000, b'{"approved": 1}']
)
def test_decide_needs_a_small_json_body_with_a_boolean(
    desk_server: ViewerServer, body: bytes
) -> None:
    headers = decide_headers(desk_server)
    status, found, _ = request(
        desk_server, "POST", "/api/approvals/apr_1/decide", body=body, headers=headers
    )
    assert status == 400 and "approved" in found["error"]


def test_decide_with_a_non_string_reason_stores_no_reason(
    desk_server: ViewerServer, db: Path
) -> None:
    body = json.dumps({"approved": False, "reason": 5}).encode()
    headers = decide_headers(desk_server)
    status, found, _ = request(
        desk_server, "POST", "/api/approvals/apr_1/decide", body=body, headers=headers
    )
    assert status == 200 and found["approval"]["status"] == "denied"


def test_post_elsewhere_or_without_a_desk_and_other_methods(runs: Path) -> None:
    with served(runs) as server:
        headers = decide_headers(server)
        no_desk = request(server, "POST", "/api/approvals/a/decide", body=YES, headers=headers)
        assert no_desk[0] == 404
        assert request(server, "PUT", "/api/runs", body=b"{}")[0] == 405
        assert request(server, "DELETE", "/api/runs")[0] == 405
        foreign = request(server, "POST", "/x", body=YES, headers={"Host": "evil.example"})
        assert foreign[0] == 403


def test_a_store_failure_while_deciding_answers_500(
    desk_server: ViewerServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert desk_server.desk is not None

    def corrupt(approval_id: str, **_: Any) -> dict[str, Any]:
        raise ConfigError("database is locked")

    monkeypatch.setattr(desk_server.desk, "decide", corrupt)
    status, body, _ = request(
        desk_server,
        "POST",
        "/api/approvals/apr_1/decide",
        body=YES,
        headers=decide_headers(desk_server),
    )
    assert status == 500 and "not stored" in body["error"]


def test_desk_refuses_a_missing_database_and_a_missing_reviewer(tmp_path: Path, db: Path) -> None:
    with pytest.raises(ConfigError, match="no approvals database"):
        ApprovalDesk(tmp_path / "missing.db", "ana")
    for reviewer in ("", "   ", "x" * 200):
        with pytest.raises(ConfigError, match="reviewer"):
            ApprovalDesk(db, reviewer)


def test_desk_decide_directly(db: Path) -> None:
    desk = ApprovalDesk(db, " ana ")
    assert desk.reviewer == "ana"
    assert desk.decide("apr_1", approved=False, reason="r" * 900)["status"] == "denied"
    with pytest.raises(ApprovalAlreadyDecided):
        desk.decide("apr_1", approved=True)
    with pytest.raises(ToolUsageError, match="flow-node"):
        desk.decide("apr_node", approved=True)
    assert [a["approval_id"] for a in desk.pending()] == ["apr_server", "apr_node"]
