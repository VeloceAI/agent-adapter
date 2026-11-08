"""Saving a recorded run as an eval case from the Runs view (`POST /api/runs/KEY/eval-case`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from agent_adapter.evals import load_cases
from agent_adapter.viewer._tags import TagStore
from agent_adapter.viewer.chat import evalcases
from agent_adapter.viewer.server import ViewerServer
from tests.evals.test_from_runs import chat_run, recorded
from tests.viewer.helpers import decide_headers, request, served


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    folder = recorded(tmp_path / "runs", chat_run("good"))
    recorded(folder, chat_run("bad", ok=False))
    recorded(folder, chat_run("quiet"), capture=False)
    TagStore(folder).update("good", add=["vip"])
    return folder


def post(server: ViewerServer, key: str, body: Any, **headers: str) -> tuple[int, Any]:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    status, payload, _ = request(
        server,
        "POST",
        f"/api/runs/{key}/eval-case",
        body=raw,
        headers=decide_headers(server, **headers),
    )
    return status, payload


def saved(folder: Path) -> list[dict[str, Any]]:
    path = folder / evalcases.FILE_NAME
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_saves_a_run_and_a_regression_case(runs: Path, tmp_path: Path) -> None:
    home = tmp_path / "chats"
    with served(runs, cases_dir=home) as server:
        status, body = post(server, "good", {"tags": ["checkout"]})
        assert status == 201 and body["file"] == "eval_cases.jsonl"
        assert body["case"]["input"] == "Where is 1042?"
        assert body["case"]["expected"] == "It shipped."
        assert body["case"]["tags"] == ["checkout", "vip", "status:ok", "tool:order_status"]
        assert body["case"]["source_run"] == "good"
        status, body = post(server, "bad", {"regression": True})
        assert status == 201 and body["case"]["id"] == "regression-bad"
        assert body["case"]["expected"] is None and body["case"]["tags"][0] == "regression"
        status, body = post(server, "good", {"id": "mine", "expected": {"ok": True}})
        assert status == 201 and body["case"]["expected"] == {"ok": True}
        assert post(server, "good", {})[0] == 409  # run-good exists: ids stay unique
    assert [c["id"] for c in saved(home)] == ["run-good", "regression-bad", "mine"]
    assert [c.id for c in load_cases(home / evalcases.FILE_NAME)] == [
        "run-good",
        "regression-bad",
        "mine",
    ]


def test_refuses_runs_without_content_and_bad_requests(runs: Path, tmp_path: Path) -> None:
    home = tmp_path / "chats"
    with served(runs, cases_dir=home) as server:
        status, body = post(server, "quiet", {})
        assert status == 422 and "capture_content=True" in body["error"]
        assert post(server, "missing", {})[0] == 404
        assert post(server, "good", {"unknown": 1})[0] == 400
        assert post(server, "good", b"not json")[0] == 400
        assert post(server, "good", b"[]")[0] == 400
        assert post(server, "good", {"expected_from": "all"})[0] == 400
        assert post(server, "good", {"tags": "vip"})[0] == 400
        assert post(server, "good", {"id": "x" * 201})[0] == 400
        assert post(server, "good", {}, Origin="http://evil.example")[0] == 403
        assert post(server, "good", {}, **{"X-AA-CSRF": "wrong"})[0] == 403
        assert post(server, "good", {}, **{"Content-Type": "text/plain"})[0] == 403
        status, _, _ = request(server, "POST", "/api/runs/good/eval-case", body=b"{}", auth=False)
        assert status == 401
    assert not (home / evalcases.FILE_NAME).exists()


def test_a_full_file_or_a_failed_write_is_refused(runs: Path, tmp_path: Path) -> None:
    home = tmp_path / "chats"
    with served(runs, cases_dir=home) as server:
        with mock.patch.object(evalcases, "MAX_FILE", 10):
            status, body = post(server, "good", {})
        assert status == 413 and "full" in body["error"]
        with mock.patch("pathlib.Path.open", side_effect=OSError("disk")):
            status, body = post(server, "good", {})
        assert status == 500 and body["error"] == "the case was not saved"
        with mock.patch.object(server.folder, "refresh", side_effect=OSError("gone")):
            status, body = post(server, "good", {})
        assert status == 500


def test_with_chat_on_the_chat_file_is_shared(runs: Path) -> None:
    chat = mock.Mock()
    chat.evals = evalcases.EvalCases(runs.parent / "chat")
    with served(runs, chat=chat) as server:
        assert server.cases is chat.evals
        assert post(server, "good", {"expected_from": "none"})[0] == 201
    assert saved(runs.parent / "chat")[0]["expected"] is None


def test_an_empty_body_is_refused(runs: Path, tmp_path: Path) -> None:
    with served(runs, cases_dir=tmp_path / "chats") as server:
        assert post(server, "good", b"")[0] == 400


def test_the_case_route_takes_up_to_256_kb(runs: Path, tmp_path: Path) -> None:
    from agent_adapter.viewer._case_routes import MAX_BODY

    def body(size: int) -> bytes:
        start = b'{"id": "big", "expected": "'
        return start + b"x" * (size - len(start) - 2) + b'"}'

    with served(runs, cases_dir=tmp_path / "chats") as server:
        assert post(server, "good", body(MAX_BODY + 1))[0] == 400
        status, saved_case = post(server, "good", body(MAX_BODY))
        assert status == 201 and len(saved_case["case"]["expected"]) > 200_000
        tags = request(
            server,
            "POST",
            "/api/runs/good/tags",
            body=b'{"add": ["' + b"x" * 5_000 + b'"]}',
            headers=decide_headers(server),
        )
        assert tags[0] == 400  # other routes keep the small limit


def test_deep_json_and_unexpected_errors_get_json_replies(runs: Path, tmp_path: Path) -> None:
    with served(runs, cases_dir=tmp_path / "chats") as server:
        deep = b'{"tags": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"
        assert post(server, "good", deep)[0] == 400
        with mock.patch(
            "agent_adapter.viewer._case_routes.case_from_run", side_effect=KeyError("x")
        ):
            status, body = post(server, "good", {})
        assert status == 500 and body == {"error": "the case was not saved"}


def test_a_second_writer_and_an_unreadable_file(runs: Path, tmp_path: Path) -> None:
    home = tmp_path / "chats"
    with served(runs, cases_dir=home) as server:
        home.mkdir(parents=True, exist_ok=True)
        (home / evalcases.FILE_NAME).write_text('{"id": "run-good", "input": "x"}\n', "utf-8")
        assert post(server, "good", {})[0] == 409  # the other writer's id is seen
        assert post(server, "good", {"id": "t", "tags": ["x" * 101]})[0] == 400
        with mock.patch("pathlib.Path.read_text", side_effect=OSError("locked")):
            (home / evalcases.FILE_NAME).write_text("{}\n", "utf-8")  # changed: read again
            status, body = post(server, "good", {"id": "new"})
        assert status == 500 and "cannot be read" in body["error"]
