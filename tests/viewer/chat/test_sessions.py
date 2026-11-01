"""Sessions survive a restart; bad files are skipped; eval cases load with `load_cases`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agent_adapter.optimiser.cases import load_cases
from agent_adapter.viewer.chat import sessions as sessions_module
from agent_adapter.viewer.chat.evalcases import EvalCases
from agent_adapter.viewer.chat.records import Refused
from agent_adapter.viewer.chat.sessions import SessionStore
from tests.viewer.chat.helpers import chat_served, get, new_session, post, read_stream, send_turn
from tests.viewer.chat.test_contract import async_text


def test_sessions_and_turns_survive_a_restart(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server, title="kept", state={"n": 1})
        turn = send_turn(server, sid, "hi")
        read_stream(server, turn["turn_id"])
    with chat_served(tmp_path, agent=async_text) as server:
        listed = get(server, "/api/chat/sessions")[1]["sessions"]
        detail = get(server, f"/api/chat/sessions/{sid}")[1]
        events = read_stream(server, turn["turn_id"])
        later = read_stream(server, turn["turn_id"], since=1)
    assert [(s["id"], s["title"], s["turns"]) for s in listed] == [(sid, "kept", 1)]
    assert detail["state"] == {"n": 1} and detail["turns"][0]["answer"] == "echo hi"
    assert [e["type"] for e in events] == ["text", "done"] and events[0]["text"] == "echo hi"
    assert [e["type"] for e in later] == ["done"]


def test_a_turn_cut_off_by_a_restart_is_stored_as_stopped(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create("t")
    data = session.to_dict()
    data["turns"] = [{"id": "turn_1", "session_id": session.id, "run_id": "run_1", "text": "x"}]
    (tmp_path / f"{session.id}.json").write_text(json.dumps(data), encoding="utf-8")
    again = SessionStore(tmp_path).get(session.id)
    assert again.turns[0].status == "stopped" and again.turns[0].error["kind"] == "interrupted"
    saved = json.loads((tmp_path / f"{session.id}.json").read_text(encoding="utf-8"))
    assert saved["turns"][0]["status"] == "stopped"


def test_unreadable_or_foreign_session_files_are_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "ses_0000000000000001.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "ses_0000000000000002.json").write_text(
        json.dumps({"id": "ses_0000000000000003"}), encoding="utf-8"
    )
    (tmp_path / "ses_0000000000000004.json").write_text(
        json.dumps(
            {
                "id": "ses_0000000000000004",
                "turns": [
                    {"id": "t", "session_id": "s", "run_id": "r", "text": "x", "status": "odd"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        store = SessionStore(tmp_path)
    assert store.summaries() == []
    assert sum("skipped" in r.getMessage() for r in caplog.records) == 3


def test_a_failed_save_is_logged_and_the_session_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = SessionStore(tmp_path)

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(sessions_module.os, "replace", broken)
    with caplog.at_level(logging.ERROR):
        session = store.create("x")
    assert store.get(session.id) is session
    assert "saving session" in caplog.text
    assert not list(tmp_path.glob("*.tmp"))


def test_a_chat_folder_that_cannot_be_made_is_a_config_error(tmp_path: Path) -> None:
    from agent_adapter.core.errors import ConfigError

    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ConfigError, match="cannot create the chat folder"):
        SessionStore(blocker / "chats")


def test_session_and_turn_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sessions_module, "MAX_SESSIONS", 1)
    monkeypatch.setattr(sessions_module, "MAX_TURNS", 0)
    store = SessionStore(tmp_path)
    session = store.create()
    with pytest.raises(Refused) as full:
        store.create()
    assert full.value.status == 409
    with pytest.raises(Refused) as turns:
        store.check_turn_room(session)
    assert turns.value.status == 409


def test_a_saved_eval_case_loads_with_load_cases(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        turn = send_turn(server, new_session(server), "what is up?")
        read_stream(server, turn["turn_id"])
        path = f"/api/chat/turns/{turn['turn_id']}/eval-case"
        status, body = post(server, path, {"id": "c1", "expected": "echo", "tags": ["smoke"]})
        assert status == 201 and body["file"] == "eval_cases.jsonl"
        assert post(server, path, {"id": "c1"}) == (409, {"error": "a case with id 'c1' exists"})
        assert post(server, path, {"id": ""})[0] == 400
        assert post(server, path, {"id": "c2", "tags": "smoke"})[0] == 400
        assert (
            post(server, "/api/chat/turns/turn_0000000000000000/eval-case", {"id": "z"})[0] == 404
        )
        assert post(server, path, {"id": "c2"})[0] == 201
    (case, second) = load_cases(tmp_path / "chats" / "eval_cases.jsonl")
    assert (case.id, case.input, case.expected, case.tags) == (
        "c1",
        "what is up?",
        "echo",
        ("smoke",),
    )
    assert second.id == "c2" and second.expected is None


def test_eval_case_ids_already_in_the_file_stay_unique(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "eval_cases.jsonl").write_text('{"id": "old", "input": "x"}\nnot json\n\n')
    from agent_adapter.viewer.chat.records import Turn

    with caplog.at_level(logging.WARNING):
        cases = EvalCases(tmp_path)
    assert "is not a case" in caplog.text
    turn = Turn(id="turn_1", session_id="ses_1", run_id="run_1", text="hi")
    with pytest.raises(Refused) as duplicate:
        cases.add(turn, "old", None, None)
    assert duplicate.value.status == 409
