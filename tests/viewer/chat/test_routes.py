"""Chat endpoints: the guards every request passes, sessions, state, caps and 409s."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from agent_adapter.viewer.chat import ChatMessage, ChatSession
from agent_adapter.viewer.chat import host as host_module
from tests.viewer.chat.helpers import (
    chat,
    chat_served,
    get,
    new_session,
    post,
    read_stream,
    send_turn,
    wait_for,
)
from tests.viewer.helpers import decide_headers, request, served

release = threading.Event()


async def echo(message: ChatMessage, session: ChatSession) -> str:
    return f"echo {message.text}"


async def slow(message: ChatMessage, session: ChatSession) -> str:
    while not release.is_set():
        await asyncio.sleep(0.01)
    return "late"


@pytest.fixture(autouse=True)
def reset_release() -> None:
    release.clear()


def test_chat_is_off_without_an_agent(tmp_path: Path) -> None:
    with served(tmp_path) as server:
        assert get(server, "/api/chat") == (
            200,
            {"enabled": False, "mode": "", "agent_name": ""},
        )
        assert get(server, "/api/chat/sessions")[0] == 404
        assert post(server, "/api/chat/sessions", {})[0] == 404
        page = request(server, "GET", "/")[1]
        assert '<meta name="aa-chat" content="">' in page


def test_the_page_names_the_mode_and_allows_blob_images(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo, agent_name="demo") as server:
        status, page, headers = request(server, "GET", "/", auth=False)
        assert status == 200 and '<meta name="aa-chat" content="agent">' in page
        assert "img-src 'self' data: blob:;" in headers["content-security-policy"]
        assert get(server, "/api/chat")[1] == {
            "enabled": True,
            "mode": "agent",
            "agent_name": "demo",
        }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/chat"),
        ("GET", "/api/chat/sessions"),
        ("POST", "/api/chat/sessions"),
        ("GET", "/api/chat/turns/turn_0000000000000000/stream"),
        ("POST", "/api/chat/files?session=ses_0000000000000000"),
    ],
)
def test_every_chat_route_needs_the_token_and_a_known_host(
    tmp_path: Path, method: str, path: str
) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        headers = decide_headers(server)
        body = b"{}" if method == "POST" else None
        assert request(server, method, path, body=body, headers=headers, auth=False)[0] == 401
        wrong_host = {**headers, "Host": "evil.example:80"}
        assert request(server, method, path, body=body, headers=wrong_host)[0] == 403


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"Origin": "http://evil.example"}, "Origin"),
        ({"Origin": ""}, "Origin"),
        ({"X-AA-CSRF": "guess"}, "CSRF"),
        ({"X-AA-CSRF": ""}, "CSRF"),
        ({"Content-Type": "text/plain"}, "application/json"),
    ],
)
def test_posts_need_origin_csrf_and_json(tmp_path: Path, change: dict, why: str) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        status, body = post(server, "/api/chat/sessions", {}, **change)
        assert status == 403 and why in body["error"]
        assert get(server, "/api/chat/sessions")[1] == {"sessions": []}


def test_bodies_are_capped_and_must_be_json_objects(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        headers = decide_headers(server)
        big = json.dumps({"title": "x" * 70_000}).encode()
        assert request(server, "POST", "/api/chat/sessions", body=big, headers=headers)[0] == 413
        for raw in (b"not json", b"[1, 2]"):
            status = request(server, "POST", "/api/chat/sessions", body=raw, headers=headers)[0]
            assert status == 400
        assert post(server, "/api/chat/sessions", {"title": 3})[0] == 400
        assert post(server, "/api/chat/sessions", {"state": [1]})[0] == 400


def test_sessions_are_created_listed_and_detailed(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        sid = new_session(server, title="trip", state={"city": "Lisbon", "api_key": "k-1"})
        listed = get(server, "/api/chat/sessions")[1]["sessions"]
        assert [(s["id"], s["title"], s["turns"]) for s in listed] == [(sid, "trip", 0)]
        detail = get(server, f"/api/chat/sessions/{sid}")[1]
        assert detail["state"] == {"city": "Lisbon", "api_key": "***"}
        assert detail["turns"] == [] and detail["branches"] == [] and detail["files"] == []
        assert get(server, "/api/chat/sessions/ses_0000000000000000")[0] == 404
        assert get(server, "/api/chat/sessions/../../etc")[0] == 404
        assert get(server, "/api/chat/nothing")[0] == 404


@pytest.mark.parametrize(
    ("body", "status", "why"),
    [
        ({"text": 5}, 400, "text"),
        ({"text": ""}, 400, "send text"),
        ({"text": "x" * 33_000}, 413, "bytes"),
        ({"text": "x", "attachment_ids": "fil_1"}, 400, "attachment_ids"),
        ({"text": "x", "attachment_ids": ["fil_0000000000000000"]}, 404, "no file"),
        ({"text": "x", "edit_of": "not-a-turn"}, 400, "edit_of"),
        ({"text": "x", "edit_of": "turn_0000000000000000"}, 404, "no such turn"),
    ],
)
def test_bad_turns_are_refused(tmp_path: Path, body: dict, status: int, why: str) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        sid = new_session(server)
        found = post(server, f"/api/chat/sessions/{sid}/turns", body)
        assert found[0] == status and why in found[1]["error"]


def test_turns_on_an_unknown_session_are_404(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        path = "/api/chat/sessions/ses_0000000000000000/turns"
        assert post(server, path, {"text": "x"})[0] == 404


def test_one_running_turn_per_session(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=slow) as server:
        sid = new_session(server)
        first = send_turn(server, sid, "a")
        status, body = post(server, f"/api/chat/sessions/{sid}/turns", {"text": "b"})
        assert status == 409 and "already running" in body["error"]
        state = post(server, f"/api/chat/sessions/{sid}/state", {"state": {}})
        assert state[0] == 409
        assert post(server, f"/api/chat/turns/{first['turn_id']}/replay", {})[0] == 409
        listed = get(server, "/api/chat/sessions")[1]["sessions"]
        assert listed[0]["running"] == first["turn_id"]
        release.set()
        assert read_stream(server, first["turn_id"])[-1]["type"] == "done"
        assert send_turn(server, sid, "b")["session_id"] == sid


def test_running_turns_are_capped_over_all_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_module, "MAX_RUNNING", 1)
    with chat_served(tmp_path, agent=slow) as server:
        first = send_turn(server, new_session(server), "a")
        status, body = post(
            server, f"/api/chat/sessions/{new_session(server)}/turns", {"text": "b"}
        )
        assert status == 409 and "at most 1" in body["error"]
        release.set()
        read_stream(server, first["turn_id"])


def test_state_is_replaced_between_turns_and_masked_values_keep_their_secret(
    tmp_path: Path,
) -> None:
    seen: list[dict] = []

    async def remember(message: ChatMessage, session: ChatSession) -> str:
        seen.append(dict(session.state))
        return "ok"

    with chat_served(tmp_path, agent=remember) as server:
        sid = new_session(server, state={"api_key": "k-1", "n": 1})
        path = f"/api/chat/sessions/{sid}/state"
        status, body = post(server, path, {"state": {"api_key": "***", "n": 2}})
        assert status == 200 and body == {"state": {"api_key": "***", "n": 2}}
        assert post(server, path, {"state": [1]})[0] == 400
        assert (
            post(server, "/api/chat/sessions/ses_0000000000000000/state", {"state": {}})[0] == 404
        )
        read_stream(server, send_turn(server, sid, "go")["turn_id"])
    assert seen == [{"api_key": "k-1", "n": 2}]


def test_edit_forks_a_branch_with_the_turns_before_it(tmp_path: Path) -> None:
    histories: list[int] = []

    async def counting(message: ChatMessage, session: ChatSession) -> str:
        histories.append(len(session.history))
        return message.text.upper()

    with chat_served(tmp_path, agent=counting) as server:
        sid = new_session(server, title="main")
        first = send_turn(server, sid, "a")
        read_stream(server, first["turn_id"])
        second = send_turn(server, sid, "b")
        read_stream(server, second["turn_id"])
        edited = send_turn(server, sid, "B2", edit_of=second["turn_id"])
        assert edited["session_id"] != sid
        events = read_stream(server, edited["turn_id"])
        branch = get(server, f"/api/chat/sessions/{edited['session_id']}")[1]
        parent = get(server, f"/api/chat/sessions/{sid}")[1]
    assert events[-1]["turn"]["edit_of"] == second["turn_id"]
    assert histories == [0, 2, 2]
    assert [t["text"] for t in branch["turns"]] == ["a", "B2"]
    assert branch["session"]["parent"] == {"session_id": sid, "turn_id": second["turn_id"]}
    assert branch["session"]["title"] == "main (edit)"
    assert parent["branches"] == [
        {"id": edited["session_id"], "title": "main (edit)", "turn_id": second["turn_id"]}
    ]


def test_replay_runs_the_same_input_in_a_new_branch(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        sid = new_session(server)
        first = send_turn(server, sid, "again")
        read_stream(server, first["turn_id"])
        status, replayed = post(server, f"/api/chat/turns/{first['turn_id']}/replay", {})
        assert status == 201 and replayed["session_id"] != sid
        events = read_stream(server, replayed["turn_id"])
        assert post(server, "/api/chat/turns/turn_0000000000000000/replay", {})[0] == 404
    assert events[-1]["turn"]["replay_of"] == first["turn_id"]
    assert events[-1]["turn"]["answer"] == "echo again"


def test_stop_of_an_unknown_or_ended_turn(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        assert post(server, "/api/chat/turns/turn_0000000000000000/stop", {})[0] == 404
        turn = send_turn(server, new_session(server), "x")
        read_stream(server, turn["turn_id"])
        assert post(server, f"/api/chat/turns/{turn['turn_id']}/stop", {})[0] == 409


def test_a_closed_host_refuses_new_turns(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=slow) as server:
        sid = new_session(server)
        turn = send_turn(server, sid, "a")
        host = chat(server)
        threading.Thread(target=host.close, daemon=True).start()
        wait_for(lambda: host.closed)
        status = post(server, f"/api/chat/sessions/{sid}/turns", {"text": "b"})[0]
        assert status in (409, 503)
        events = read_stream(server, turn["turn_id"])
    assert events[-1]["status"] == "stopped"
