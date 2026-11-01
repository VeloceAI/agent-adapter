"""Second review round: one-use launch links, turns that fail to settle, close, and names."""

from __future__ import annotations

import importlib
import io
import logging
import socket
import tempfile
from pathlib import Path

import pytest

from agent_adapter.viewer import serve
from agent_adapter.viewer import server as server_module
from agent_adapter.viewer.chat.live import LiveTurn
from agent_adapter.viewer.chat.records import Turn
from agent_adapter.viewer.chat.settle import Settler
from agent_adapter.viewer.launch import LaunchCodes
from tests.viewer.chat.helpers import (
    chat,
    chat_served,
    get,
    new_session,
    pending_approval,
    read_stream,
    send_turn,
    tools,
)
from tests.viewer.chat.test_contract import async_text
from tests.viewer.chat.test_review_fixes import uses_tools
from tests.viewer.chat.test_setup import no_serving
from tests.viewer.helpers import request, served

# --- launch links ------------------------------------------------------------------------------


def test_a_launch_code_redirects_once_to_the_tokened_page(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG), served(tmp_path) as server:
        code = server.launch.new()
        status, body, headers = request(server, "GET", f"/launch/{code}", auth=False)
        assert status == 302 and body == ""
        assert headers["location"] == f"/#token={server.token}"
        assert headers["cache-control"] == "no-store"
        assert headers["referrer-policy"] == "no-referrer"
        again = request(server, "GET", f"/launch/{code}", auth=False)
        assert again[0] == 404 and "used or expired" in again[1]["error"]
        assert request(server, "GET", "/launch/unknown-code", auth=False)[0] == 404
        wrong_host = {"Host": "evil.example:80"}
        fresh = server.launch.new()
        assert request(server, "GET", f"/launch/{fresh}", headers=wrong_host, auth=False)[0] == 403
        assert request(server, "GET", f"/launch/{fresh}", auth=False)[0] == 302
        token = server.token
    assert code not in caplog.text and fresh not in caplog.text and token not in caplog.text
    refusals = [r for r in caplog.records if "launch link" in r.getMessage()]
    assert [r.levelno for r in refusals] == [logging.WARNING, logging.WARNING]  # reuse, probe


def test_a_launch_code_expires_after_its_time() -> None:
    now = [100.0]
    codes = LaunchCodes(ttl=60.0, clock=lambda: now[0])
    late, early = codes.new(), codes.new()
    now[0] += 59.0
    assert codes.take(early)
    now[0] += 2.0
    assert not codes.take(late) and not codes.take(early)


def test_serve_opens_a_launch_link_and_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    served_servers = no_serving(monkeypatch)
    opened: list[str] = []
    monkeypatch.setattr("agent_adapter.viewer.serving.webbrowser.open", opened.append)
    serve(
        async_text, folder=tmp_path / "runs", chat_dir=tmp_path / "chats", port=0, out=io.StringIO()
    )
    (url,) = opened
    server = served_servers[0]
    assert url.startswith(f"{server.base_url}launch/") and server.token not in url
    code = url.rsplit("/", 1)[1]
    assert server.launch.take(code) and not server.launch.take(code)
    assert list(temp.iterdir()) == []


def test_the_serving_module_is_importable_by_name() -> None:
    module = importlib.import_module("agent_adapter.viewer.serving")
    assert module.serve is serve and callable(module.run_server)


# --- turns that fail to settle -----------------------------------------------------------------


def test_a_turn_whose_settle_fails_still_ends_once_and_frees_its_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def broken(self: Settler, live: LiveTurn, session: object, turn: Turn, *args: object) -> None:
        calls.append(turn.id)
        raise RecursionError("deep state")  # a RuntimeError, but not "no worker threads"

    monkeypatch.setattr(Settler, "settle", broken)
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        turn = send_turn(server, sid, "x")
        events = read_stream(server, turn["turn_id"])
        assert get(server, f"/api/chat/sessions/{sid}")[1]["session"]["running"] is None
    assert calls == [turn["turn_id"]]  # settled once, never again on the loop
    end = events[-1]
    assert (end["type"], end["status"], end["turn"]["status"]) == ("error", "failed", "failed")


def test_close_ends_a_turn_that_registered_while_it_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        host = chat(server)
        session = host.store.create("late")
        turn = Turn(id="turn_00000000000000aa", session_id=session.id, run_id="run_x", text="x")
        session.turns.append(turn)
        session.running = turn.id
        late = LiveTurn(turn.id, session.id, turn.run_id)
        original = host._runner.close

        def close_then_register(timeout: float = 5.0) -> None:
            original(timeout=timeout)
            with host._lock:
                host._live[turn.id] = late  # missed by close()'s first snapshot

        monkeypatch.setattr(host._runner, "close", close_then_register)
        host.close()
    end = late.events[-1]
    assert late.ended and (end["type"], end["status"], end["kind"]) == (
        "error",
        "stopped",
        "interrupted",
    )
    assert end["turn"]["status"] == "stopped" and end["turn"]["error"]["kind"] == "interrupted"
    assert turn.status == "stopped" and session.running == ""


def test_close_during_an_approval_reports_a_stopped_turn(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=uses_tools, registry=tools) as server:
        turn = send_turn(server, new_session(server), "go")
        pending_approval(server, turn["turn_id"])
        chat(server).close()
        events = read_stream(server, turn["turn_id"])
    end = events[-1]
    assert end["status"] == "stopped" and end["turn"]["status"] == "stopped"


# --- a client that leaves mid-reply ------------------------------------------------------------


def test_a_client_gone_mid_reply_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def gone(*args: object) -> None:
        raise BrokenPipeError("client closed")

    monkeypatch.setattr(server_module, "handle_get", gone)
    caplog.set_level(logging.DEBUG)
    with (
        chat_served(tmp_path, agent=async_text) as server,
        socket.create_connection(("127.0.0.1", server.port), timeout=5) as client,
    ):
        head = [
            "GET /api/chat/sessions HTTP/1.1",
            f"Host: 127.0.0.1:{server.port}",
            f"X-AA-Token: {server.token}",
            "",
            "",
        ]
        client.sendall("\r\n".join(head).encode())
        assert client.recv(100) == b""  # no 500 written to a gone client
    assert "client gone" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


# --- round 3: the theft signal, and the shape of a late end event ------------------------------


def test_reusing_an_issued_code_always_warns_while_probes_warn_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [0.0]
    codes = LaunchCodes(ttl=60.0, clock=lambda: now[0])
    used, unused = codes.new(), codes.new()
    assert codes.take(used)
    with caplog.at_level(logging.DEBUG, logger="agent_adapter.viewer.launch"):
        for probe in ("x1", "x2", "x3"):
            assert not codes.take(probe)
        assert not codes.take(used) and not codes.take(used)
        now[0] += 61.0
        assert not codes.take(unused)  # issued, then expired
    levels = [(r.levelno, "someone else" in r.getMessage()) for r in caplog.records]
    assert levels == [
        (logging.WARNING, False),
        (logging.DEBUG, False),
        (logging.DEBUG, False),
        (logging.WARNING, True),
        (logging.WARNING, True),
        (logging.WARNING, True),
    ]
    assert used not in caplog.text and unused not in caplog.text


def test_spent_codes_are_forgotten_after_another_ttl() -> None:
    now = [0.0]
    codes = LaunchCodes(ttl=60.0, clock=lambda: now[0])
    code = codes.new()
    assert codes.take(code)
    now[0] += 121.0
    assert not codes.take(code)
    assert codes.refusals == 1  # counted as unknown again: nothing kept forever


def test_a_late_end_event_has_the_turn_shape_even_if_the_store_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        host = chat(server)
        late = LiveTurn("turn_00000000000000bb", "ses_00000000000000bb", "run_y")
        late.text("partial")

        def lost(turn_id: str) -> None:
            raise RuntimeError("store gone")

        monkeypatch.setattr(host.store, "find_turn", lost)
        host._interrupt(late)
    end = late.events[-1]
    shape = set(Turn(id="t", session_id="s", run_id="r", text="").view())
    assert (end["type"], end["status"], end["kind"]) == ("error", "stopped", "interrupted")
    assert set(end["turn"]) == shape and end["turn"]["status"] == "stopped"
    assert end["turn"]["answer"] == "partial" and end["turn"]["id"] == late.turn_id
