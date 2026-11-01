"""SSE framing, heartbeats, `since`, and a stream the client drops."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path

import pytest

from agent_adapter.viewer.chat import ChatMessage, ChatSession
from agent_adapter.viewer.chat.live import MAX_PREVIEW, LiveTurn, preview
from agent_adapter.viewer.chat.stream import frame
from tests.viewer.chat.helpers import chat_served, get, new_session, parse, read_stream, send_turn

go = threading.Event()


async def waits(message: ChatMessage, session: ChatSession) -> str:
    while not go.is_set():
        await asyncio.sleep(0.01)
    return "line one\nline two \u2028 é"


@pytest.fixture(autouse=True)
def reset() -> None:
    go.clear()


def test_a_frame_is_one_id_event_and_data_line() -> None:
    raw = frame({"seq": 3, "type": "text", "text": "a\nb\r\n\u2028"})
    assert "\u2028".encode() not in raw
    lines = raw.decode().split("\n")
    assert lines[:2] == ["id: 3", "event: text"]
    assert lines[2].startswith("data: ") and json.loads(lines[2][6:])["text"] == "a\nb\r\n\u2028"
    assert lines[3:] == ["", ""]


def test_pings_keep_an_idle_stream_open_until_the_end_event(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=waits) as server:
        turn = send_turn(server, new_session(server), "x")
        timer = threading.Timer(0.4, go.set)
        timer.start()
        raw = read_stream(server, turn["turn_id"], raw=True)
        timer.join()
    assert isinstance(raw, str) and ": ping\n\n" in raw
    events = parse(raw)
    assert events[-1]["type"] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "text") == (
        "line one\nline two \u2028 é"
    )


def test_since_skips_events_already_seen(tmp_path: Path) -> None:
    go.set()
    with chat_served(tmp_path, agent=waits) as server:
        turn = send_turn(server, new_session(server), "x")
        events = read_stream(server, turn["turn_id"])
        tail = read_stream(server, turn["turn_id"], since=len(events) - 1)
        beyond = read_stream(server, turn["turn_id"], since=999)
    assert tail == events[-1:] and beyond == events[-1:]  # always ends with the end event


def test_a_stream_of_an_unknown_turn_is_404(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=waits) as server:
        assert get(server, "/api/chat/turns/turn_0000000000000000/stream")[0] == 404


def test_a_client_that_leaves_does_not_break_the_turn(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=waits) as server:
        turn = send_turn(server, new_session(server), "x")
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as client:
            head = [
                f"GET /api/chat/turns/{turn['turn_id']}/stream HTTP/1.1",
                f"Host: 127.0.0.1:{server.port}",
                f"X-AA-Token: {server.token}",
                "",
                "",
            ]
            client.sendall("\r\n".join(head).encode())
            assert client.recv(15).startswith(b"HTTP/1.0 200")
        go.set()
        events = read_stream(server, turn["turn_id"])
    assert events[-1]["type"] == "done"


def test_previews_are_masked_and_cut() -> None:
    assert preview({"password": "p", "n": 1}) == '{"password": "***", "n": 1}'
    long = preview("x" * (MAX_PREVIEW + 5))
    assert long.endswith("... [+5 chars]") and len(long) < MAX_PREVIEW + 20


def test_a_live_turn_keeps_nothing_after_its_end_and_caps_its_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_adapter.viewer.chat import live as live_module

    monkeypatch.setattr(live_module, "MAX_EVENTS", 2)
    monkeypatch.setattr(live_module, "MAX_ANSWER", 4)
    live = LiveTurn("turn_1", "ses_1", "run_1")
    live.text("abc")
    live.text("defg")
    live.text("more")
    live.text("")
    assert "".join(live.answer) == "abcd" and live.dropped == 0
    live.push("tool", name="x")  # past MAX_EVENTS: dropped
    live.push("done", status="done")
    live.push("text", text="late")
    assert [e["type"] for e in live.events] == ["text", "text", "done"] and live.dropped == 1
    assert live.read(0, 0.01) == (live.events, True)
