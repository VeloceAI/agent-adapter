"""Shared builders for chat tests: a served viewer with chat on, JSON posts and SSE reads."""

from __future__ import annotations

import dataclasses
import http.client
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_adapter.core.registry import Registry
from agent_adapter.viewer.chat.host import ChatHost
from agent_adapter.viewer.server import ViewerServer
from agent_adapter.viewer.serving import make_chat_server
from tests.viewer.helpers import decide_headers, request

tools = Registry()
sent: list[str] = []


@tools.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tools.tool(confirm=True)
def send(to: str, api_key: str = "") -> str:
    """Send a message (needs approval)."""
    sent.append(to)
    return f"sent to {to}"


@contextmanager
def chat_served(tmp_path: Path, **options: Any) -> Iterator[ViewerServer]:
    """A viewer with chat on, on a free port, serving in a thread; heartbeat every 0.05 s."""
    options.setdefault("chat_dir", tmp_path / "chats")
    options.setdefault("folder", tmp_path / "runs")
    server = make_chat_server(port=0, **options)
    host = chat(server)
    host.options = dataclasses.replace(host.options, heartbeat_s=0.05)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def chat(server: ViewerServer) -> ChatHost:
    assert server.chat is not None
    return server.chat


def post(server: ViewerServer, path: str, body: Any = None, **headers: str) -> tuple[int, Any]:
    """POST JSON as the page does; `headers` override (an empty value drops one)."""
    raw = b"" if body is None else json.dumps(body).encode()
    sent_headers = decide_headers(server, **headers)
    status, data, _ = request(server, "POST", path, body=raw, headers=sent_headers)
    return status, data


def get(server: ViewerServer, path: str) -> tuple[int, Any]:
    status, data, _ = request(server, "GET", path)
    return status, data


def new_session(server: ViewerServer, **body: Any) -> str:
    status, data = post(server, "/api/chat/sessions", body)
    assert status == 201, data
    return str(data["session"]["id"])


def send_turn(server: ViewerServer, session_id: str, text: str, **body: Any) -> dict[str, Any]:
    status, data = post(server, f"/api/chat/sessions/{session_id}/turns", {"text": text, **body})
    assert status == 201, data
    return dict(data)


def read_stream(
    server: ViewerServer, turn_id: str, since: int = 0, *, raw: bool = False
) -> list[dict[str, Any]] | str:
    """Every SSE event of a turn until the stream ends (or the raw text with `raw`)."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=15)
    try:
        connection.putrequest(
            "GET", f"/api/chat/turns/{turn_id}/stream?since={since}", skip_host=True
        )
        connection.putheader("Host", f"127.0.0.1:{server.port}")
        connection.putheader("X-AA-Token", server.token)
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 200, response.read()
        assert response.getheader("Content-Type", "").startswith("text/event-stream")
        text = response.read().decode("utf-8")
    finally:
        connection.close()
    return text if raw else parse(text)


def parse(text: str) -> list[dict[str, Any]]:
    """The data objects of SSE frames, checking that `id` and `event` match them."""
    events = []
    for block in text.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        fields = dict(line.split(": ", 1) for line in lines)
        if not fields:
            continue
        data = json.loads(fields["data"])
        assert fields["id"] == str(data["seq"]) and fields["event"] == data["type"]
        events.append(data)
    return events


def wait_for(check: Any, timeout: float = 10.0) -> Any:
    """Poll `check()` until it returns something truthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def pending_approval(server: ViewerServer, turn_id: str) -> dict[str, Any]:
    """The first pending approval event the turn has sent."""
    live = chat(server).live(turn_id)

    def find() -> dict[str, Any] | None:
        return next(
            (e for e in list(live.events) if e["type"] == "approval" and e["status"] == "pending"),
            None,
        )

    return dict(wait_for(find))
