"""Shared builders for viewer tests: events, run files and a served viewer."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_adapter.core.types import Event
from agent_adapter.viewer._files import RunFile, list_segments
from agent_adapter.viewer.recorder import flush_all
from agent_adapter.viewer.runs import RunFolder
from agent_adapter.viewer.server import ViewerServer, make_server


def event(type: str, ts: float, call_id: str = "", run_id: str = "run_1", **data: Any) -> Event:
    return Event(type, run_id=run_id, call_id=call_id, ts=ts, data=data)


def as_dict(item: Event, parent_id: str = "") -> dict[str, Any]:
    found = item.to_dict()
    found["parent_id"] = parent_id or found["parent_id"]
    return found


def lines(path: Path) -> list[dict[str, Any]]:
    """The parsed lines of a run file, once every recorder has written its queue."""
    assert flush_all()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def segments(folder: Path) -> list[RunFile]:
    """The run files in `folder`, once every recorder has written its queue."""
    assert flush_all()
    return list_segments(folder)


def refreshed(target: Path) -> RunFolder:
    """A RunFolder read after every recorder has written its queue."""
    assert flush_all()
    folder = RunFolder(target)
    folder.refresh()
    return folder


@contextmanager
def served(target: Path, **options: Any) -> Iterator[ViewerServer]:
    """A viewer on a free loopback port, serving in a thread."""
    server = make_server(target, port=0, **options)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(
    server: ViewerServer,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    auth: bool = True,
) -> tuple[int, Any, dict[str, str]]:
    """Status, decoded body (JSON when it is JSON) and headers; `auth` sends the access token."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    sent = {"Host": f"127.0.0.1:{server.port}"}
    if auth:
        sent["X-AA-Token"] = server.token
    sent.update(headers or {})
    try:
        connection.putrequest(method, path, skip_host=True)
        for name, value in sent.items():
            connection.putheader(name, value)
        if body is not None:
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        raw = response.read()
        found = {name.lower(): value for name, value in response.getheaders()}
    finally:
        connection.close()
    if found.get("content-type", "").startswith("application/json"):
        return response.status, json.loads(raw), found
    return response.status, raw.decode("utf-8"), found


def decide_headers(server: ViewerServer, **changes: str) -> dict[str, str]:
    """The headers a real page sends with a decision."""
    headers = {
        "Origin": f"http://127.0.0.1:{server.port}",
        "X-AA-CSRF": server.csrf,
        "Content-Type": "application/json",
    }
    headers.update(changes)
    return {name: value for name, value in headers.items() if value}
