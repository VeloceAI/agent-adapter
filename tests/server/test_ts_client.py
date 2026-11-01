"""The `ts` session harness: a live server on an ephemeral port for `node --test` (offline).

Starts uvicorn with `tests/server/ts_app.py`, then runs `node --test "test/**/*.test.ts"` in
`clients/typescript` (a glob: Node does not search a directory argument) with `AA_BASE_URL`,
the tokens in `ts_app.ENV` and the strict scenario tools in `ts_app.LIVE_ENV`. Skips
without Node >= 22.18 or the client's tests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

import uvicorn

from tests.server.ts_app import ENV, LIVE_ENV, build_app

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "clients" / "typescript"
TEST_GLOB = "test/**/*.test.ts"
NODE_MIN = (22, 18)
STARTUP_S = 10.0
NODE_TIMEOUT_S = 300


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    """The live server's URL; the server stops when the module's tests end."""
    config = uvicorn.Config(
        build_app(),
        host="127.0.0.1",
        port=0,
        log_level="warning",
        ws="none",
        lifespan="on",
        timeout_graceful_shutdown=1,  # open SSE streams never hold the shutdown
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_S
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            pytest.fail("uvicorn did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=STARTUP_S)


def request(url: str, *, token: str = "", body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    sent = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(sent, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_live_server_answers_over_a_socket(base_url: str):
    token = ENV["AA_TOKEN"]
    assert request(f"{base_url}/healthz")[0] == 200
    assert request(f"{base_url}/tools")[0] == 401
    status, body = request(
        f"{base_url}/tools/add_numbers/invoke", token=token, body={"arguments": {"a": 2, "b": 3}}
    )
    assert (status, body["result"]) == (200, 5)
    status, parked = request(
        f"{base_url}/tools/send_email/invoke",
        token=token,
        body={"arguments": {"to": "a@example.com", "body": "hi"}},
    )
    assert status == 202
    approval = f"{base_url}/approvals/{parked['approval_id']}"
    assert request(approval, token=ENV["AA_OTHER_TOKEN"])[0] == 404
    status, done = request(
        f"{approval}/decide", token=ENV["AA_APPROVER_TOKEN"], body={"approved": True}
    )
    assert (status, done["result"]["sent"]) == (200, True)


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not on PATH")
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True)
    found = re.match(r"v(\d+)\.(\d+)", version.stdout.strip())
    if found is None or (int(found[1]), int(found[2])) < NODE_MIN:
        pytest.skip(f"Node >= {NODE_MIN[0]}.{NODE_MIN[1]} needed, found {version.stdout.strip()}")
    return node


def test_typescript_client_against_the_live_server(base_url: str):
    node = _node()
    if not any(CLIENT.glob(TEST_GLOB)):
        pytest.skip("clients/typescript has no tests yet (WP19)")
    env = {**os.environ, **ENV, **LIVE_ENV, "AA_BASE_URL": base_url}
    result = subprocess.run(
        [node, "--test", "--test-reporter=tap", TEST_GLOB],
        cwd=CLIENT,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=NODE_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # every live scenario ran against this server (strict: none skipped)
    assert re.search(r"^# skipped 0$", result.stdout, re.M), result.stdout[-2000:]
    assert re.search(r"^ +ok \d+ - approval round trip with events$", result.stdout, re.M)
