"""The JS recipe pages, live: each framework's own MCP client against our public app.

Runs `examples/connect/js/check.ts` with Node against `public_app` on a free port. Skips
without Node or without `npm ci` in `examples/connect/js` (the check needs the registry).
"""

from __future__ import annotations

import os
import runpy
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "examples" / "connect" / "js"
AGENT, REVIEWER = "agent-test-token", "reviewer-test-token"
TOOLS = "send_invoice,shipping_rate"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (JS / "node_modules").is_dir(),
    reason="needs Node and `npm ci` in examples/connect/js",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    pytest.importorskip("mcp")
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port()
    host = f"127.0.0.1:{port}"
    example = runpy.run_path(str(ROOT / "examples" / "connect_server.py"))
    app = example["public_app"](example["registry"], host, AGENT, REVIEWER)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="none")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "uvicorn did not start"
        time.sleep(0.01)
    yield f"http://{host}"
    server.should_exit = True
    thread.join(5)


def test_every_js_framework_lists_calls_and_parks(base_url: str) -> None:
    env = {**os.environ, "AA_URL": base_url, "AA_AGENT_TOKEN": AGENT}
    run = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "check.ts"],
        cwd=JS,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.splitlines() == [
        f'vercel: {TOOLS} | "18.0" | parked=true',
        f'mastra: shopTools_{TOOLS.replace(",", ",shopTools_")} | {{"result":18}} | parked=true',
        f'openai-agents: {TOOLS} | "18.0" | parked=true',
        f'genkit: shopTools/{TOOLS.replace(",", ",shopTools/")} | {{"result":18}} | parked=true',
    ]
