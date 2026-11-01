"""`agui_endpoint` on Starlette and FastAPI: auth, body checks, and an SSE stream."""

import asyncio
import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("ag_ui")
pytest.importorskip("starlette")
from dataclasses import dataclass
from json import dumps
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.routing import Route

from agent_adapter.integrations.ag_ui import agui_endpoint, anonymous
from tests.integrations.ag_ui.support import Shop, refund_agent, run_input

pytestmark = pytest.mark.contract

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "ag_ui_starlette.py"
AUTH = {"Authorization": "Bearer s3cret-token"}
JSON = {**AUTH, "Content-Type": "application/json"}


@dataclass
class _Response:
    status_code: int
    headers: dict[str, str]
    text: str


async def _post(app, base_url="http://test", *, json=None, content=None, headers=None):
    """POST /agui through the ASGI interface directly (no HTTP client needed)."""
    sent = dict(headers or {})
    if json is not None:
        content = dumps(json)
        sent.setdefault("Content-Type", "application/json")
    body = (content or "").encode() if isinstance(content, str | None) else content
    url = urlsplit(base_url)
    sent.setdefault("Host", url.netloc)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": url.scheme,
        "path": "/agui",
        "raw_path": b"/agui",
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in sent.items()],
        "client": ("127.0.0.1", 5000),
        "server": (url.hostname or "test", url.port or 80),
    }
    chunks = [body[i : i + 65536] for i in range(0, len(body), 65536)] or [b""]
    messages = [
        {"type": "http.request", "body": c, "more_body": n < len(chunks) - 1}
        for n, c in enumerate(chunks)
    ]
    out: list[dict] = []
    done = asyncio.Event()

    async def receive():
        if messages:
            return messages.pop(0)
        await done.wait()  # the client stays connected until the response ends
        return {"type": "http.disconnect"}

    async def send(message):
        out.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body"):
            done.set()

    await app(scope, receive, send)
    start = next(m for m in out if m["type"] == "http.response.start")
    text = b"".join(m.get("body", b"") for m in out if m["type"] == "http.response.body")
    found = {k.decode(): v.decode() for k, v in start["headers"]}
    return _Response(start["status"], found, text.decode())


def _frames(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE), run_name="ag_ui_starlette_example")


@pytest.mark.parametrize("factory", ["starlette_app", "fastapi_app"])
async def test_example_streams_a_run(example, factory):
    if factory == "fastapi_app":
        pytest.importorskip("fastapi")
    response = await _post(example[factory](), json=run_input("r"), headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _frames(response.text)
    assert frames[0]["type"] == "RUN_STARTED"
    assert frames[-1] == {"type": "RUN_FINISHED", "threadId": "t-1", "runId": "r", "result": 120.0}


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({"Content-Type": "application/json"}, json.dumps(run_input("r")), 401),
        ({**JSON, "Authorization": "Bearer wrong"}, json.dumps(run_input("r")), 401),
        ({**AUTH, "Content-Type": "text/plain"}, json.dumps(run_input("r")), 415),
        (JSON, "{}", 422),
        (JSON, "x" * 2_000_000, 413),
    ],
)
async def test_example_refuses_bad_requests(example, headers, body, status):
    response = await _post(example["starlette_app"](), content=body, headers=headers)
    assert response.status_code == status
    assert "data:" not in response.text


async def test_interrupt_round_trip_over_http():
    shop = Shop()
    endpoint = agui_endpoint(shop.bridge(), refund_agent, authenticate=anonymous)
    app = Starlette(routes=[Route("/agui", endpoint, methods=["POST"])])
    frames = _frames((await _post(app, "http://localhost:8000", json=run_input("r-1"))).text)
    outcome = frames[-1]["outcome"]
    assert outcome["type"] == "interrupt"
    (interrupt,) = outcome["interrupts"]
    assert interrupt["toolCallId"] == "c-refund" and interrupt["reason"] == "tool_call"
    assert interrupt["expiresAt"].endswith("Z") and shop.refunds == []


def _anonymous_app(**options):
    endpoint = agui_endpoint(Shop().bridge(), refund_agent, authenticate=anonymous, **options)
    return Starlette(routes=[Route("/agui", endpoint, methods=["POST"])])


@pytest.mark.parametrize("host", ["http://evil.example", "http://test", "http://127.0.0.2"])
async def test_an_anonymous_endpoint_refuses_foreign_hosts(host):
    response = await _post(_anonymous_app(), host, json=run_input("r"))
    assert response.status_code == 421
    assert "data:" not in response.text


@pytest.mark.parametrize("host", ["http://localhost", "http://127.0.0.1:9", "http://[::1]:8000"])
async def test_an_anonymous_endpoint_serves_local_hosts(host):
    response = await _post(_anonymous_app(), host, json=run_input("r"))
    assert response.status_code == 200


async def test_allowed_hosts_replace_the_defaults():
    app = _anonymous_app(allowed_hosts=["Agents.Example.com"])
    assert (await _post(app, "https://agents.example.com", json=run_input("r"))).status_code == 200
    assert (await _post(app, "http://localhost", json=run_input("r"))).status_code == 421
    anyone = _anonymous_app(allowed_hosts=["*"])
    assert (await _post(anyone, "http://evil.example", json=run_input("r"))).status_code == 200


async def test_an_authenticated_endpoint_checks_hosts_only_when_given(example):
    response = await _post(
        example["starlette_app"](), "http://evil.example", json=run_input("r"), headers=AUTH
    )
    assert response.status_code == 200


@pytest.mark.parametrize("hosts", [[], "localhost", [""]])
def test_bad_allowed_hosts_are_refused(hosts):
    from agent_adapter.core.errors import ConfigError

    with pytest.raises(ConfigError, match="allowed_hosts"):
        agui_endpoint(Shop().bridge(), refund_agent, authenticate=anonymous, allowed_hosts=hosts)
