"""The backend guide examples run offline: the service layer, each framework, Celery, scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from agent_adapter import ApprovalDecision, run_sync
from agent_adapter.errors import ApprovalAlreadyDecided

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
DEV = {"Authorization": "Bearer dev-token"}
REVIEWER = {"Authorization": "Bearer reviewer-token"}
INVOICE = {"arguments": {"customer_id": "cus_1024", "amount": 49.5}}


@pytest.fixture(autouse=True)
def _examples_importable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The examples import their shared `backend_common` as a sibling module."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    yield


def load(name: str) -> ModuleType:
    """Import `examples/<name>.py` fresh, registered as `name` (Django resolves its URLconf)."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- the shared service layer (no framework needed) ---


async def test_service_layer_runs_parks_decides_and_resumes_once():
    common = load("backend_common")
    backend = common.build_backend()
    status, rate = await common.invoke(backend, "shipping_rate", {"region": "us"})
    assert (status, rate["result"]) == (200, 18.0)

    status, parked = await common.invoke(backend, "send_invoice", INVOICE["arguments"])
    assert (status, parked["status"]) == (202, "pending_approval")
    approval_id = parked["approval_id"]

    status, refused = await common.decide(
        backend, approval_id, reviewer="dev@example.com", approved=True
    )
    assert (status, refused["title"]) == (403, "Forbidden")

    status, _ = await common.decide(
        backend,
        approval_id,
        reviewer="reviewer@example.com",
        approved=True,
        edited_args={"amount": "x"},
    )
    assert status == 422  # edits are validated before the decision is stored
    assert (await backend.store.get(approval_id)).status == "pending"

    status, done = await common.decide(
        backend, approval_id, reviewer="reviewer@example.com", approved=True
    )
    assert (status, done["result"]) == (200, "invoiced cus_1024 for 49.50 USD")
    status, again = await common.resume(backend, approval_id)
    assert (status, again["title"]) == (409, "Approval already decided")
    status, _ = await common.resume(backend, "apr_missing")
    assert status == 404


async def test_service_layer_denies_and_reports_cost_and_budget():
    common = load("backend_common")
    backend = common.build_backend()
    _, parked = await common.invoke(backend, "send_invoice", INVOICE["arguments"])
    status, denied = await common.decide(
        backend, parked["approval_id"], reviewer="reviewer@example.com", approved=False
    )
    assert (status, denied["status"]) == (200, "denied")

    status, answer = await common.ask(backend, "Shipping to the US?")
    assert status == 200
    assert answer == {"status": "stop", "answer": "The shipping rate is 18.0%.", "cost_usd": 0.0048}

    tight = common.build_backend(max_usd=0.002)  # the second model call is over budget
    status, problem = await common.ask(tight, "Shipping to the US?")
    assert (status, problem["type"]) == (429, "urn:agent-adapter:error:budget-exceeded")


# --- FastAPI ---


def test_fastapi_routes_approve_ask_and_mount_the_tool_server():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = load("backend_fastapi").make_app()
    with TestClient(app, base_url="http://localhost") as client:
        rate = client.post(
            "/tools/shipping_rate", json={"arguments": {"region": "us"}}, headers=DEV
        )
        assert (rate.status_code, rate.json()["result"]) == (200, 18.0)
        assert client.post("/tools/shipping_rate", json={"arguments": {}}).status_code == 401

        parked = client.post("/tools/send_invoice", json=INVOICE, headers=DEV)
        assert parked.status_code == 202
        decide = f"/approvals/{parked.json()['approval_id']}/decide"
        assert client.post(decide, json={"approved": True}, headers=DEV).status_code == 403
        done = client.post(decide, json={"approved": True}, headers=REVIEWER)
        assert (done.status_code, done.json()["result"]) == (200, "invoiced cus_1024 for 49.50 USD")
        assert client.post(decide, json={"approved": True}, headers=REVIEWER).status_code == 409

        answer = client.post("/ask", json={"question": "Shipping to the US?"}, headers=DEV)
        assert answer.json()["cost_usd"] == 0.0048

        mounted = client.post(
            "/agent/tools/shipping_rate/invoke", json={"arguments": {"region": "eu"}}, headers=DEV
        )
        assert (mounted.status_code, mounted.json()["result"]) == (200, 5.0)
        assert client.get("/agent/healthz").json()["status"] == "ok"


def test_fastapi_budget_is_per_request():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    module = load("backend_fastapi")
    app = module.make_app(sys.modules["backend_common"].build_backend(max_usd=0.002))
    with TestClient(app, base_url="http://localhost") as client:
        over = client.post("/ask", json={"question": "Shipping?"}, headers=DEV)
        assert (over.status_code, over.json()["status"]) == (429, 429)


# --- Flask ---


@pytest.fixture
def flask_app() -> Iterator[Any]:
    """The Flask app with the tool server mounted; its WSGI lifespan is closed afterwards."""
    pytest.importorskip("flask")
    pytest.importorskip("a2wsgi")
    pytest.importorskip("starlette")
    example = load("backend_flask")
    app = example.make_app()
    yield app
    tools = app.extensions[example.TOOLS_KEY]
    tools.close()
    tools.close()  # a second close (the atexit hook's) is a no-op


def test_flask_views_run_sync_approve_and_mount_the_wsgi_tool_server(flask_app: Any) -> None:
    client = flask_app.test_client()  # Host: localhost
    rate = client.post("/tools/shipping_rate", json={"arguments": {"region": "us"}}, headers=DEV)
    assert (rate.status_code, rate.get_json()["result"]) == (200, 18.0)
    assert client.post("/tools/shipping_rate", json={}).status_code == 401
    assert client.post("/tools/shipping_rate", data="x", headers=DEV).status_code == 400

    parked = client.post("/tools/send_invoice", json=INVOICE, headers=DEV)
    assert parked.status_code == 202
    decide = f"/approvals/{parked.get_json()['approval_id']}/decide"
    done = client.post(decide, json={"approved": True}, headers=REVIEWER)
    assert (done.status_code, done.get_json()["result"]) == (200, "invoiced cus_1024 for 49.50 USD")
    answer = client.post("/ask", json={"question": "Shipping?"}, headers=DEV)
    assert answer.get_json()["cost_usd"] == 0.0048

    mounted = client.post(
        "/agent/tools/shipping_rate/invoke", json={"arguments": {"region": "eu"}}, headers=DEV
    )
    assert (mounted.status_code, mounted.get_json()["result"]) == (200, 5.0)


def test_flask_without_the_tool_server():
    pytest.importorskip("flask")

    client = load("backend_flask").make_app(mount_tools=False).test_client()
    assert client.post("/agent/tools/shipping_rate/invoke", json={}, headers=DEV).status_code == 404
    denied = client.post("/tools/send_invoice", json=INVOICE, headers=DEV)
    decide = f"/approvals/{denied.get_json()['approval_id']}/decide"
    reply = client.post(decide, json={"approved": False}, headers=REVIEWER)
    assert reply.get_json()["status"] == "denied"


# --- Django ---


@pytest.fixture(scope="module")
def django_example() -> Iterator[ModuleType]:
    """The one-file Django project, its URLconf active even if Django was configured first."""
    pytest.importorskip("django")
    from django.test.utils import override_settings

    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(EXAMPLES))
        module = load("backend_django")
    with override_settings(ROOT_URLCONF="backend_django", ALLOWED_HOSTS=["localhost"]):
        yield module


def _django_post(client: Any, path: str, body: Any, headers: dict[str, str]) -> Any:
    return client.post(path, json.dumps(body), content_type="application/json", headers=headers)


def test_django_async_and_sync_views_approve_and_ask(django_example: ModuleType):
    from django.test import Client

    client = Client(HTTP_HOST="localhost")
    rate = _django_post(client, "/tools/shipping_rate", {"arguments": {"region": "us"}}, DEV)
    assert (rate.status_code, rate.json()["result"]) == (200, 18.0)
    assert _django_post(client, "/tools/shipping_rate", {}, {}).status_code == 401
    bad = client.post("/tools/shipping_rate", "x", content_type="application/json", headers=DEV)
    assert bad.status_code == 400
    assert client.get("/tools/shipping_rate", headers=DEV).status_code == 405

    parked = _django_post(client, "/tools/send_invoice", INVOICE, DEV)
    assert parked.status_code == 202
    decide = f"/approvals/{parked.json()['approval_id']}/decide"
    done = _django_post(client, decide, {"approved": True}, REVIEWER)  # the sync view
    assert (done.status_code, done.json()["result"]) == (200, "invoiced cus_1024 for 49.50 USD")
    answer = _django_post(client, "/ask", {"question": "Shipping?"}, DEV)
    assert answer.json()["cost_usd"] == 0.0048


async def test_django_async_client_and_the_asgi_app_with_the_tool_server(
    django_example: ModuleType,
):
    from django.test import AsyncClient

    client = AsyncClient(HTTP_HOST="localhost")
    parked = await client.post(
        "/tools/send_invoice", INVOICE, content_type="application/json", headers=DEV
    )
    decide = f"/approvals/{parked.json()['approval_id']}/decide"
    denied = await client.post(
        decide, {"approved": False}, content_type="application/json", headers=REVIEWER
    )
    assert denied.json()["status"] == "denied"  # the sync view ran in Django's thread


def test_django_asgi_app_serves_django_and_the_tool_server(django_example: ModuleType):
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    with TestClient(django_example.asgi_application(), base_url="http://localhost") as client:
        mounted = client.post(
            "/agent/tools/shipping_rate/invoke", json={"arguments": {"region": "eu"}}, headers=DEV
        )
        assert (mounted.status_code, mounted.json()["result"]) == (200, 5.0)
        rate = client.post(
            "/tools/shipping_rate", json={"arguments": {"region": "us"}}, headers=DEV
        )
        assert rate.json()["result"] == 18.0


# --- Celery ---


def test_celery_task_bodies_park_and_resume_without_celery():
    tasks = load("backend_celery")
    assert tasks.run_tool("shipping_rate", {"region": "us"}, "run_a", "call_a") == {
        "status": "ok",
        "result": 18.0,
    }
    parked = tasks.run_tool("send_invoice", INVOICE["arguments"], "run_b", "call_b")
    assert parked["status"] == "pending_approval"
    again = tasks.run_tool("send_invoice", INVOICE["arguments"], "run_b", "call_b")
    assert again == parked  # a retry with the same ids is the same call: no second approval

    decision = ApprovalDecision(approved=True, decided_by="reviewer@example.com")
    run_sync(tasks.backend().store.decide(parked["approval_id"], decision))
    assert tasks.resume_tool(parked["approval_id"]) == {
        "status": "ok",
        "result": "invoiced cus_1024 for 49.50 USD",
    }
    with pytest.raises(ApprovalAlreadyDecided):  # a duplicate delivery runs nothing
        tasks.resume_tool(parked["approval_id"])


def test_celery_tasks_run_eagerly_and_resume_after_review():
    pytest.importorskip("celery")

    tasks = load("backend_celery")
    app = tasks.make_celery()
    app.conf.task_always_eager = True  # run in this process: no broker in tests
    assert tasks.enqueue(app, "shipping_rate", {"region": "eu"}).get() == {
        "status": "ok",
        "result": 5.0,
    }
    parked = tasks.enqueue(app, "send_invoice", INVOICE["arguments"]).get()
    refused, none = tasks.review(
        app, parked["approval_id"], reviewer="dev@example.com", approved=True
    )
    assert (refused[0], none) == (403, None)
    reply, queued = tasks.review(
        app, parked["approval_id"], reviewer="reviewer@example.com", approved=True
    )
    assert reply == (200, {"status": "approved", "approval_id": parked["approval_id"]})
    assert queued.get()["result"] == "invoiced cus_1024 for 49.50 USD"


# --- Scripts ---


def test_script_runs_a_batch_denies_unattended_and_sets_the_exit_code(
    capsys: pytest.CaptureFixture[str],
):
    script = load("backend_script")
    assert script.main(["eu", "us", "--invoice", "cus_1024", "49.5"]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "eu ok 5.0",
        "us ok 18.0",
        "invoice error approval-denied",
        "calls=3 cost=$0.0",
    ]


def test_script_asks_on_the_terminal_when_interactive(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    script = load("backend_script")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert script.main(["--invoice", "cus_1024", "49.5", "--interactive"]) == 0
    assert capsys.readouterr().out.splitlines()[1] == "invoice ok invoiced cus_1024 for 49.50 USD"


def test_examples_without_framework_imports_type_check_strictly():
    mypy_api = pytest.importorskip("mypy.api")
    names = ("backend_common.py", "backend_script.py", "tutorial_research.py")
    limit = sys.getrecursionlimit()
    try:
        out, err, status = mypy_api.run(
            ["--strict", "--no-error-summary", *(str(EXAMPLES / n) for n in names)]
        )
    finally:
        sys.setrecursionlimit(limit)  # mypy raises it for the whole process
    assert status == 0, out + err
