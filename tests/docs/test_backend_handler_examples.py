"""The serverless, queue, pipeline, notebook and other-framework examples run offline."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agent_adapter import run_sync

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
DEV = {"Authorization": "Bearer dev-token"}
JSON_DEV = {**DEV, "Content-Type": "application/json"}
INVOICE = {"tool": "send_invoice", "arguments": {"customer_id": "cus_1024", "amount": 49.5}}
RATE = {"tool": "shipping_rate", "arguments": {"region": "us"}}


@pytest.fixture(autouse=True)
def _examples_importable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    yield


def load(name: str) -> ModuleType:
    """Import `examples/<name>.py` fresh (and a fresh `backend_common`)."""
    sys.modules.pop("backend_common", None)
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sqs(*bodies: dict[str, Any]) -> dict[str, Any]:
    return {
        "Records": [
            {"messageId": f"m{i}", "body": json.dumps(b), "eventSource": "aws:sqs"}
            for i, b in enumerate(bodies)
        ]
    }


# --- serverless ---


def test_serverless_lambda_parks_reviews_and_resumes_once():
    lam = load("backend_serverless")
    assert lam.handler(RATE)["result"] == 18.0
    assert lam.handler(sqs(RATE, INVOICE, {"tool": "nope"})) == {
        "batchItemFailures": [{"itemIdentifier": "m2"}]
    }
    (record,) = run_sync(lam.SHARED_STORE.list())
    sent: list[dict[str, Any]] = []
    status, _ = lam.review(
        record.approval_id, reviewer="dev@example.com", approved=True, send=sent.append
    )
    assert (status, sent) == (403, [])
    status, _ = lam.review(
        record.approval_id, reviewer="reviewer@example.com", approved=True, send=sent.append
    )
    assert (status, sent) == (200, [{"approval_id": record.approval_id}])
    assert lam.handler(sqs(*sent)) == {"batchItemFailures": []}
    redelivered = lam.handler(sqs(*sent))  # used once: a duplicate runs nothing, is acked
    assert redelivered == {"batchItemFailures": []}


async def test_serverless_async_function_returns_problems():
    lam = load("backend_serverless")
    status, reply = await lam.tool_function(RATE)
    assert (status, reply["status"], reply["result"]) == (200, "ok", 18.0)
    status, _ = await lam.tool_function({"tool": "wipe_disk"})
    assert status == 404
    status, _ = await lam.tool_function({"tool": "shipping_rate", "approved": True})
    assert status == 400


# --- queues ---


async def test_queue_consumer_acks_parks_dead_letters_and_redelivers():
    queues = load("backend_queue")
    queue = queues.FakeQueue()
    for i, body in enumerate([RATE, INVOICE, {"tool": "x", "approved": True}]):
        queue.pending.append(queues.Message(f"k{i}", json.dumps(body).encode()))
    replies = await queues.consume_once(queue)
    assert [r["status"] for r in replies] == ["ok", "pending_approval"]
    assert (queue.acked, queue.dead) == (["k0", "k1", "k2"], ["k2"])

    queue.pending.append(queues.Message("k1", json.dumps(INVOICE).encode()))  # redelivered
    (again,) = await queues.consume_once(queue)
    assert again["approval_id"] == replies[1]["approval_id"]


def test_queue_sync_job_and_activity():
    queues = load("backend_queue")
    assert queues.tool_job(RATE)["result"] == 18.0
    assert run_sync(queues.tool_activity(RATE))["result"] == 18.0


async def test_queue_pubsub_push_uses_the_stable_message_id():
    queues = load("backend_queue")
    data = base64.b64encode(json.dumps(INVOICE).encode()).decode()
    body = {"message": {"data": data, "messageId": "ps-1"}, "subscription": "s"}
    first, again = await queues.pubsub_push(body), await queues.pubsub_push(body)
    assert first["approval_id"] == again["approval_id"]


# --- pipelines ---


def test_pipeline_task_partition_and_one_backend_per_process():
    pipes = load("backend_pipeline")
    out = pipes.shipping_rates_sync(["eu", "us"])
    assert out == {"rates": {"eu": 5.0, "us": 18.0}, "cost_usd": 0.0}
    assert list(pipes.rates_for_partition(["us", "eu", "us"])) == [18.0, 5.0, 18.0]
    assert pipes.backend() is pipes.backend()


# --- notebooks ---


class FakeStreamlit:
    """Records what the page shows; `cache_resource` caches like Streamlit's."""

    def __init__(self, choice: str) -> None:
        self.choice, self.shown, self.cache = choice, [], {}

    def cache_resource(self, func: Any) -> Any:
        return lambda: self.cache.setdefault(func, func())

    def selectbox(self, label: str, options: list[str]) -> str:
        assert self.choice in options
        return self.choice

    def metric(self, label: str, value: Any) -> None:
        self.shown.append((label, value))


async def test_notebook_cell_streamlit_and_gradio():
    books = load("backend_notebook")
    backend = sys.modules["backend_common"].build_backend()
    assert await books.notebook_cell(backend) == 18.0
    assert await books.gradio_handler(backend)("eu") == "5.0%"
    st = FakeStreamlit("us")
    books.streamlit_page(st)  # inside a running loop: run_sync uses a worker thread
    books.streamlit_page(st)
    assert st.shown == [("Shipping %", 18.0), ("Shipping %", 18.0)] and len(st.cache) == 1


# --- other frameworks ---


async def test_other_asgi_forward_and_grpc_servicer():
    other = load("backend_other_asgi")
    backend = sys.modules["backend_common"].build_backend()
    service = other.tool_service(backend)
    body = json.dumps({"arguments": {"region": "us"}}).encode()
    status, headers, raw = await other.forward(
        service, "POST", "/tools/shipping_rate/invoke", JSON_DEV, body, "127.0.0.1"
    )
    assert (status, json.loads(raw)["result"]) == (200, 18.0)
    assert headers["Content-Type"] == "application/json"
    anonymous = {"Content-Type": "application/json"}
    status, _, _ = await other.forward(
        service, "POST", "/tools/shipping_rate/invoke", anonymous, body, None
    )
    assert status == 401
    servicer = other.ToolServicer(backend)
    request = SimpleNamespace(tool="shipping_rate", arguments=json.dumps({"region": "eu"}))
    reply = await servicer.Invoke(request, None)
    assert reply["status"] == 200 and json.loads(reply["body"])["result"] == 5.0


def test_other_asgi_starlette_mount():
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient

    other = load("backend_other_asgi")
    with TestClient(other.starlette_app(), base_url="http://localhost") as client:
        reply = client.post(
            "/agent/tools/shipping_rate/invoke", json={"arguments": {"region": "eu"}}, headers=DEV
        )
        assert (reply.status_code, reply.json()["result"]) == (200, 5.0)


async def test_other_asgi_aiohttp_shim():
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    other = load("backend_other_asgi")
    service = other.tool_service(sys.modules["backend_common"].build_backend())
    async with TestClient(TestServer(other.aiohttp_app(service))) as client:
        reply = await client.post(
            "/tools/shipping_rate/invoke", json={"arguments": {"region": "eu"}}, headers=DEV
        )
        assert reply.status == 200 and (await reply.json())["result"] == 5.0
