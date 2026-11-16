"""The voice recipe's bridge: OpenAI's `mcp_approval_request` parks one of our approvals."""

import asyncio
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "connect_server.py"
HOST, AGENT, REVIEWER = "tools.example.com", "agent-test-token", "reviewer-test-token"
ITEM = {
    "type": "mcp_approval_request",
    "id": "mcpr_1",
    "server_label": "shop_tools",
    "name": "send_invoice",
    "arguments": json.dumps({"customer_id": "cus_1024", "amount": 49.5}),
}


async def _replay() -> tuple[Any, ...]:
    import httpx2

    example = runpy.run_path(str(EXAMPLE))
    app = example["public_app"](example["registry"], HOST, AGENT, REVIEWER)

    def http(token: str) -> httpx2.AsyncClient:
        peer = httpx2.ASGITransport(app=app, client=("203.0.113.9", 5000))
        auth = {"Authorization": f"Bearer {token}"}
        return httpx2.AsyncClient(transport=peer, base_url=f"https://{HOST}", headers=auth)

    async with app.router.lifespan_context(app), http(AGENT) as rest, http(REVIEWER) as reviewer:
        park = example["park_realtime_approval"]
        approval_id, reply = await park(ITEM, rest, "sess_1")
        again, _ = await park(ITEM, rest, "sess_1")  # the same event, replayed
        decided = await reviewer.post(f"/approvals/{approval_id}/decide", json={"approved": True})
        events = example["realtime_outcome"](approval_id, decided.json())
        listed = (await reviewer.get("/approvals", params={"run_id": "sess_1"})).json()
    return approval_id, again, reply, events, listed


def test_mcp_approval_request_parks_once_and_runs_on_our_decision() -> None:
    pytest.importorskip("mcp")
    pytest.importorskip("starlette")
    approval_id, again, reply, events, listed = asyncio.run(_replay())
    assert approval_id.startswith("apr_") and again == approval_id
    assert reply["item"]["type"] == "mcp_approval_response"
    assert reply["item"]["approval_request_id"] == "mcpr_1"
    assert reply["item"]["approve"] is False  # OpenAI never runs it
    said = events[0]["item"]["content"][0]["text"]
    assert said == f"Approval {approval_id}: ok, result invoiced cus_1024 for 49.50 USD"
    assert events[1] == {"type": "response.create"}
    assert [item["approval_id"] for item in listed["approvals"]] == [approval_id]
