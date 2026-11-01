from typing import Any

from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.types import ApprovalRequest, LedgerEntry
from agent_adapter.cost.ledger import Ledger
from tests.server.helpers import call, drain, invoke, make_registry, make_service

EMAIL = {"to": "a@example.com", "body": "hi"}


async def pending(service: Any, **kwargs: Any) -> dict[str, Any]:
    reply = await invoke(service, "send_email", EMAIL, **kwargs)
    assert reply.status == 202, reply.body
    assert reply.headers["Location"] == f"/approvals/{reply.body['approval_id']}"
    assert reply.headers["Retry-After"] == "2"
    body: dict[str, Any] = reply.body
    return body


async def test_pending_then_approved_runs_the_tool_once():
    log: list[str] = []
    service = make_service(make_registry(log))
    parked = await pending(service, extra={"run_id": "r1", "thread_id": "t1"})
    assert (parked["status"], parked["run_id"]) == ("pending_approval", "r1")
    assert log == []
    shown = await call(service, "GET", f"/approvals/{parked['approval_id']}")
    assert shown.status == 200
    assert {k: shown.body[k] for k in ("status", "tool", "arguments", "run_id", "call_id")} == {
        "status": "pending",
        "tool": "send_email",
        "arguments": EMAIL,
        "run_id": "r1",
        "call_id": parked["call_id"],
    }
    assert shown.body["decision"] is None
    decided = await call(
        service, "POST", f"/approvals/{parked['approval_id']}/decide", body={"approved": True}
    )
    assert decided.status == 200
    assert decided.body == {
        "status": "ok",
        "run_id": "r1",
        "call_id": parked["call_id"],
        "result": {"sent": True, "to": "a@example.com"},
        "cached": False,
        "attempts": 1,
    }
    assert log == ["send a@example.com"]
    after = await call(service, "GET", f"/approvals/{parked['approval_id']}")
    assert after.body["status"] == "approved"
    assert after.body["decision"]["decided_by"] == "alice@example.com"
    again = await call(
        service, "POST", f"/approvals/{parked['approval_id']}/decide", body={"approved": True}
    )
    assert again.status == 409 and log == ["send a@example.com"]


async def test_edited_arguments_run_and_are_validated_before_deciding():
    log: list[str] = []
    service = make_service(make_registry(log))
    parked = await pending(service)
    path = f"/approvals/{parked['approval_id']}/decide"
    bad = await call(service, "POST", path, body={"approved": True, "edited_arguments": {"to": 1}})
    assert bad.status == 422
    assert (await call(service, "GET", f"/approvals/{parked['approval_id']}")).body[
        "status"
    ] == "pending"
    edited = {"to": "b@example.com", "body": "edited"}
    ok = await call(service, "POST", path, body={"approved": True, "edited_arguments": edited})
    assert ok.status == 200 and ok.body["result"]["to"] == "b@example.com"
    assert log == ["send b@example.com"]


async def test_denied_returns_denied_and_never_runs():
    log: list[str] = []
    service = make_service(make_registry(log))
    parked = await pending(service)
    reply = await call(
        service,
        "POST",
        f"/approvals/{parked['approval_id']}/decide",
        body={"approved": False, "reason": "not now"},
    )
    assert reply.status == 200
    assert reply.body == {
        "status": "denied",
        "run_id": parked["run_id"],
        "call_id": parked["call_id"],
        "approval_id": parked["approval_id"],
        "reason": "not now",
    }
    assert log == []


async def test_decide_body_errors_and_unknown_approvals():
    service = make_service()
    parked = await pending(service)
    path = f"/approvals/{parked['approval_id']}/decide"
    for body in (
        {},
        {"approved": "yes"},
        {"approved": True, "reason": 5},
        {"approved": True, "edited_arguments": [1]},
        {"approved": True, "x": 1},
    ):
        assert (await call(service, "POST", path, body=body)).status == 400, body
    assert (
        await call(service, "POST", "/approvals/apr_nope/decide", body={"approved": True})
    ).status == 404
    assert (await call(service, "GET", "/approvals/apr_nope")).status == 404


async def test_list_approvals_by_run_and_status():
    service = make_service()
    first = await pending(service, extra={"run_id": "r1"})
    await pending(service, extra={"run_id": "r2"})
    everything = await call(service, "GET", "/approvals")
    assert len(everything.body["approvals"]) == 2
    only_r1 = await call(service, "GET", "/approvals", query={"run_id": "r1"})
    assert [a["approval_id"] for a in only_r1.body["approvals"]] == [first["approval_id"]]
    await call(
        service, "POST", f"/approvals/{first['approval_id']}/decide", body={"approved": False}
    )
    denied = await call(service, "GET", "/approvals", query={"status": "denied"})
    assert [a["approval_id"] for a in denied.body["approvals"]] == [first["approval_id"]]
    assert (await call(service, "GET", "/approvals", query={"status": "nah"})).status == 400


async def test_approvals_parked_by_another_surface_are_decided_not_run():
    store = MemoryApprovalStore()
    service = make_service(approvals=store)
    request = ApprovalRequest(
        approval_id="apr_mcp",
        run_id="run_x",
        call_id="call_x",
        tool="send_email",
        arguments=EMAIL,
        metadata={"principal": "alice@example.com"},
    )
    await store.create(request)
    reply = await call(service, "POST", "/approvals/apr_mcp/decide", body={"approved": True})
    assert reply.status == 202
    assert reply.body == {
        "status": "approved",
        "run_id": "run_x",
        "call_id": "call_x",
        "approval_id": "apr_mcp",
        "note": reply.body["note"],
    }
    record = await store.get("apr_mcp")
    assert (record.status, record.consumed) == ("approved", False)  # its client resumes it
    await store.create(request.model_copy(update={"approval_id": "apr_2", "call_id": "c2"}))
    no = await call(service, "POST", "/approvals/apr_2/decide", body={"approved": False})
    assert (no.status, no.body["status"]) == (200, "denied")


async def test_events_replay_with_last_event_id_and_close_on():
    service = make_service()
    await invoke(service, "add", {"a": 1, "b": 2}, extra={"run_id": "r1"})
    reply = await call(service, "GET", "/runs/r1/events", query={"close_on": "tool.finished"})
    assert reply.status == 200 and reply.media_type == "text/event-stream"
    assert reply.headers["Cache-Control"] == "no-cache"
    frames = await drain(reply.raw)
    assert [f["event"] for f in frames] == ["tool.started", "tool.finished", "end"]
    assert frames[0]["id"] == "1" and frames[0]["data"]["run_id"] == "r1"
    assert frames[1]["data"]["data"]["tool"] == "add"
    replay = await call(
        service,
        "GET",
        "/runs/r1/events",
        headers={"Last-Event-ID": "1"},
        query={"close_on": "tool.finished"},
    )
    assert [f["id"] for f in await drain(replay.raw)] == ["2", None]
    after = await call(
        service, "GET", "/runs/r1/events", query={"after": "1", "close_on": "tool.finished"}
    )
    assert [f["event"] for f in await drain(after.raw)] == ["tool.finished", "end"]
    for bad in ("x", "-1", "²"):
        assert (await call(service, "GET", "/runs/r1/events", query={"after": bad})).status == 400


async def test_cost_needs_a_ledger_and_summarises_the_run():
    assert (await call(make_service(), "GET", "/runs/r1/cost")).status == 404
    ledger = Ledger()
    service = make_service(ledger=ledger)
    await invoke(service, "add", {"a": 1, "b": 2}, extra={"run_id": "r1"})
    reply = await call(service, "GET", "/runs/r1/cost")
    assert reply.status == 200
    assert reply.body["run_id"] == "r1" and reply.body["by_kind"]["tool"]["calls"] == 1
    other = await call(service, "GET", "/runs/r2/cost")
    assert other.body["by_kind"] == {}
    assert isinstance(ledger.entries()[0], LedgerEntry)
