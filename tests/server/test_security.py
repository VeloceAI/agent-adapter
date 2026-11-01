"""§15.5: run ownership, principal-scoped ids and approval authorisation (adversarial)."""

import asyncio
import logging
from typing import Any

import pytest

from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.registry import Registry
from agent_adapter.cost.ledger import Ledger
from agent_adapter.runtime.ids import scoped_id
from agent_adapter.server.auth import APPROVER_SCOPE, Principal, is_invoker
from tests.server.helpers import SECRET, call, drain, invoke, make_registry, make_service

EMAIL = {"to": "a@example.com", "body": "hi"}


async def alice_pending(service: Any, run_id: str = "r1", **kwargs: Any) -> dict[str, Any]:
    reply = await invoke(service, "send_email", EMAIL, extra={"run_id": run_id}, **kwargs)
    assert reply.status == 202
    body: dict[str, Any] = reply.body
    return body


async def test_a_second_principal_cannot_invoke_into_another_principals_run():
    service = make_service()
    assert (await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"})).status == 200
    bob = await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"}, token="t-bob")
    assert bob.status == 403 and bob.body["type"] == "urn:agent-adapter:error:forbidden"
    by_header = await invoke(
        service, "add", {"a": 1, "b": 1}, headers={"X-Run-Id": "r1"}, token="t-bob"
    )
    assert by_header.status == 403
    reviewer = await invoke(
        service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"}, token="t-reviewer"
    )
    assert reviewer.status == 403  # the approver scope decides approvals; it never joins runs
    assert (await invoke(service, "add", {"a": 2, "b": 2}, extra={"run_id": "r1"})).status == 200


async def test_the_body_run_id_beats_the_header():
    service = make_service()
    await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "mine"})
    reply = await invoke(
        service,
        "add",
        {"a": 1, "b": 1},
        extra={"run_id": "bobs"},
        headers={"X-Run-Id": "mine"},
        token="t-bob",
    )
    assert reply.status == 200 and reply.body["run_id"] == "bobs"


async def test_a_second_principal_cannot_read_events_or_cost_of_another_run():
    service = make_service(ledger=Ledger())
    await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"})
    assert (await call(service, "GET", "/runs/r1/events", token="t-bob")).status == 403
    assert (await call(service, "GET", "/runs/r1/cost", token="t-bob")).status == 403
    assert (await call(service, "GET", "/runs/r1/events", token="t-reviewer")).status == 403
    assert (await call(service, "GET", "/runs/r1/cost")).status == 200


async def test_a_second_principal_cannot_list_read_or_decide_another_runs_approvals():
    log: list[str] = []
    service = make_service(make_registry(log))
    parked = await alice_pending(service)
    approval_id = parked["approval_id"]
    by_run = await call(service, "GET", "/approvals", query={"run_id": "r1"}, token="t-bob")
    assert by_run.status == 403
    listed = await call(service, "GET", "/approvals", token="t-bob")
    assert listed.status == 200 and listed.body["approvals"] == []
    assert (await call(service, "GET", f"/approvals/{approval_id}", token="t-bob")).status == 404
    decide = await call(
        service, "POST", f"/approvals/{approval_id}/decide", body={"approved": True}, token="t-bob"
    )
    assert decide.status == 404  # no oracle: another principal's approval does not exist
    assert log == []
    mine = await call(service, "GET", "/approvals", query={"run_id": "r1"})
    assert [a["approval_id"] for a in mine.body["approvals"]] == [approval_id]


async def test_an_approver_may_list_read_and_decide_any_run():
    log: list[str] = []
    service = make_service(make_registry(log))
    parked = await alice_pending(service)
    approval_id = parked["approval_id"]
    listed = await call(service, "GET", "/approvals", token="t-reviewer")
    assert [a["approval_id"] for a in listed.body["approvals"]] == [approval_id]
    by_run = await call(service, "GET", "/approvals", query={"run_id": "r1"}, token="t-reviewer")
    assert [a["approval_id"] for a in by_run.body["approvals"]] == [approval_id]
    shown = await call(service, "GET", f"/approvals/{approval_id}", token="t-reviewer")
    assert shown.status == 200 and shown.body["run_id"] == "r1"
    decided = await call(
        service,
        "POST",
        f"/approvals/{approval_id}/decide",
        body={"approved": True},
        token="t-reviewer",
    )
    assert decided.status == 200 and log == ["send a@example.com"]
    after = await call(service, "GET", f"/approvals/{approval_id}")
    assert after.body["decision"]["decided_by"] == "reviewer"


async def test_a_distinct_approver_rule_stops_the_invoker_deciding():
    async def four_eyes(principal: Principal, record: ApprovalRecord) -> bool:
        return APPROVER_SCOPE in principal.scopes and not is_invoker(principal, record)

    service = make_service(authorize_decision=four_eyes)
    parked = await alice_pending(service)
    path = f"/approvals/{parked['approval_id']}/decide"
    own = await call(service, "POST", path, body={"approved": True})
    assert own.status == 403 and own.body["approval_id"] == parked["approval_id"]
    by_reviewer = await call(service, "POST", path, body={"approved": True}, token="t-reviewer")
    assert by_reviewer.status == 200


async def test_a_raising_decision_rule_fails_closed():
    async def broken(principal: Principal, record: ApprovalRecord) -> bool:
        raise RuntimeError(SECRET)

    service = make_service(authorize_decision=broken)
    parked = await alice_pending(service)
    reply = await call(
        service, "POST", f"/approvals/{parked['approval_id']}/decide", body={"approved": True}
    )
    assert reply.status == 500 and SECRET not in str(reply.body)
    shown = await call(service, "GET", f"/approvals/{parked['approval_id']}")
    assert shown.body["status"] == "pending"


async def test_reusing_another_principals_idempotency_key_gives_another_call_id():
    log: list[str] = []
    service = make_service(make_registry(log))
    key = {"Idempotency-Key": "k-1"}
    alice = await invoke(service, "add", {"a": 1, "b": 1}, headers=key)
    bob = await invoke(service, "add", {"a": 1, "b": 1}, headers=key, token="t-bob")
    assert alice.body["call_id"] == scoped_id("call_", "alice@example.com", "k-1")
    assert bob.body["call_id"] == scoped_id("call_", "bob@example.com", "k-1")
    assert alice.body["call_id"] != bob.body["call_id"]
    assert "Idempotent-Replayed" not in bob.headers  # not Alice's response
    assert log == ["add 1 1", "add 1 1"]


async def test_another_principals_key_never_resumes_their_pending_approval():
    log: list[str] = []
    service = make_service(make_registry(log))
    key = {"Idempotency-Key": "k-mail"}
    alice = await invoke(service, "send_email", EMAIL, headers=key)
    bob = await invoke(service, "send_email", EMAIL, headers=key, token="t-bob")
    assert bob.status == 202 and bob.body["approval_id"] != alice.body["approval_id"]
    assert bob.body["call_id"] != alice.body["call_id"]


async def test_a_replayed_key_with_other_arguments_is_refused():
    log: list[str] = []
    service = make_service(make_registry(log))
    key = {"Idempotency-Key": "k-2"}
    first = await invoke(service, "add", {"a": 1, "b": 1}, headers=key)
    same = await invoke(service, "add", {"a": 1, "b": 1}, headers=key)
    assert same.status == 200 and same.headers["Idempotent-Replayed"] == "true"
    assert same.body == first.body
    for other in (
        {"tool": "add", "arguments": {"a": 9, "b": 1}, "extra": {}},
        {"tool": "add", "arguments": {"a": 1, "b": 1}, "extra": {"run_id": "elsewhere"}},
        {"tool": "add", "arguments": {"a": 1, "b": 1}, "extra": {"thread_id": "t"}},
        {"tool": "charge", "arguments": {"amount": 1}, "extra": {}},
    ):
        reply = await invoke(
            service, other["tool"], other["arguments"], headers=key, extra=other["extra"]
        )
        assert reply.status == 400, other
        assert reply.body["detail"] == "this Idempotency-Key was used for another request"
    assert log == ["add 1 1"]


async def test_concurrent_duplicates_run_once():
    log: list[str] = []
    gate = asyncio.Event()
    reg = Registry()

    @reg.tool
    async def pay(amount: int) -> str:
        """Pay (slow)."""
        log.append(f"pay {amount}")
        await gate.wait()
        return "paid"

    service = make_service(reg)
    key = {"Idempotency-Key": "k-pay"}
    first = asyncio.ensure_future(invoke(service, "pay", {"amount": 5}, headers=key))
    await asyncio.sleep(0.01)
    second = await invoke(service, "pay", {"amount": 5}, headers=key)
    assert second.status == 409
    assert second.body["type"] == "urn:agent-adapter:error:request-in-progress"
    gate.set()
    assert (await first).status == 200
    third = await invoke(service, "pay", {"amount": 5}, headers=key)
    assert third.headers["Idempotent-Replayed"] == "true"
    assert log == ["pay 5"]


async def test_replaying_the_invoke_after_the_decision_returns_the_decided_outcome():
    log: list[str] = []
    service = make_service(make_registry(log))
    key = {"Idempotency-Key": "k-mail"}
    parked = await invoke(service, "send_email", EMAIL, headers=key)
    replay = await invoke(service, "send_email", EMAIL, headers=key)
    assert replay.status == 202 and replay.body == parked.body
    await call(
        service,
        "POST",
        f"/approvals/{parked.body['approval_id']}/decide",
        body={"approved": True},
        token="t-reviewer",
    )
    done = await invoke(service, "send_email", EMAIL, headers=key)
    assert done.status == 200 and done.body["result"]["sent"] is True
    assert done.headers["Idempotent-Replayed"] == "true"
    assert log == ["send a@example.com"]


async def test_sse_streams_never_carry_another_principals_events():
    service = make_service()
    service._state.owners = type(service._state.owners)(max_runs=1)  # forget owners quickly
    await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "shared"})
    await invoke(service, "add", {"a": 2, "b": 2}, extra={"run_id": "pushout"})  # evicts
    bob = await invoke(service, "whoami", {}, extra={"run_id": "shared"}, token="t-bob")
    assert bob.status == 200  # the owner entry is gone, but Bob lands in his own run
    stream = await call(
        service, "GET", "/runs/shared/events", query={"close_on": "tool.finished"}, token="t-bob"
    )
    frames = await drain(stream.raw)
    tools = [f["data"]["data"].get("tool") for f in frames if f["event"] != "end"]
    assert tools == ["whoami", "whoami"]
    assert all(f["data"]["run_id"] == "shared" for f in frames if f["event"] != "end")
    alice_run = scoped_id("run_", "alice@example.com", "shared")
    assert bob.body["result"]["run_id"] != alice_run


async def test_tools_see_scoped_run_and_thread_ids_and_the_principal():
    service = make_service()
    extra = {"run_id": "r1", "thread_id": "t1"}
    alice = await invoke(service, "whoami", {}, extra=extra)
    bob = await invoke(service, "whoami", {}, extra={**extra, "run_id": "r2"}, token="t-bob")
    seen_a, seen_b = alice.body["result"], bob.body["result"]
    assert seen_a["run_id"] == scoped_id("run_", "alice@example.com", "r1")
    assert seen_a["thread_id"] == scoped_id("thr_", "alice@example.com", "t1")
    assert seen_a["thread_id"] != seen_b["thread_id"]  # one thread key, two principals
    assert (seen_a["principal"], seen_b["principal"]) == ("alice@example.com", "bob@example.com")


async def test_records_name_the_invoker_and_the_scoped_run():
    service = make_service()
    parked = await alice_pending(service)
    record = await service.approvals.get(parked["approval_id"])
    assert record.request.run_id == scoped_id("run_", "alice@example.com", "r1")
    assert record.request.metadata["principal"] == "alice@example.com"
    assert record.request.metadata["run_key"] == "r1"
    assert record.request.metadata["origin"] == "http"


async def test_logs_hold_no_tokens_secrets_or_principal_ids(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    service = make_service()
    await invoke(service, "boom", {"token": SECRET})
    await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"}, token="t-bob")
    await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": "r1"})
    parked = await alice_pending(service, run_id="r9")
    await call(service, "GET", f"/approvals/{parked['approval_id']}", token="t-bob")
    for leaked in ("t-alice", "t-bob", "alice@example.com", "bob@example.com"):
        assert leaked not in caplog.text, leaked
    # The executor logs a failing tool's traceback for operators; the server adds nothing.
    server = [r.getMessage() for r in caplog.records if r.name.startswith("agent_adapter.server")]
    assert server and not any(SECRET in line for line in server)


async def test_concurrent_decisions_run_the_call_once():
    log: list[str] = []
    service = make_service(make_registry(log))
    parked = await alice_pending(service)
    path = f"/approvals/{parked['approval_id']}/decide"
    replies = await asyncio.gather(
        call(service, "POST", path, body={"approved": True}),
        call(service, "POST", path, body={"approved": True}, token="t-reviewer"),
        call(service, "POST", path, body={"approved": False}, token="t-reviewer"),
    )
    assert sorted(r.status for r in replies) == [200, 409, 409]
    assert log == ["send a@example.com"]


async def test_another_principal_cannot_follow_a_server_generated_run():
    service = make_service()
    mine = await invoke(service, "add", {"a": 1, "b": 1})
    run_key = mine.body["run_id"]
    assert (await call(service, "GET", f"/runs/{run_key}/events", token="t-bob")).status == 403
    assert (
        await invoke(service, "add", {"a": 1, "b": 1}, extra={"run_id": run_key}, token="t-bob")
    ).status == 403


async def test_a_failed_call_replays_its_failure_without_running_again():
    log: list[str] = []
    service = make_service(make_registry(log))
    key = {"Idempotency-Key": "k-charge"}
    first = await invoke(service, "charge", {"amount": 1}, headers=key, extra={"run_id": "r"})
    other = await invoke(service, "charge", {"amount": 2}, extra={"run_id": "r"})
    assert (first.status, other.status) == (200, 409)
    boom = {"Idempotency-Key": "k-boom"}
    failed = await invoke(service, "boom", {"token": SECRET}, headers=boom)
    again = await invoke(service, "boom", {"token": SECRET}, headers=boom)
    assert (failed.status, again.status) == (502, 502)
    assert again.headers["Idempotent-Replayed"] == "true" and again.body == failed.body
    assert log == ["charge 1"]
