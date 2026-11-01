"""Records the HTTP server parks never run through `Executor.resume` or a queue (sec M3)."""

import asyncio
import json
from typing import Any

import pytest

from agent_adapter import (
    ApprovalDecision,
    ApprovalRequest,
    DeferredApproval,
    Executor,
    MemoryApprovalStore,
    Registry,
    ToolCall,
)
from agent_adapter.core.tool import ToolSpec
from agent_adapter.errors import ToolUsageError
from agent_adapter.handlers import handle_envelope, tool_handler
from agent_adapter.server.auth import Principal
from tests.server.helpers import call, invoke, make_service

TARGET = {"target": "prod-db"}
MARKERS = [
    {"origin": "http"},
    {"principal": "alice@example.com", "principal_scopes": []},
    {"anonymous": True},
]


def wipe_registry(ran: list[str]) -> Registry:
    registry = Registry()

    @registry.tool(confirm=True)
    async def remove(target: str) -> str:
        """Remove a target (needs approval)."""
        ran.append(target)
        return f"removed {target}"

    return registry


class Revocable:
    """An `authorize` hook whose invokers the operator can revoke."""

    def __init__(self) -> None:
        self.revoked: set[str] = set()

    async def __call__(self, principal: Principal, spec: ToolSpec, args: Any) -> bool:
        return principal.id not in self.revoked


async def server_parked(ran: list[str]) -> tuple[Executor, MemoryApprovalStore, str, Any]:
    """Alice parks `remove` on the server, then the operator revokes her."""
    store = MemoryApprovalStore()
    base = Executor(wipe_registry(ran), approval=DeferredApproval(store))
    authorize = Revocable()
    service = make_service(base.registry, executor=base, authorize=authorize)
    reply = await invoke(service, "remove", TARGET)
    assert reply.status == 202, reply.body
    authorize.revoked.add("alice@example.com")
    approval_id = reply.body["approval_id"]
    refused = await call(
        service, "POST", f"/approvals/{approval_id}/decide", body={"approved": True}
    )
    assert refused.status == 403, refused.body
    return base, store, approval_id, reply.body


async def test_resume_of_a_server_record_is_refused_after_the_server_refused_it():
    ran: list[str] = []
    base, store, approval_id, _ = await server_parked(ran)
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await base.resume(approval_id, True, decided_by="ops")
    assert ran == []
    assert (await store.get(approval_id)).status == "pending"  # nothing decided


async def test_resume_of_a_server_record_decided_in_the_store_is_refused():
    ran: list[str] = []
    base, store, approval_id, _ = await server_parked(ran)
    await store.decide(approval_id, ApprovalDecision(approved=True, decided_by="ops"))
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await base.resume(approval_id)
    assert ran == [] and not (await store.get(approval_id)).consumed


def test_resume_sync_of_a_server_record_is_refused():
    ran: list[str] = []
    store = MemoryApprovalStore()
    executor = Executor(wipe_registry(ran), approval=DeferredApproval(store))
    created = asyncio.run(store.create(server_request()))
    asyncio.run(store.decide(created.approval_id, ApprovalDecision(approved=True)))
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        executor.resume_sync(created.approval_id, timeout=5)
    assert ran == []


def server_request(**metadata: Any) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="",
        run_id="run_s",
        call_id="call_s",
        tool="remove",
        arguments=TARGET,
        metadata=metadata or {"origin": "http", "principal": "alice@example.com"},
    )


@pytest.mark.parametrize("marker", MARKERS)
async def test_resume_refuses_every_server_marker(marker: dict[str, Any]):
    ran: list[str] = []
    store = MemoryApprovalStore()
    executor = Executor(wipe_registry(ran), approval=DeferredApproval(store))
    record = await store.create(server_request(**marker))
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await executor.resume(record.approval_id)
    assert ran == []


async def test_a_call_envelope_with_the_server_records_ids_is_refused():
    ran: list[str] = []
    base, store, approval_id, body = await server_parked(ran)
    await store.decide(approval_id, ApprovalDecision(approved=True, decided_by="ops"))
    record = await store.get(approval_id)
    envelope = {
        "tool": "remove",
        "arguments": TARGET,
        "run_id": record.request.run_id,
        "call_id": record.request.call_id,
    }
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await handle_envelope(base, envelope)
    assert ran == [] and not (await store.get(approval_id)).consumed
    assert body["approval_id"] == approval_id


@pytest.mark.parametrize("marker", MARKERS)
async def test_call_envelopes_refuse_every_server_marker(marker: dict[str, Any]):
    ran: list[str] = []
    store = MemoryApprovalStore()
    executor = Executor(wipe_registry(ran), approval=DeferredApproval(store))
    record = await store.create(server_request(**marker))
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    envelope = {"tool": "remove", "arguments": TARGET, "run_id": "run_s", "call_id": "call_s"}
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await handle_envelope(executor, envelope)
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await handle_envelope(executor, {"approval_id": record.approval_id})
    assert ran == []


def test_a_serverless_event_with_the_server_records_ids_is_refused():
    ran: list[str] = []
    store = MemoryApprovalStore()
    executor = Executor(wipe_registry(ran), approval=DeferredApproval(store))
    record = asyncio.run(store.create(server_request()))
    asyncio.run(store.decide(record.approval_id, ApprovalDecision(approved=True)))
    handler = tool_handler(executor, timeout=5)
    envelope = {"tool": "remove", "arguments": TARGET, "run_id": "run_s", "call_id": "call_s"}
    batch = {
        "Records": [
            {"eventSource": "aws:sqs", "messageId": "m-1", "body": _json(envelope)},
            {"eventSource": "aws:sqs", "messageId": "m-2", "body": _json(record.approval_id)},
        ]
    }
    failures = handler(batch)["batchItemFailures"]
    assert failures == [{"itemIdentifier": "m-1"}, {"itemIdentifier": "m-2"}]
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        handler(envelope)
    assert ran == []


def _json(value: object) -> str:
    body = {"approval_id": value} if isinstance(value, str) else value
    return json.dumps(body)


async def test_a_plain_record_still_resumes_through_both_paths():
    ran: list[str] = []
    store = MemoryApprovalStore()
    executor = Executor(wipe_registry(ran), approval=DeferredApproval(store))
    plain = server_request(thread_id="t-1")  # thread_id alone is not a server marker
    plain = plain.model_copy(update={"metadata": {"thread_id": "t-1"}})
    record = await store.create(plain)
    result = await executor.resume(record.approval_id, True, decided_by="ops")
    assert result.ok and ran == ["prod-db"]


async def test_a_direct_run_call_with_the_server_records_ids_is_refused():
    ran: list[str] = []
    base, store, approval_id, _ = await server_parked(ran)
    await store.decide(approval_id, ApprovalDecision(approved=True, decided_by="ops"))
    request = (await store.get(approval_id)).request
    call_ = ToolCall(name="remove", arguments=TARGET, id=request.call_id)
    result = await base.run_call(call_, base.context(run_id=request.run_id))
    assert not result.ok and result.error is not None
    assert result.error.kind == "tool-usage-error"
    assert ran == [] and not (await store.get(approval_id)).consumed


@pytest.mark.parametrize("marker", MARKERS)
async def test_a_plain_port_never_consumes_a_server_record(marker: dict[str, Any]):
    store = MemoryApprovalStore()
    port = DeferredApproval(store)
    record = await store.create(server_request(**marker))
    await store.decide(record.approval_id, ApprovalDecision(approved=True))
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await port.request(record.request)
    with pytest.raises(ToolUsageError, match="parked by the HTTP tool server"):
        await port.consume(record, record.request)
    assert not (await store.get(record.approval_id)).consumed


async def test_the_server_port_still_consumes_its_own_records():
    from agent_adapter.server._approval import ServerApproval

    store = MemoryApprovalStore()
    record = await store.create(server_request())
    decided = await store.decide(record.approval_id, ApprovalDecision(approved=True))
    decision = await ServerApproval(store).consume(decided, record.request)
    assert decision.approved and (await store.get(record.approval_id)).consumed
