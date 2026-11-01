"""Nested approvals and records without an invoker (fix round 2)."""

from __future__ import annotations

from typing import Any

from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.runtime.current import use_context
from agent_adapter.server._approval import with_metadata
from agent_adapter.server.auth import Principal
from agent_adapter.server.service import ToolService
from tests.server.helpers import call, invoke, make_service

KEY = {"Idempotency-Key": "n"}


def nested_registry(
    log: list[str], holder: dict[str, ToolService], *, inner_http: bool
) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True, targets=["http"] if inner_http else ["mcp"])
    async def inner(item: str) -> str:
        """Needs approval; called only from `outer`."""
        log.append(f"inner {item}")
        return f"inner {item}"

    @reg.tool(targets=["http"])
    async def outer(item: str, ctx: ToolContext) -> str:
        """Calls `inner` through the server's executor."""
        value = await holder["service"].executor.run("inner", {"item": item}, ctx)
        return f"outer saw {value}"

    return reg


async def setup(inner_http: bool) -> tuple[ToolService, list[str], dict[str, Any]]:
    log: list[str] = []
    holder: dict[str, ToolService] = {}
    service = make_service(nested_registry(log, holder, inner_http=inner_http))
    holder["service"] = service
    parked = await invoke(service, "outer", {"item": "x"}, headers=KEY)
    assert parked.status == 202, parked.body
    return service, log, parked.body


async def test_a_nested_record_carries_the_invoker_but_not_the_outer_replay_key():
    service, _, parked = await setup(inner_http=True)
    record = await service.approvals.get(parked["approval_id"])
    metadata = record.request.metadata
    assert metadata["principal"] == "alice@example.com"
    assert not {"origin", "idempotency", "fingerprint", "run_key"} & set(metadata)


async def test_replaying_the_outer_key_still_returns_the_outer_response():
    service, log, parked = await setup(inner_http=True)
    decided = await call(
        service, "POST", f"/approvals/{parked['approval_id']}/decide", body={"approved": True}
    )
    assert decided.status == 202 and decided.body["status"] == "approved"  # decided only
    assert log == []
    replay = await invoke(service, "outer", {"item": "x"}, headers=KEY)
    assert (replay.status, replay.body) == (202, parked)  # outer's own response, never inner's


async def test_a_nested_tool_not_exported_to_http_can_be_decided():
    service, _, parked = await setup(inner_http=False)
    path = f"/approvals/{parked['approval_id']}"
    assert (await call(service, "GET", path)).status == 200
    decided = await call(service, "POST", f"{path}/decide", body={"approved": True})
    assert decided.status == 202 and decided.body["status"] == "approved"
    denied_too = await call(service, "POST", f"{path}/decide", body={"approved": False})
    assert denied_too.status == 409


def test_only_an_anonymous_principal_is_marked_anonymous():
    request = ApprovalRequest(approval_id="a", run_id="r", call_id="c", tool="t")
    for principal, expected in (
        (Principal(""), {"anonymous": True}),
        (None, {}),
        ("", {}),
        (42, {}),
        ("svc", {"principal": "svc", "principal_scopes": []}),
    ):
        ctx = ToolContext(run_id="r", call_id="c", extras={"principal": principal})
        with use_context(ctx):
            assert with_metadata(request).metadata == expected, principal


async def test_a_record_without_an_invoker_cannot_run():
    store = MemoryApprovalStore()
    reg = Registry()

    @reg.tool(confirm=True)
    def act() -> str:
        """Needs approval."""
        return "ran"

    service = make_service(reg, approvals=store, allow_anonymous=True)
    await store.create(ApprovalRequest(approval_id="apr_x", run_id="r", call_id="c", tool="act"))
    by_approver = await call(
        service, "POST", "/approvals/apr_x/decide", body={"approved": True}, token="t-reviewer"
    )
    assert by_approver.status == 403
    by_anonymous = await call(
        service, "POST", "/approvals/apr_x/decide", body={"approved": True}, token=None
    )
    assert by_anonymous.status == 404
    assert (await store.get("apr_x")).status == "pending"


async def test_an_approval_decided_elsewhere_says_when_it_runs():
    service, log, parked = await setup(inner_http=False)
    decided = await call(
        service, "POST", f"/approvals/{parked['approval_id']}/decide", body={"approved": True}
    )
    assert decided.status == 202
    note = decided.body["note"]
    assert "same call id" in note and "re-runs" in note and "unused" in note
    assert log == []  # approved, not run
