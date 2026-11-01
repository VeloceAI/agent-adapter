"""Deciding re-authorizes: edited arguments and the invoker's final call (fix round 1)."""

from typing import Any

from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.core.context import ToolContext
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.server.auth import APPROVER_SCOPE, Principal, bearer_auth, is_invoker
from agent_adapter.server.service import ToolService
from tests.server.helpers import TOKENS, call, invoke, make_service

CORP = {"to": "x@corp.com", "body": "hi"}
EVIL = {"to": "x@evil.com", "body": "hi"}


def mail_registry(log: list[str]) -> Registry:
    reg = Registry()

    @reg.tool(confirm=True)
    async def send_email(to: str, body: str) -> str:
        """Send an email (needs approval)."""
        log.append(to)
        return f"sent {to}"

    return reg


async def corp_only(principal: Principal, spec: ToolSpec, args: Any) -> bool:
    """Argument-level rule: mail may only go to corp.com (approvers: anywhere)."""
    if args is None or APPROVER_SCOPE in principal.scopes:
        return True
    return str(args.get("to", "")).endswith("@corp.com")


async def parked(service: ToolService, arguments: dict[str, Any], **kwargs: Any) -> str:
    reply = await invoke(service, "send_email", arguments, **kwargs)
    assert reply.status == 202, reply.body
    approval_id: str = reply.body["approval_id"]
    return approval_id


async def test_the_invoker_cannot_approve_its_own_call_with_forbidden_edits():
    log: list[str] = []
    service = make_service(mail_registry(log), authorize=corp_only)
    approval_id = await parked(service, CORP)
    reply = await call(
        service,
        "POST",
        f"/approvals/{approval_id}/decide",
        body={"approved": True, "edited_arguments": EVIL},
    )
    assert reply.status == 403 and reply.body["type"].endswith(":forbidden")
    assert log == []
    shown = await call(service, "GET", f"/approvals/{approval_id}")
    assert shown.body["status"] == "pending"  # nothing recorded


async def test_an_approver_may_edit_but_the_invoker_must_still_be_allowed_the_result():
    log: list[str] = []
    service = make_service(mail_registry(log), authorize=corp_only)
    approval_id = await parked(service, CORP)
    reply = await call(
        service,
        "POST",
        f"/approvals/{approval_id}/decide",
        body={"approved": True, "edited_arguments": EVIL},
        token="t-reviewer",
    )
    assert reply.status == 403  # the reviewer may send there; the invoker may not
    assert log == []
    allowed = {"to": "y@corp.com", "body": "edited"}
    ok = await call(
        service,
        "POST",
        f"/approvals/{approval_id}/decide",
        body={"approved": True, "edited_arguments": allowed},
        token="t-reviewer",
    )
    assert ok.status == 200 and log == ["y@corp.com"]


async def test_a_revoked_permission_stops_the_approved_call():
    log: list[str] = []
    revoked: set[str] = set()

    async def rule(principal: Principal, spec: ToolSpec, args: Any) -> bool:
        return principal.id not in revoked

    service = make_service(mail_registry(log), authorize=rule)
    approval_id = await parked(service, CORP)
    revoked.add("alice@example.com")
    reply = await call(
        service,
        "POST",
        f"/approvals/{approval_id}/decide",
        body={"approved": True},
        token="t-reviewer",
    )
    assert reply.status == 403 and log == []


async def test_a_denial_needs_no_authorization_of_the_call():
    async def nobody(principal: Principal, spec: ToolSpec, args: Any) -> bool:
        return args is None or principal.id != "alice@example.com" or args["to"] == "x@corp.com"

    log: list[str] = []
    service = make_service(mail_registry(log), authorize=nobody)
    approval_id = await parked(service, CORP)
    reply = await call(
        service, "POST", f"/approvals/{approval_id}/decide", body={"approved": False}
    )
    assert (reply.status, reply.body["status"]) == (200, "denied")


async def test_the_invoker_is_rebuilt_with_its_scopes_when_the_call_runs():
    seen: list[frozenset[str]] = []
    reg = Registry()

    @reg.tool(confirm=True)
    async def scoped(ctx: ToolContext) -> str:
        """Report the principal's scopes."""
        seen.append(ctx.extras["principal"].scopes)
        return "ok"

    tokens = {**TOKENS, "t-power": Principal("power", scopes=frozenset({"b", "a"}))}
    service = make_service(reg, authenticate=bearer_auth(tokens))
    reply = await invoke(service, "scoped", {}, token="t-power")
    record = await service.approvals.get(reply.body["approval_id"])
    assert record.request.metadata["principal_scopes"] == ["a", "b"]
    assert "claims" not in record.request.metadata
    await call(
        service,
        "POST",
        f"/approvals/{reply.body['approval_id']}/decide",
        body={"approved": True},
        token="t-reviewer",
    )
    assert seen == [frozenset({"a", "b"})]


async def test_anonymous_invokers_are_marked_and_never_match_invoker_less_records():
    store = MemoryApprovalStore()
    service = ToolService(mail_registry([]), allow_anonymous=True, approvals=store)
    approval_id = await parked(service, CORP, token=None)
    record = await store.get(approval_id)
    assert record.request.metadata["anonymous"] is True
    assert "principal" not in record.request.metadata
    assert is_invoker(Principal(""), record)
    await store.create(
        ApprovalRequest(approval_id="apr_orphan", run_id="r", call_id="c", tool="send_email")
    )
    assert (await call(service, "GET", "/approvals/apr_orphan", token=None)).status == 404
    orphan = await call(
        service, "POST", "/approvals/apr_orphan/decide", body={"approved": True}, token=None
    )
    assert orphan.status == 404
    legacy = ApprovalRequest(
        approval_id="apr_legacy",
        run_id="r2",
        call_id="c2",
        tool="send_email",
        metadata={"principal": ""},
    )
    await store.create(legacy)
    assert (await call(service, "GET", "/approvals/apr_legacy", token=None)).status == 404


async def test_mcp_parked_edits_are_authorized_for_the_decider_and_the_invoker():
    store = MemoryApprovalStore()
    service = make_service(mail_registry([]), authorize=corp_only, approvals=store)
    await store.create(
        ApprovalRequest(
            approval_id="apr_mcp",
            run_id="run_x",
            call_id="call_x",
            tool="send_email",
            arguments=CORP,
            metadata={"principal": "alice@example.com", "principal_scopes": []},
        )
    )
    reply = await call(
        service,
        "POST",
        "/approvals/apr_mcp/decide",
        body={"approved": True, "edited_arguments": EVIL},
        token="t-reviewer",
    )
    assert reply.status == 403
    assert (await store.get("apr_mcp")).status == "pending"
