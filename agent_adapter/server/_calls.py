"""`POST /tools/{name}/invoke` and `POST /approvals/{id}/decide`: run tools for their owner."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalAlreadyDecided, ApprovalNotFound
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ToolCall, ToolResult, to_jsonable
from agent_adapter.runtime.ids import new_id
from agent_adapter.server._approval import CALL_EXTRAS, ORIGIN_HTTP
from agent_adapter.server._idempotency import fingerprint
from agent_adapter.server._payloads import (
    InvokeBody,
    client_key,
    decide_body,
    invoke_body,
    refuse,
    run_key_of,
)
from agent_adapter.server._runs import call_id_for, owner_key, run_id_for, thread_id_for
from agent_adapter.server._state import (
    ServiceResponse,
    State,
    can_read,
    json_response,
    problem_response,
)
from agent_adapter.server.auth import Principal, RequestInfo, invoker_of, principal_ref
from agent_adapter.server.idempotency import Claim, Stored
from agent_adapter.server.problems import TYPE_PREFIX, problem, problem_for

logger = logging.getLogger(__name__)

REPLAYED = "Idempotent-Replayed"
NOT_KEPT = frozenset({"run-limit-exceeded", "budget-exceeded"})
ELSEWHERE_NOTE = (
    "approved, not run: the call runs only when its caller re-runs it with the same call id "
    "(an MCP client resuming it, or an outer tool that re-runs a nested call with a fixed "
    "call_id); otherwise the approval stays unused"
)


async def invoke(
    state: State, principal: Principal, request: RequestInfo, body: bytes, name: str
) -> ServiceResponse:
    """Run one tool call for `principal`; a repeated `Idempotency-Key` replays its response."""
    spec = state.spec(name)
    parsed = invoke_body(body, request.header("x-run-id"), request.header("x-thread-id"))
    if not await state.allowed(principal, spec, parsed.arguments):
        logger.warning("principal %s may not call %s", principal_ref(principal.id), spec.name)
        refuse("forbidden", "not allowed to call this tool", tool=spec.name)
    if parsed.run_key and not state.owners.claim(parsed.run_key, principal.id):
        logger.warning(
            "principal %s refused: run owned by another (tool %s)",
            principal_ref(principal.id),
            spec.name,
        )
        refuse("forbidden", "this run belongs to another principal")
    key = client_key(request.header("idempotency-key"), "Idempotency-Key")
    if not key:
        return await _execute(state, principal, spec, parsed, new_id("call_"))
    cache_key = state.idempotency.key(owner_key(principal), key)
    sent = fingerprint(spec.name, parsed.arguments, parsed.run_key, parsed.thread_key)
    lease = state.idempotency.lease_for(spec.timeout, spec.retries)
    claim = await state.idempotency.claim(cache_key, sent, lease)
    if claim.state != "new":
        return _held(claim, sent, spec.name)
    try:
        response = await _execute(
            state, principal, spec, parsed, call_id_for(principal, key), (cache_key, sent)
        )
    except BaseException:
        await state.idempotency.release(cache_key, claim.token)
        raise
    if _refused_before_running(response):
        await state.idempotency.release(cache_key, claim.token)
    else:
        await state.idempotency.complete(cache_key, claim.token, _stored(sent, response))
    return response


def _held(claim: Claim, sent: str, tool: str) -> ServiceResponse:
    """A key someone already holds: replay its outcome, or refuse a busy or reused key."""
    if claim.fingerprint and claim.fingerprint != sent:
        logger.warning("Idempotency-Key reused for another request (tool %s)", tool)
        refuse("bad-request", "this Idempotency-Key was used for another request")
    if claim.state == "done" and claim.stored is not None:
        return _replay(claim.stored)
    refuse("request-in-progress", "a request with this Idempotency-Key is in progress")


def _refused_before_running(response: ServiceResponse) -> bool:
    """Run limits and budgets refuse before the tool runs, so a retry is evaluated again."""
    body = response.body if isinstance(response.body, dict) else {}
    return str(body.get("type", "")).removeprefix(TYPE_PREFIX) in NOT_KEPT


async def _execute(
    state: State,
    principal: Principal,
    spec: ToolSpec,
    parsed: InvokeBody,
    call_id: str,
    replay: tuple[str, str] = ("", ""),
) -> ServiceResponse:
    run_key = parsed.run_key
    if not run_key:
        run_key = new_id("run_")
        state.owners.claim(run_key, principal.id)
    ctx = state.executor.context(
        run_id=run_id_for(owner_key(principal), run_key),
        thread_id=thread_id_for(principal, parsed.thread_key),
        extras={
            "principal": principal,
            CALL_EXTRAS: {
                "call_id": call_id,
                "run_key": run_key,
                "idempotency": replay[0],
                "fingerprint": replay[1],
            },
        },
    )
    call = ToolCall(name=spec.name, arguments=parsed.arguments, id=call_id)
    result = await state.executor.run_call(call, ctx, registry=state.tools())
    logger.info(
        "tool %s for %s: %s (call %s)",
        spec.name,
        principal_ref(principal.id),
        result.status,
        result.call_id,
    )
    return outcome(result, run_key, ctx)


def outcome(result: ToolResult, run_key: str, ctx: ToolContext) -> ServiceResponse:
    """InvokeOk (200), InvokePending (202) or the problem of a failed call."""
    if result.ok:
        value = _json_value(result.value)
        if value is _UNENCODABLE:
            logger.warning("tool %s returned a value that is not JSON", result.name)
            return problem_response(problem("internal-error", "the tool result is not JSON"))
        return json_response(
            200,
            {
                "status": "ok",
                "run_id": run_key,
                "call_id": result.call_id,
                "result": value,
                "cached": result.cached,
                "attempts": result.attempts,
            },
        )
    if result.status == "pending":
        approval_id = str(result.error.details.get("approval_id", "")) if result.error else ""
        body = {
            "status": "pending_approval",
            "run_id": run_key,
            "call_id": result.call_id,
            "approval_id": approval_id,
        }
        headers = {"Location": f"/approvals/{approval_id}", "Retry-After": "2"}
        return json_response(202, body, headers)
    found = (
        problem_for(result.exception)
        if result.exception is not None
        else problem(
            result.error.kind if result.error else "internal-error",
            result.error.message if result.error else "",
        )
    )
    detail = found.detail.replace(ctx.run_id, run_key) if ctx.run_id else found.detail
    return problem_response(found.model_copy(update={"detail": detail}))


async def decide(
    state: State, principal: Principal, body: bytes, approval_id: str
) -> ServiceResponse:
    """Record a decision; HTTP calls then run (approved) or end (denied) right here."""
    parsed = decide_body(body)
    record = await visible_record(state, principal, approval_id)
    if not await state.may_decide(principal, record):
        logger.warning(
            "principal %s may not decide approval %s", principal_ref(principal.id), approval_id
        )
        refuse("forbidden", "not allowed to decide this approval", approval_id=approval_id)
    if record.status != "pending":
        raise ApprovalAlreadyDecided(approval_id, status=record.status)
    request = record.request
    ours = request.metadata.get("origin") == ORIGIN_HTTP and request.kind == "tool"
    spec = state.spec(request.tool) if ours else _registered(state, request.tool)
    if spec is None:
        _check_unvalidated(record, parsed.approved, parsed.edited_arguments)
    else:
        if parsed.edited_arguments is not None:
            spec.validate_args(parsed.edited_arguments)  # 422 before deciding
        if parsed.approved:
            await _authorize_approval(state, principal, record, spec, parsed.edited_arguments)
    decision = ApprovalDecision(
        approved=parsed.approved,
        edited_args=parsed.edited_arguments,
        reason=parsed.reason,
        decided_by=principal.id,
        decided_at=time.time(),
    )
    await state.store.decide(approval_id, decision)
    logger.info(
        "approval %s %s by %s",
        approval_id,
        "approved" if decision.approved else "denied",
        principal_ref(principal.id),
    )
    if not ours:  # another surface (e.g. MCP) parked it: its client resumes the call
        return _decided_elsewhere(record, decision)
    return await _resume(state, record, decision)


async def _authorize_approval(
    state: State,
    decider: Principal,
    record: ApprovalRecord,
    spec: ToolSpec,
    edited: dict[str, Any] | None,
) -> None:
    """Refuse (403, nothing recorded) unless the decider may run the edits and the invoker
    may still run the final arguments (permissions can change while a call waits)."""
    if edited is not None and not await state.allowed(decider, spec, edited):
        logger.warning(
            "principal %s may not run edited arguments of %s", principal_ref(decider.id), spec.name
        )
        refuse("forbidden", "not allowed to run the edited arguments", tool=spec.name)
    invoker = invoker_of(record)
    final = edited if edited is not None else record.request.arguments
    if invoker is None or not await state.allowed(invoker, spec, final):
        logger.warning("approval %s: the invoker may not run this call", record.approval_id)
        refuse("forbidden", "the invoker is not allowed to run this call", tool=spec.name)


def _check_unvalidated(
    record: ApprovalRecord, approved: bool, edited: dict[str, Any] | None
) -> None:
    """A tool this server cannot validate or `authorize`: no edits (400), and approving
    needs an invoker on the record (403); nothing is recorded either way."""
    tool = record.request.tool
    if edited is not None:
        logger.warning(
            "approval %s: edits refused for unregistered tool %s", record.approval_id, tool
        )
        refuse("bad-request", "edited arguments need a tool registered on this server", tool=tool)
    if approved and invoker_of(record) is None:
        logger.warning("approval %s: no invoker recorded, cannot approve", record.approval_id)
        refuse("forbidden", "the invoker is not allowed to run this call", tool=tool)


def _registered(state: State, name: str) -> ToolSpec | None:
    return state.registry.get(name) if name in state.registry else None


async def visible_record(state: State, principal: Principal, approval_id: str) -> ApprovalRecord:
    """The record, if `principal` may see it; otherwise not found (no existence oracle).

    Raises:
        ApprovalNotFound: unknown, or another principal's.
    """
    record = await state.store.get(approval_id)
    if not can_read(principal, record):
        logger.warning(
            "principal %s asked for another principal's approval", principal_ref(principal.id)
        )
        raise ApprovalNotFound(approval_id)
    return record


async def _resume(
    state: State, record: ApprovalRecord, decision: ApprovalDecision
) -> ServiceResponse:
    request = record.request
    metadata = request.metadata
    run_key = run_key_of(record)
    cache_key = str(metadata.get("idempotency", ""))
    sent = str(metadata.get("fingerprint", ""))
    ctx = state.executor.context(
        run_id=request.run_id,
        thread_id=str(metadata.get("thread_id", "")),
        extras={
            "principal": invoker_of(record) or Principal(""),
            CALL_EXTRAS: {
                "call_id": request.call_id,
                "run_key": run_key,
                "idempotency": cache_key,
                "fingerprint": sent,
            },
        },
    )
    call = ToolCall(name=request.tool, arguments=request.arguments, id=request.call_id)
    result = await state.executor.run_call(call, ctx, registry=state.tools())
    as_invoked = outcome(result, run_key, ctx)
    if cache_key and sent:  # a repeat of the invoke replays the decided outcome, not the 202
        await state.idempotency.update(cache_key, _stored(sent, as_invoked))
    denied = result.error is not None and result.error.kind == "approval-denied"
    if not decision.approved and denied:
        return json_response(200, _denied(record, run_key, decision.reason))
    return as_invoked


def _decided_elsewhere(record: ApprovalRecord, decision: ApprovalDecision) -> ServiceResponse:
    run_key = run_key_of(record)
    if not decision.approved:
        return json_response(200, _denied(record, run_key, decision.reason))
    body = {
        "status": "approved",
        "run_id": run_key,
        "call_id": record.request.call_id,
        "approval_id": record.approval_id,
        "note": ELSEWHERE_NOTE,
    }
    return json_response(202, body)


def _denied(record: ApprovalRecord, run_key: str, reason: str) -> dict[str, Any]:
    return {
        "status": "denied",
        "run_id": run_key,
        "call_id": record.request.call_id,
        "approval_id": record.approval_id,
        "reason": reason,
    }


def _stored(sent: str, response: ServiceResponse) -> Stored:
    return Stored(
        fingerprint=sent,
        status=response.status,
        body=response.body,
        media_type=response.media_type,
        headers=dict(response.headers),
    )


def _replay(stored: Stored) -> ServiceResponse:
    return ServiceResponse(
        status=stored.status,
        body=stored.body,
        media_type=stored.media_type,
        headers={**stored.headers, REPLAYED: "true"},
    )


_UNENCODABLE = object()


def _json_value(value: Any) -> Any:
    try:
        jsonable = to_jsonable(value)
        json.dumps(jsonable, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return _UNENCODABLE
    return jsonable
