"""Pipeline step 5: pre-approval, the confirm rule, the approval port and edited args."""

from __future__ import annotations

import dataclasses
import logging
import time
import weakref
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel

from agent_adapter.approval.builtin import evaluate_confirm
from agent_adapter.approval.store import ApprovalStore, request_digest
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalDenied,
    ApprovalNotFound,
    ApprovalPending,
)
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, ToolCall
from agent_adapter.exec._resume import record_pending, resumed_answer
from agent_adapter.exec.args import dump_args
from agent_adapter.hooks.base import ToolInvocation
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, require_recorded

logger = logging.getLogger(__name__)

NO_PORT = "no approval port configured"
# Fix-it notes: shown in tracebacks, never in the model-visible message.
ADD_A_PORT = (
    "pass approval= to Executor (or agent_adapter.configure): AutoApprove() for tests, "
    "DeferredApproval(store) for reviews, CallbackApproval(fn) to ask a function"
)
HOW_TO_RESUME = (
    "decide and run it with executor.resume(approval_id, True, decided_by='you'), or "
    "decide it with "
    "store.decide(approval_id, ApprovalDecision(...)) and then call executor.resume("
    "approval_id); the approval id is on this error's .approval_id"
)
# the store of the port that raised each pending (kept off the exception, which must pickle)
_STORES: weakref.WeakKeyDictionary[BaseException, ApprovalStore | None] = (
    weakref.WeakKeyDictionary()
)

Emit = Callable[..., None]


async def approve(
    inv: ToolInvocation,
    validated: BaseModel,
    *,
    pre_approval: ApprovalDecision | None,
    port: ApprovalPort | None,
    emit: Emit,
    approval_id: str = "",
) -> tuple[BaseModel, ToolInvocation]:
    """Return the (possibly edited and re-validated) arguments the call may run with.

    Args:
        pre_approval: The caller's decision for exactly this call (`ctx.approval`), if any.
        approval_id: The pending approval this call resumes ("" for a first request).

    Raises:
        ApprovalDenied: denied, or approval needed with no port configured.
        ApprovalPending: the port (or the confirm rule) defers the decision.
        ToolUsageError: the confirm rule returned a non-bool, or the port refused the request
            (e.g. a reused call id with other arguments); the tool does not run.
        ToolArgsInvalid: edited arguments fail validation.
        AdapterError: a fail-closed sink (`require_recorded`) could not record; nothing ran.
    """
    spec, ctx = inv.spec, inv.ctx
    decision = pre_approval
    if decision is None:
        if not await evaluate_confirm(spec.confirm, inv.call, ctx):
            await require_recorded(ctx.sink)  # a fail-closed audit log that failed stops the call
            return validated, inv
        decision = await _ask(inv, port, emit, approval_id)
    await require_recorded(ctx.sink)
    if not decision.approved:
        logger.info("tool %s denied (run %s, call %s)", spec.name, ctx.run_id, ctx.call_id)
        raise ApprovalDenied(spec.name, call_id=ctx.call_id, reason=decision.reason)
    if decision.edited_args is None:
        return validated, inv
    edited = spec.validate_args(decision.edited_args)
    call = ToolCall(name=spec.name, arguments=dump_args(edited), id=ctx.call_id)
    logger.debug("tool %s runs with edited arguments (call %s)", spec.name, ctx.call_id)
    return edited, dataclasses.replace(inv, call=call)


async def _ask(
    inv: ToolInvocation, port: ApprovalPort | None, emit: Emit, resumed: str
) -> ApprovalDecision:
    spec, ctx = inv.spec, inv.ctx
    request = ApprovalRequest(
        approval_id=resumed or new_id("apr_"),
        run_id=ctx.run_id,
        call_id=ctx.call_id,
        tool=spec.name,
        arguments=inv.call.arguments,
        created_at=time.time(),
    )
    answer = resumed_answer(request)
    decision = answer[1] if answer is not None else None
    if answer is not None:
        await _settle_own(approval_store(port), *answer)
    if decision is None and port is None:
        logger.warning(
            "tool %s needs approval but %s; denied (run %s)", spec.name, NO_PORT, ctx.run_id
        )
        denied = ApprovalDenied(spec.name, call_id=ctx.call_id, reason=NO_PORT)
        denied.add_note(ADD_A_PORT)
        raise denied
    if not resumed:  # a resumed call was announced when it first went pending
        emit(EventType.APPROVAL_REQUESTED, tool=spec.name, approval_id=request.approval_id)
        await require_recorded(ctx.sink)  # never ask a person for a decision the log cannot keep
    if decision is None:
        try:
            decision = await cast(ApprovalPort, port).request(request)  # not None: checked above
        except ApprovalPending as exc:
            _describe_pending(exc, request)
            store = approval_store(port)
            _STORES[exc] = store  # where a flow can withdraw the record
            record_pending(request.model_copy(update={"approval_id": exc.approval_id}), store)
            raise
    emit(
        EventType.APPROVAL_DECIDED,
        tool=spec.name,
        approval_id=request.approval_id,
        approved=decision.approved,
        edited=decision.edited_args is not None,
    )
    return decision


def approval_store(port: ApprovalPort | None) -> ApprovalStore | None:
    """The approval store behind a port (`port.store`), if it has one."""
    store = getattr(port, "store", None)
    return store if isinstance(store, ApprovalStore) else None


def pending_store(exc: BaseException, default: ApprovalStore | None) -> ApprovalStore | None:
    """The store of the port that raised `exc` (or a pending it caused), else `default`."""
    seen: BaseException | None = exc
    visited: set[int] = set()
    while seen is not None and id(seen) not in visited:
        if seen in _STORES:
            return _STORES[seen]
        visited.add(id(seen))
        seen = seen.__cause__
    return default


def same_decision(stored: ApprovalDecision | None, decision: ApprovalDecision) -> bool:
    """True when a stored decision approves (or denies) the same way with the same edits."""
    return stored is not None and (stored.approved, stored.edited_args) == (
        decision.approved,
        decision.edited_args,
    )


async def _settle_own(
    store: ApprovalStore | None, answered: ApprovalRequest, decision: ApprovalDecision
) -> None:
    """Decide and consume a resumed answer's record in this executor's own store, if it is there.

    Raises:
        ApprovalAlreadyDecided: the store holds another decision for it.
    """
    if store is None:
        return
    try:
        record = await store.get(answered.approval_id)
    except ApprovalNotFound:
        return  # another port's record (e.g. the flow runner's): settled there
    stored = record.request
    if (stored.run_id, stored.call_id) != (answered.run_id, answered.call_id) or request_digest(
        stored
    ) != request_digest(answered):
        logger.debug("approval %s in this store is for another call", answered.approval_id)
        return
    if record.status == "pending":
        try:
            record = await store.decide(answered.approval_id, decision)
        except ApprovalAlreadyDecided:
            record = await store.get(answered.approval_id)
    if not same_decision(record.decision, decision):
        logger.warning("approval %s was decided otherwise in the store", answered.approval_id)
        raise ApprovalAlreadyDecided(answered.approval_id, status=record.status)
    if not record.consumed and not await store.consume(answered.approval_id):
        logger.warning("approval %s was consumed by another caller", answered.approval_id)
        raise ApprovalAlreadyDecided(answered.approval_id, status="consumed")


def _describe_pending(exc: ApprovalPending, request: ApprovalRequest) -> None:
    """Fill what the port left out, so a resume can bind to the exact request."""
    exc.tool = exc.tool or request.tool
    exc.run_id = exc.run_id or request.run_id
    exc.call_id = exc.call_id or request.call_id
    if exc.arguments is None:
        exc.arguments = dict(request.arguments)
    add_note_once(exc, HOW_TO_RESUME)


def add_note_once(exc: BaseException, note: str) -> None:
    """Add a traceback note unless `exc` already carries it (it may be re-raised)."""
    if note not in getattr(exc, "__notes__", ()):
        exc.add_note(note)
