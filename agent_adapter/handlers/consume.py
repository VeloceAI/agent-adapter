"""`handle_envelope`: run one queue message or event through the executor, framework-free."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

from agent_adapter.approval.store import ApprovalStore
from agent_adapter.core.errors import (
    AdapterError,
    ApprovalPending,
    ToolNotFound,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.types import to_jsonable
from agent_adapter.exec._approve import approval_store
from agent_adapter.exec._sugar import check_resumable
from agent_adapter.exec.executor import Executor
from agent_adapter.handlers.envelope import Envelope, parse_envelope

logger = logging.getLogger(__name__)

Reply = dict[str, Any]


async def handle_envelope(
    executor: Executor,
    payload: Mapping[str, Any] | str | bytes | Envelope,
    *,
    tools: Collection[str] | None = None,
    message_id: str = "",
    source: str = "",
) -> Reply:
    """Run one call (or resume one decided approval) and return a JSON-safe reply.

    Replies: `{"status": "ok", "result", ...}`, `{"status": "pending_approval",
    "approval_id", ...}` or, for a redelivered resume whose decision was already used,
    `{"status": "already_done", "approval_id", ...}`; all with `tool`, `run_id`, `call_id`.
    `message_id` (the source's delivery id, stable across redeliveries) names the call when
    the envelope has no ids; `source` keeps ids of different sources apart. `tools` limits
    what runs.

    Raises:
        ToolUsageError: a malformed envelope, or a call or resume of an approval the HTTP
            tool server (or a flow) owns.
        ToolNotFound: the tool is unknown or not in `tools`.
        ApprovalNotFound: a resume of a missing approval.
        AdapterError: any other failure of the call (ApprovalDenied, ToolFailed, ...).
    """
    try:
        envelope = (
            payload
            if isinstance(payload, Envelope)
            else parse_envelope(payload, message_id=message_id, source=source)
        )
        if envelope.is_resume:
            resumed = await _resumable(executor, envelope.approval_id)
            if isinstance(resumed, dict):
                return resumed
            envelope = resumed
        else:
            await _check_ids(executor, envelope)
        return await _run(executor, envelope, tools)
    except Exception as exc:
        level = logging.WARNING if isinstance(exc, AdapterError) else logging.ERROR
        log_failure(logger, level, exc, "envelope failed (message %r)", message_id)
        raise


async def _run(executor: Executor, envelope: Envelope, tools: Collection[str] | None) -> Reply:
    if tools is not None and envelope.tool not in tools:
        raise ToolNotFound(f"tool {envelope.tool!r} is not served by this handler")
    ids = {"tool": envelope.tool, "run_id": envelope.run_id, "call_id": envelope.call_id}
    try:
        async with executor.run_scope(run_id=envelope.run_id) as ctx:
            value = await executor.run(
                envelope.tool, envelope.arguments, ctx, call_id=envelope.call_id
            )
    except ApprovalPending as pending:  # parked: ack the message, a resume runs it later
        logger.info("envelope parked: approval %s (%s)", pending.approval_id, _ids(ids))
        return {"status": "pending_approval", "approval_id": pending.approval_id, **ids}
    logger.info("envelope ok (%s)", _ids(ids))
    return {"status": "ok", **ids, "result": to_jsonable(value)}


async def _resumable(executor: Executor, approval_id: str) -> Envelope | Reply:
    """The decided call `approval_id` names, or a reply when it is pending or already used."""
    record = await _store(executor, "a resume envelope").get(approval_id)  # ApprovalNotFound
    check_resumable(record)
    request = record.request
    ids = {"tool": request.tool, "run_id": request.run_id, "call_id": request.call_id}
    if record.consumed:  # a redelivered resume: its decision already ran (at least once)
        logger.info("resume of approval %s: already done (%s)", approval_id, _ids(ids))
        return {"status": "already_done", "approval_id": approval_id, **ids}
    if record.status == "pending":
        logger.info("resume of approval %s: still pending", approval_id)
        return {"status": "pending_approval", "approval_id": approval_id, **ids}
    # The original arguments: the decision is bound to them; the executor applies edits.
    return Envelope(
        tool=request.tool,
        arguments=dict(request.arguments),
        run_id=request.run_id,
        call_id=request.call_id,
    )


async def _check_ids(executor: Executor, envelope: Envelope) -> None:
    """Refuse a call envelope whose ids name an open record another surface owns."""
    store = approval_store(executor.approval)
    if store is None:
        return
    record = await store.find(envelope.run_id, envelope.call_id)
    if record is not None:
        check_resumable(record)


def _store(executor: Executor, what: str) -> ApprovalStore:
    store = approval_store(executor.approval)
    if store is None:
        raise ToolUsageError(f"{what} needs an executor with DeferredApproval(store)")
    return store


def _ids(ids: Mapping[str, str]) -> str:
    return ", ".join(f"{key} {value}" for key, value in ids.items())
