"""Read routes scoped to the caller: approvals, run events (SSE) and run cost."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from agent_adapter.approval.store import STATUSES, ApprovalRecord
from agent_adapter.cost.summary import summarise
from agent_adapter.server import _sse
from agent_adapter.server._calls import visible_record
from agent_adapter.server._payloads import approval_payload, client_key, csv, refuse, run_key_of
from agent_adapter.server._runs import owner_key, run_id_for
from agent_adapter.server._state import (
    ServiceResponse,
    State,
    can_read,
    forbid_other_owner,
    json_response,
)
from agent_adapter.server.auth import APPROVER_SCOPE, Principal, RequestInfo, principal_ref

logger = logging.getLogger(__name__)

SCAN_LIMIT = 1_000  # records read to answer one listing that filters by owner or run key
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def list_approvals(
    state: State, principal: Principal, request: RequestInfo
) -> ServiceResponse:
    """The approvals of runs `principal` owns (every run for `APPROVER_SCOPE`), newest first."""
    status = request.query.get("status") or None
    if status is not None and status not in STATUSES:
        refuse("bad-request", "status must be pending, approved or denied")
    run_key = client_key(request.query.get("run_id"), "run_id")
    approver = APPROVER_SCOPE in principal.scopes
    if run_key and not approver:
        forbid_other_owner(state, principal, run_key)
        run_id = run_id_for(owner_key(principal), run_key)
        records = await state.store.list(run_id=run_id, status=status, limit=state.max_listed)
    else:
        records = await state.store.list(status=status, limit=SCAN_LIMIT)
        if run_key:
            records = [r for r in records if run_key_of(r) == run_key]
    shown = [r for r in records if can_read(principal, r)][: state.max_listed]
    return json_response(200, {"approvals": [approval_payload(r) for r in shown]})


async def get_approval(state: State, principal: Principal, approval_id: str) -> ServiceResponse:
    record: ApprovalRecord = await visible_record(state, principal, approval_id)
    return json_response(200, approval_payload(record))


async def run_events(
    state: State,
    principal: Principal,
    request: RequestInfo,
    run_key: str,
    *,
    keep_alive_s: float,
    idle_s: float,
    per_principal: int,
    total: int,
    clock: Callable[[], float] = time.monotonic,
) -> ServiceResponse:
    """SSE of the caller's own run; replay from `Last-Event-ID` or `?after=`."""
    run_key = _path_run(run_key)
    forbid_other_owner(state, principal, run_key)
    after = _after(request.header("last-event-id") or request.query.get("after"))
    close_on = frozenset(csv(request.query.get("close_on")) or ())
    release = state.streams.acquire(principal.id, per_principal=per_principal, total=total)
    if release is None:
        logger.warning("principal %s refused: too many event streams", principal_ref(principal.id))
        refuse("too-many-streams", "too many open event streams; close one and retry")
    source = state.events.subscribe(run_id_for(owner_key(principal), run_key), after=after)
    body = _sse.stream(
        source,
        run_key=run_key,
        close_on=close_on,
        keep_alive_s=keep_alive_s,
        idle_s=idle_s,
        clock=clock,
    )
    return ServiceResponse(
        status=200,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        stream=_sse.Guarded(body, release),
    )


async def run_cost(state: State, principal: Principal, run_key: str) -> ServiceResponse:
    """The ledger summary of the caller's own run (404 when no ledger is configured)."""
    run_key = _path_run(run_key)
    forbid_other_owner(state, principal, run_key)
    if state.ledger is None:
        refuse("not-found", "no cost ledger is configured")
    run_id = run_id_for(owner_key(principal), run_key)
    summary = summarise(state.ledger.entries(run_id), run_id)
    return json_response(200, {**summary.model_dump(mode="json"), "run_id": run_key})


def _path_run(value: str) -> str:
    run_key = client_key(value, "run_id")
    if not run_key:
        refuse("bad-request", "run_id is required")
    return run_key


def _after(value: str | None) -> int:
    if value is None or value == "":
        return 0
    if not (value.isascii() and value.isdigit()) or len(value) > 18:
        refuse("bad-request", "Last-Event-ID / after must be a non-negative integer")
    return int(value)
