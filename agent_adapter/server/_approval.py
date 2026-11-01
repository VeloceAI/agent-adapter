"""The server's approval port: `DeferredApproval` that records who asked, for review and resume."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.runtime.current import current_context
from agent_adapter.server.auth import Principal

logger = logging.getLogger(__name__)

CALL_EXTRAS = "aa.http"  # ctx.extras key: what the HTTP server knows about this call
ORIGIN_HTTP = "http"


class ServerApproval(DeferredApproval):
    """Parks calls in the store with `metadata` naming the invoker (never trusted from clients).

    Metadata: `principal` and `principal_scopes` (or `anonymous: True`), `thread_id`, and for
    the HTTP call itself (not calls nested in it) `origin="http"`, `run_key`, `idempotency`
    and `fingerprint` (its replay entry). Claims are never stored. It is set only
    when the current context is the call being approved, so a record never names the
    wrong invoker; the digest the decision binds to ignores metadata.
    """

    owns_server_records = True  # the server re-authorizes the invoker before it resumes

    async def prepare(self, request: ApprovalRequest) -> ApprovalRecord:
        return await super().prepare(with_metadata(request))


def with_metadata(request: ApprovalRequest) -> ApprovalRequest:
    """`request` with the invoker's metadata from the current context (if it is this call)."""
    ctx = current_context()
    if ctx is None or (ctx.run_id, ctx.call_id) != (request.run_id, request.call_id):
        logger.debug("approval request outside its call context: no invoker recorded")
        return request
    metadata = _invoker(ctx.extras.get("principal"))
    if ctx.thread_id:
        metadata["thread_id"] = ctx.thread_id
    call = ctx.extras.get(CALL_EXTRAS)
    # Only the HTTP call itself is resumed by `decide`; a nested call (its child context
    # copies the extras) is decided "elsewhere" and must not claim the outer replay key.
    if isinstance(call, Mapping) and call.get("call_id") == request.call_id:
        metadata.update(
            origin=ORIGIN_HTTP,
            run_key=str(call.get("run_key", "")),
            idempotency=str(call.get("idempotency", "")),
            fingerprint=str(call.get("fingerprint", "")),
        )
    return request.model_copy(update={"metadata": {**request.metadata, **metadata}})


def _invoker(principal: Any) -> dict[str, Any]:
    """`principal` + sorted `principal_scopes`, `anonymous: True` for the anonymous
    `Principal("")`, or nothing (no invoker: approvers only, and it cannot run)."""
    if isinstance(principal, Principal):
        if not principal.id:
            return {"anonymous": True}
        return {"principal": principal.id, "principal_scopes": sorted(principal.scopes)}
    if isinstance(principal, str) and principal:
        return {"principal": principal, "principal_scopes": []}
    return {}
