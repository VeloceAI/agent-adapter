"""The viewer's approval desk: list pending records and record a reviewer's decision."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from agent_adapter.approval.deferred import server_owned
from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ApprovalAlreadyDecided, ConfigError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.hooks.builtin.logging import redact_secrets

logger = logging.getLogger(__name__)

MAX_REASON = 500
MAX_REVIEWER = 120
MAX_LISTED = 200


class ApprovalDesk:
    """Decides tool approvals in a `SQLiteApprovalStore` file; it never runs a tool.

    The app's own `executor.resume(approval_id)` later runs the call with this decision.

    Raises:
        ConfigError: no database at `path`, or an empty or too long `reviewer`.
    """

    def __init__(self, path: str | os.PathLike[str], reviewer: str) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ConfigError(f"no approvals database at {self.path}")
        name = str(reviewer or "").strip()
        if not name or len(name) > MAX_REVIEWER:
            raise ConfigError(f"--reviewer must name the reviewer (1-{MAX_REVIEWER} characters)")
        self.reviewer = name
        self.store = SQLiteApprovalStore(self.path)

    def pending(self) -> list[dict[str, Any]]:
        """Pending records, newest first, with secret-named arguments masked."""
        records = asyncio.run(self.store.list(status="pending", limit=MAX_LISTED))
        return [view(record) for record in records]

    def decide(self, approval_id: str, *, approved: bool, reason: str = "") -> dict[str, Any]:
        """Record approve or deny as this desk's reviewer; only a pending tool record.

        Raises:
            ApprovalNotFound: unknown id.
            ApprovalAlreadyDecided: the record is no longer pending (here or by a race).
            ToolUsageError: a flow-node record, or one the HTTP tool server parked.
        """
        record = asyncio.run(self.store.get(approval_id))
        if record.status != "pending":
            raise ApprovalAlreadyDecided(approval_id, status=record.status)
        refusal = _refusal(record)
        if refusal:
            raise ToolUsageError(refusal)
        decision = ApprovalDecision(
            approved=approved, reason=str(reason)[:MAX_REASON], decided_by=self.reviewer
        )
        updated = asyncio.run(self.store.decide(approval_id, decision))
        logger.info(
            "viewer: approval %s %s by %s",
            approval_id,
            "approved" if approved else "denied",
            self.reviewer,
        )
        return view(updated)


def view(record: ApprovalRecord) -> dict[str, Any]:
    """What the page shows of a record."""
    request = record.request
    refusal = _refusal(record)
    return {
        "approval_id": request.approval_id,
        "run_id": request.run_id,
        "call_id": request.call_id,
        "tool": request.tool,
        "kind": request.kind,
        "hint": request.hint,
        "arguments": redact_secrets(request.arguments),
        "created_at": request.created_at,
        "status": record.status,
        "decided_by": record.decision.decided_by if record.decision else "",
        "decidable": record.status == "pending" and not refusal,
        "why_not": refusal,
    }


def _refusal(record: ApprovalRecord) -> str:
    if record.request.kind != "tool":
        return "a flow-node approval: resume it with FlowRunner.resume"
    if server_owned(record):
        return "parked by the HTTP tool server: decide it at its /approvals endpoint"
    return ""
