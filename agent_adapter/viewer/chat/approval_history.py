"""Chat approval memory: earlier chat decisions as records, for the approval card context."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_adapter.approval.memory import approval_context
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)

MAX_HISTORY = 5_000  # stored decisions read for one approval
History = Callable[[], Iterable[Mapping[str, Any]]]


def record_of(view: Mapping[str, Any]) -> ApprovalRecord | None:
    """A decided chat approval (its stored view) as a record; None unless a person decided it.

    Arguments are the stored preview (secret-named values already masked); a cut preview
    compares by tool only.
    """
    status = view.get("status")
    decided_by = view.get("decided_by")
    if status not in ("approved", "denied") or not isinstance(decided_by, str) or not decided_by:
        return None
    try:
        created = _number(view.get("created_at"))
        request = ApprovalRequest(
            approval_id=str(view.get("approval_id", "")),
            run_id="",
            call_id=str(view.get("call_id", "")),
            tool=str(view.get("tool", "")),
            arguments=arguments_of(view.get("arguments")),
            created_at=created,
        )
        decision = ApprovalDecision(
            approved=status == "approved",
            decided_by=decided_by,
            decided_at=_number(view.get("decided_at")) or created,
        )
        return ApprovalRecord(request=request, status=status, decision=decision)
    except (ToolUsageError, ValueError) as exc:
        logger.debug("chat: skipped a stored approval for memory (%s)", type(exc).__name__)
        return None


def arguments_of(preview: object) -> dict[str, Any]:
    """The arguments object a stored preview holds, else {}."""
    if isinstance(preview, dict):
        return preview
    if not isinstance(preview, str):
        return {}
    try:
        parsed = json.loads(preview)
    except (ValueError, RecursionError):
        return {}  # cut to the preview size, or not JSON
    return parsed if isinstance(parsed, dict) else {}


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def context_view(request: ApprovalRequest, views: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The approval context of `request` over stored approval views, as JSON for the card."""
    seen: set[str] = set()
    records: list[ApprovalRecord] = []
    for view in views:
        record = record_of(view)
        if record is None or record.approval_id in seen:
            continue
        seen.add(record.approval_id)
        records.append(record)
        if len(records) >= MAX_HISTORY:
            break
    return approval_context(request, records).view()
