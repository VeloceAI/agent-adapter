"""Shared pieces of the durable approval stores: record codec and row checks."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_adapter.approval._records import check_list as check_list
from agent_adapter.approval._records import filled as filled
from agent_adapter.approval._records import stamped as stamped
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ApprovalStoreCorrupt as ApprovalStoreCorrupt
from agent_adapter.core.errors import ToolUsageError, log_failure

logger = logging.getLogger(__name__)

_SHOWN = 64  # characters of a stored id quoted in errors and logs


def key(text: str) -> bytes:
    """The stored bytes of an id; lone surrogates survive (as in `request_digest`)."""
    return text.encode("utf-8", "surrogatepass")


def encode(record: ApprovalRecord) -> str:
    """ASCII JSON of `record`; `decode` reads it back equal (metadata becomes plain JSON).

    Raises:
        ToolUsageError: the record holds a value JSON cannot carry (e.g. an object in metadata).
    """
    try:
        data = record.model_dump(mode="json")
        return json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        log_failure(
            logger, logging.WARNING, exc, "approval %s cannot be stored", record.approval_id
        )
        raise ToolUsageError(
            f"approval {record.approval_id!r} cannot be stored: metadata is not JSON"
        ) from exc


def decode(raw: object, where: str) -> ApprovalRecord:
    """The record in a stored JSON text.

    Raises:
        ApprovalStoreCorrupt: not JSON, not an `ApprovalRecord`, or a digest that disagrees.
    """
    try:
        if not isinstance(raw, str | bytes):
            raise TypeError(f"stored record is {type(raw).__name__}, not text")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError(f"stored record is a JSON {type(data).__name__}, not an object")
        return ApprovalRecord.model_validate(data)
    except (ValueError, TypeError, RecursionError) as exc:
        log_failure(logger, logging.ERROR, exc, "approval store row %s is unreadable", where)
        raise ApprovalStoreCorrupt(f"approval store row {where} is unreadable") from exc


def read_row(
    raw: object,
    *,
    approval_id: object,
    run_id: object,
    call_id: object,
    status: object,
    consumed: object,
) -> ApprovalRecord:
    """Decode a row and check that its indexed columns agree with its record.

    Raises:
        ApprovalStoreCorrupt: an unreadable record, or columns that disagree with it.
    """
    where = shown(approval_id)
    record = decode(raw, where)
    columns = (text(approval_id), text(run_id), text(call_id), text(status), flag(consumed))
    request = record.request
    own = (request.approval_id, request.run_id, request.call_id, record.status, record.consumed)
    if columns != own:
        logger.error("approval store row %s disagrees with its record: failing closed", where)
        raise ApprovalStoreCorrupt(f"approval store row {where} disagrees with its record")
    return record


def text(value: object) -> str | None:
    """A stored id or status as text; None when it is neither text nor UTF-8 bytes."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            return None
    return None


def flag(value: object) -> bool | None:
    """A stored consumed flag (0 / 1, as int, text or bytes); None for anything else."""
    if isinstance(value, bytes | str):
        value = text(value)
        return {"0": False, "1": True}.get(value) if isinstance(value, str) else None
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return value == 1
    return None


def shown(value: Any) -> str:
    """A stored id quoted for errors and logs, cut to a bounded length."""
    found = text(value)
    if found is None:
        return f"<{type(value).__name__}>"
    return repr(found[:_SHOWN])
