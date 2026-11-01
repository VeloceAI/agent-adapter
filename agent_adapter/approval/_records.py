"""Record rules every approval store shares: the JSON check, ids and times, `list` arguments."""

from __future__ import annotations

import math
import time
from typing import Any

from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime.ids import new_id

STATUSES: frozenset[str] = frozenset({"pending", "approved", "denied"})
MAX_JSON_DEPTH = 256  # nested lists/objects in approval arguments, the arguments object = 1


def filled(request: ApprovalRequest) -> ApprovalRequest:
    """`request` with an empty `approval_id` and `created_at` filled in."""
    updates: dict[str, object] = {}
    if not request.approval_id:
        updates["approval_id"] = new_id("apr_")
    if not request.created_at:
        updates["created_at"] = time.time()
    return request.model_copy(update=updates) if updates else request


def stamped(decision: ApprovalDecision) -> ApprovalDecision:
    """`decision` with an empty `decided_at` filled in.

    Raises:
        ToolUsageError: not an ApprovalDecision, or `edited_args` is not plain JSON (it is
            what runs, so never coerced).
    """
    if not isinstance(decision, ApprovalDecision):
        raise ToolUsageError(
            f"decide needs an ApprovalDecision, got {type(decision).__name__}: "
            f"store.decide(id, ApprovalDecision(approved={bool(decision)!r})), "
            "or executor.resume(id, True, decided_by='you') to decide and run in one call"
        )
    if decision.edited_args is not None:
        check_json(decision.edited_args)
    if decision.decided_at:
        return decision
    return decision.model_copy(update={"decided_at": time.time()})


def check_list(status: str | None, limit: int) -> None:
    """The `list` argument rules every store shares.

    Raises:
        ConfigError: unknown `status` or a negative `limit`.
    """
    if status is not None and status not in STATUSES:
        raise ConfigError(f"status must be one of {sorted(STATUSES)}, got {status!r}")
    if limit < 0:
        raise ConfigError(f"limit must be >= 0, got {limit}")


def check_json(value: Any) -> None:
    """Fail closed unless `value` is plain JSON (the same rule as `request_digest`).

    Raises:
        ToolUsageError: bytes, tuple, set, other objects, non-string keys, NaN, infinity,
            or nesting deeper than `MAX_JSON_DEPTH`.
    """
    _check_json(value, "$")


def _check_json(value: Any, path: str) -> None:
    # json.dumps would silently turn tuples into lists and int keys into strings, so two
    # different calls could share a digest; only exact JSON types are accepted. Iterative,
    # with a depth cap, so hostile nesting fails closed instead of raising RecursionError.
    stack: list[tuple[Any, str, int]] = [(value, path, 1)]
    while stack:
        item, where, depth = stack.pop()
        if item is None or isinstance(item, str | bool | int):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise _not_json(where, "non-finite float")
            continue
        if not isinstance(item, list | dict):
            raise _not_json(where, type(item).__name__)
        if depth > MAX_JSON_DEPTH:
            raise ToolUsageError(f"approval arguments nested too deeply (over {MAX_JSON_DEPTH})")
        if isinstance(item, list):
            children = [(child, f"{where}[{i}]", depth + 1) for i, child in enumerate(item)]
        else:
            children = []
            for key, child in item.items():
                if not isinstance(key, str):
                    raise _not_json(where, f"non-string key {type(key).__name__}")
                children.append((child, f"{where}.{key}", depth + 1))
        stack.extend(reversed(children))  # depth-first, in document order


def _not_json(path: str, what: str) -> ToolUsageError:
    # Only the location and type: the value may be sensitive.
    return ToolUsageError(f"approval arguments are not JSON at {path}: {what}")
