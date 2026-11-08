"""Approval memory: similar past decisions shown next to a new request; it never decides."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.hooks.builtin.logging import is_secret_field

MAX_SCANNED = 5_000  # decided records compared for one request
MAX_DEPTH = 3  # nested argument levels compared ("a.b.c")
DEFAULT_SHOWN = 5
DEFAULT_MIN_SIMILARITY = 0.5
SAME_SHAPE = 0.5  # weight of a field both calls have, with different values

_FROZEN = ConfigDict(frozen=True)
_SEP = "\x1f"  # joins nested keys: a key may itself hold a dot


class SimilarDecision(BaseModel):
    """One earlier decision on a similar request; `matched` names equal fields (nested: `a.*`)."""

    model_config = _FROZEN

    approval_id: str
    approved: bool
    edited: bool = False
    decided_by: str = ""
    decided_at: float = 0.0
    similarity: float
    matched: tuple[str, ...] = ()


class ApprovalContext(BaseModel):
    """What reviewers decided before on requests like this one: advice, never a decision."""

    model_config = _FROZEN

    tool: str
    similar: int = 0
    approved: int = 0
    denied: int = 0
    deciders: tuple[str, ...] = ()
    last_decided_at: float | None = None
    examples: tuple[SimilarDecision, ...] = ()

    def summary(self) -> str:
        """One line for an approval card, such as "3 similar: 2 approved, 1 denied (by ann)"."""
        if not self.similar:
            return "no similar decisions yet"
        who = f" (by {', '.join(self.deciders)})" if self.deciders else ""
        return f"{self.similar} similar: {self.approved} approved, {self.denied} denied{who}"

    def view(self) -> dict[str, Any]:
        """The context as JSON for a page or an API, with its `summary` line."""
        return {**self.model_dump(mode="json"), "summary": self.summary()}


def comparable_fields(arguments: Mapping[str, Any]) -> dict[str, str]:
    """Non-secret argument paths (keys joined by `_SEP`) to canonical JSON; secrets left out."""
    return dict(_walk(arguments, "", 1))


def _walk(value: Mapping[str, Any], prefix: str, depth: int) -> Iterator[tuple[str, str]]:
    for key, item in value.items():
        if is_secret_field(key, item):
            continue  # neither compared nor named
        path = f"{prefix}{key}"
        if isinstance(item, Mapping) and item and depth < MAX_DEPTH:
            yield from _walk(item, f"{path}{_SEP}", depth + 1)
        else:
            yield path, _canonical(item)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def argument_similarity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[float, tuple[str, ...]]:
    """0..1 over non-secret fields (equal 1, same field 0.5, one-sided 0); the equal field names."""
    a, b = comparable_fields(left), comparable_fields(right)
    names = a.keys() | b.keys()
    if not names:
        return 1.0, ()
    equal = [name for name in a.keys() & b.keys() if a[name] == b[name]]
    shared = len(a.keys() & b.keys())
    score = (len(equal) + SAME_SHAPE * (shared - len(equal))) / len(names)
    return round(score, 4), tuple(sorted({_shown(name) for name in equal}))


def _shown(path: str) -> str:
    """A field as it may be shown: nested keys can be data (an address), so they become `*`."""
    top, _, rest = path.partition(_SEP)
    return f"{top}.*" if rest else top


def decided(records: Iterable[ApprovalRecord]) -> list[ApprovalRecord]:
    """The records a person decided (approved or denied, with a decision), newest first."""
    kept = [r for r in records if r.status in ("approved", "denied") and r.decision is not None]
    return sorted(kept, key=_decided_at, reverse=True)


def _decided_at(record: ApprovalRecord) -> float:
    decision = record.decision
    return decision.decided_at if decision is not None else 0.0


def approval_context(
    request: ApprovalRequest,
    records: Iterable[ApprovalRecord],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    shown: int = DEFAULT_SHOWN,
) -> ApprovalContext:
    """Earlier decisions on the same tool with similar non-secret arguments, best first.

    Only reads the records: the request still waits for its own decision.

    Raises:
        ConfigError: `min_similarity` outside 0..1 or `shown` below 0.
    """
    if not 0.0 <= min_similarity <= 1.0:
        raise ConfigError(f"min_similarity must be within 0..1, got {min_similarity!r}")
    if shown < 0:
        raise ConfigError(f"shown must be >= 0, got {shown!r}")
    found: list[SimilarDecision] = []
    for record in decided(records)[:MAX_SCANNED]:
        past, decision = record.request, record.decision
        same = (past.kind, past.tool) == (request.kind, request.tool)
        if decision is None or not same or past.approval_id == request.approval_id:
            continue
        score, matched = argument_similarity(request.arguments, past.arguments)
        if score >= min_similarity:
            found.append(_similar(record.approval_id, decision, score, matched))
    found.sort(key=lambda item: (item.similarity, item.decided_at), reverse=True)
    approved = sum(1 for item in found if item.approved)
    return ApprovalContext(
        tool=request.tool,
        similar=len(found),
        approved=approved,
        denied=len(found) - approved,
        deciders=tuple(sorted({item.decided_by for item in found if item.decided_by})),
        last_decided_at=max((item.decided_at for item in found), default=None),
        examples=tuple(found[:shown]),
    )


def _similar(
    approval_id: str, decision: ApprovalDecision, score: float, matched: tuple[str, ...]
) -> SimilarDecision:
    return SimilarDecision(
        approval_id=approval_id,
        approved=decision.approved,
        edited=decision.edited_args is not None,
        decided_by=decision.decided_by,
        decided_at=decision.decided_at,
        similarity=score,
        matched=matched,
    )
