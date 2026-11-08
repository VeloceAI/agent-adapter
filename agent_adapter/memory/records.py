"""Memory values (`MemoryScope`, `MemoryRecord`, `MemoryHit`) and the checks backends share."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_adapter.core.errors import ConfigError, ToolUsageError

MAX_TEXT_CHARS = 20_000
MAX_BATCH = 1_000
MAX_METADATA_CHARS = 16_000
MAX_K = 100
MAX_ID_CHARS = 256
_FILTER_KEY = re.compile(r"[A-Za-z0-9_.-]{1,64}")

ScopeId = Annotated[str, Field(max_length=MAX_ID_CHARS)]
FilterValue = str | int | float | bool
Filters = Mapping[str, FilterValue]


class MemoryScope(BaseModel):
    """Whose memory: a user, a session (chat thread) and an agent; at least one is set.

    A query scope covers a record when the user ids are equal (both may be empty), and the
    session and agent ids are equal wherever the query sets them. So a user's scope sees
    all their sessions, and never another user's records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: ScopeId = ""
    session_id: ScopeId = ""
    agent_id: ScopeId = ""

    @model_validator(mode="after")
    def _not_empty(self) -> MemoryScope:
        if not (self.user_id or self.session_id or self.agent_id):
            raise ValueError("a memory scope needs a user_id, session_id or agent_id")
        return self

    def covers(self, other: MemoryScope) -> bool:
        """True when a record in `other` belongs to this (query) scope."""
        return (
            other.user_id == self.user_id
            and (not self.session_id or other.session_id == self.session_id)
            and (not self.agent_id or other.agent_id == self.agent_id)
        )


def scope_kind(scopes: Iterable[MemoryScope | None]) -> str:
    """Which scope fields are set, e.g. `user+session`; never the ids. `ids` for none."""
    names = {
        "+".join(
            name
            for name, value in (
                ("user", scope.user_id),
                ("session", scope.session_id),
                ("agent", scope.agent_id),
            )
            if value
        )
        for scope in scopes
        if scope is not None
    }
    return ",".join(sorted(names)) if names else "ids"


class MemoryRecord(BaseModel):
    """One remembered text in its scope; `id` and `created_at` are filled in on `add`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(max_length=MAX_ID_CHARS)] = ""
    scope: MemoryScope
    text: Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]
    metadata: dict[str, Any] = {}
    created_at: datetime | None = None
    expires_at: datetime | None = None
    source: Annotated[str, Field(max_length=MAX_ID_CHARS)] = ""  # the run or turn id

    @field_validator("metadata")
    @classmethod
    def _json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            text = json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be plain JSON (str keys, no NaN)") from exc
        if len(text) > MAX_METADATA_CHARS:
            raise ValueError(f"metadata is over {MAX_METADATA_CHARS} characters as JSON")
        return value

    @field_validator("created_at", "expires_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=UTC)  # a naive time is taken as UTC

    def live(self, now: datetime, retention_days: float | None) -> bool:
        """False once expired, or older than `retention_days`."""
        if self.expires_at is not None and self.expires_at <= now:
            return False
        if retention_days is not None and self.created_at is not None:
            return self.created_at > now - timedelta(days=retention_days)
        return True


class MemoryHit(BaseModel):
    """A search result: the record and its score (higher is better, backend-specific)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: MemoryRecord
    score: float = 0.0


def utc_now() -> datetime:
    """The current time, timezone-aware UTC."""
    return datetime.now(UTC)


def check_retention(retention_days: object) -> float | None:
    """`retention_days` as a float, or None.

    Raises:
        ConfigError: not a finite number > 0.
    """
    if retention_days is None:
        return None
    if (
        isinstance(retention_days, bool)
        or not isinstance(retention_days, int | float)
        or not math.isfinite(retention_days)
        or retention_days <= 0
    ):
        raise ConfigError(f"retention_days must be a finite number > 0, got {retention_days!r}")
    return float(retention_days)


def check_scope(scope: object, what: str = "scope") -> MemoryScope:
    """`scope` itself.

    Raises:
        ToolUsageError: not a MemoryScope.
    """
    if not isinstance(scope, MemoryScope):
        raise ToolUsageError(f"{what} must be a MemoryScope, got {type(scope).__name__}")
    return scope


def check_records(records: object) -> list[MemoryRecord]:
    """`records` as a list.

    Raises:
        ToolUsageError: not a sequence of MemoryRecord, over `MAX_BATCH`, or repeated ids.
    """
    if isinstance(records, MemoryRecord):
        records = [records]
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ToolUsageError(f"records must be a sequence of MemoryRecord, got {type(records)}")
    items = list(records)
    if not all(isinstance(item, MemoryRecord) for item in items):
        raise ToolUsageError("records must hold only MemoryRecord values")
    if len(items) > MAX_BATCH:
        raise ToolUsageError(f"add takes at most {MAX_BATCH} records per call, got {len(items)}")
    ids = [item.id for item in items if item.id]
    if len(ids) != len(set(ids)):
        raise ToolUsageError("records repeat an id")
    return items


def check_query(query: object, k: object) -> tuple[str, int]:
    """`(query, k)`.

    Raises:
        ToolUsageError: `query` not a str (or too long), or `k` not an int in 1..MAX_K.
    """
    if not isinstance(query, str):
        raise ToolUsageError(f"query must be a str, got {type(query).__name__}")
    if len(query) > MAX_TEXT_CHARS:
        raise ToolUsageError(f"query is longer than {MAX_TEXT_CHARS} characters")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ToolUsageError(f"k must be an int from 1 to {MAX_K}, got {k!r}")
    return query, k


def check_filters(filters: object) -> dict[str, FilterValue]:
    """Metadata equality filters as a dict.

    Raises:
        ToolUsageError: not a mapping of short keys to str, int, finite float or bool.
    """
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise ToolUsageError(f"filters must be a mapping, got {type(filters).__name__}")
    clean: dict[str, FilterValue] = {}
    for key, value in filters.items():
        if not isinstance(key, str) or not _FILTER_KEY.fullmatch(key):
            raise ToolUsageError(f"filter key {key!r} must match [A-Za-z0-9_.-]{{1,64}}")
        if not isinstance(value, str | int | float) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ToolUsageError(f"filter {key!r} must be a str, int, finite float or bool")
        clean[key] = value
    return clean


def check_ids(ids: object) -> list[str]:
    """`ids` as a list of non-empty strings.

    Raises:
        ToolUsageError: not a sequence of non-empty str, or over `MAX_BATCH`.
    """
    if isinstance(ids, str | bytes) or not isinstance(ids, Sequence):
        raise ToolUsageError(f"ids must be a sequence of str, got {type(ids).__name__}")
    items = list(ids)
    if not all(isinstance(item, str) and 0 < len(item) <= MAX_ID_CHARS for item in items):
        raise ToolUsageError("ids must hold only non-empty str values")
    if len(items) > MAX_BATCH:
        raise ToolUsageError(f"delete takes at most {MAX_BATCH} ids per call")
    return items


def matches(record: MemoryRecord, filters: Filters) -> bool:
    """True when every filter equals the record's metadata value (bools only match bools)."""
    for key, want in filters.items():
        if key not in record.metadata:
            return False
        have = record.metadata[key]
        if isinstance(want, bool) is not isinstance(have, bool) or have != want:
            return False
    return True
