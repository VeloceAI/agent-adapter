"""Approval records and the store that keeps them between request and decision."""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from agent_adapter.approval._records import MAX_JSON_DEPTH as MAX_JSON_DEPTH
from agent_adapter.approval._records import STATUSES as STATUSES
from agent_adapter.approval._records import check_json as check_json
from agent_adapter.approval._records import check_list, filled, stamped
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest

logger = logging.getLogger(__name__)

ApprovalStatus = Literal["pending", "approved", "denied"]


def request_digest(request: ApprovalRequest) -> str:
    """SHA-256 hex of the canonical JSON of a request's kind, tool and original arguments.

    Canonical form: sorted keys, `(",", ":")` separators, `ensure_ascii=False`, encoded
    UTF-8 with `surrogatepass` (lone surrogates stay distinct); JSON values only.

    Raises:
        ToolUsageError: an argument is not plain JSON (bytes, tuple, set, other objects,
            non-string keys, NaN or infinity) or nests deeper than `MAX_JSON_DEPTH`;
            approvals fail closed on those.
    """
    return _approval_digest(request.kind, request.tool, request.arguments)


def _approval_digest(kind: str, tool: str, arguments: Mapping[str, Any]) -> str:
    check_json(arguments)
    payload = {"kind": kind, "tool": tool, "arguments": arguments}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()


class ApprovalRecord(BaseModel):
    """One approval request, its status, its decision and whether a resume used it.

    `args_digest` binds the decision to the requested kind, tool and arguments. It is
    always computed from `request` (stores may persist and index it); a given value that
    disagrees raises `ToolUsageError`, so a tampered row never validates.
    """

    model_config = ConfigDict(frozen=True)

    request: ApprovalRequest
    status: ApprovalStatus
    decision: ApprovalDecision | None = None
    consumed: bool = False
    args_digest: str = ""

    @model_validator(mode="before")
    @classmethod
    def _bind_digest(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "request" not in data:
            return data
        request = data["request"]
        if not isinstance(request, ApprovalRequest):
            request = ApprovalRequest.model_validate(request)
        digest = request_digest(request)
        given = data.get("args_digest") or ""
        if given and given != digest:
            raise ToolUsageError(
                f"approval {request.approval_id!r}: args_digest does not match its request"
            )
        return {**data, "request": request, "args_digest": digest}

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A copy; changing `request` or `args_digest` re-validates, so the digest stays bound."""
        if not update or not {"request", "args_digest"} & set(update):
            return super().model_copy(update=update, deep=deep)
        data = {name: getattr(self, name) for name in type(self).model_fields}
        data.update(update)
        if "args_digest" not in update:
            data["args_digest"] = ""
        return type(self).model_validate(data)

    @property
    def approval_id(self) -> str:
        return self.request.approval_id


@runtime_checkable
class ApprovalStore(Protocol):
    """Where deferred approvals wait; swap the memory store for a database in production.

    `create` and `consume` must be atomic: at most one unconsumed record per
    `(run_id, call_id)`, and each decision is consumed exactly once.
    """

    # A new pending record, or the call's unconsumed record unchanged (nothing stored).
    async def create(self, request: ApprovalRequest) -> ApprovalRecord: ...

    async def get(self, approval_id: str) -> ApprovalRecord: ...  # ApprovalNotFound

    # The unconsumed record of the call, or None.
    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None: ...

    async def decide(  # ApprovalNotFound, ApprovalAlreadyDecided, ToolUsageError
        self, approval_id: str, decision: ApprovalDecision
    ) -> ApprovalRecord: ...

    # Test-and-set: True only for the caller that marks a decided record consumed; False
    # when it is already consumed or still pending; ApprovalNotFound.
    async def consume(self, approval_id: str) -> bool: ...

    async def list(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]: ...  # `builtins.list`: the method shadows `list`


class MemoryApprovalStore:
    """In-process ApprovalStore; one `threading.Lock` makes it safe across event loops.

    Every operation is synchronous under the lock (no awaits inside), so loops in several
    threads (`run_sync`) can share one store. Past `max_records` it evicts the earliest
    consumed record, else the oldest decided one, else the oldest pending one (WARNING for
    both: a decided record is lost before use); each eviction is O(1).
    """

    def __init__(self, *, max_records: int = 10_000) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ConfigError(f"max_records must be a positive int, got {max_records!r}")
        self.max_records = max_records
        self._lock = threading.Lock()
        self._records: dict[str, ApprovalRecord] = {}  # creation order
        self._by_call: dict[tuple[str, str], builtins.list[str]] = {}
        self._consumed: deque[str] = deque()  # consume order
        self._decided: dict[str, None] = {}  # decided and not consumed, in decide order

    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        """Store a pending record (empty id / `created_at` filled in), or return the open one.

        Atomic per `(run_id, call_id)`: while the call has an unconsumed record, that record
        is returned unchanged and nothing is stored.

        Raises:
            ToolUsageError: another call's record already has this approval id.
        """
        request = filled(request)
        record = ApprovalRecord(request=request, status="pending")
        key = (request.run_id, request.call_id)
        with self._lock:
            open_record = self._open(key)
            if open_record is not None:
                return open_record
            if record.approval_id in self._records:
                raise ToolUsageError(f"approval {record.approval_id!r} already exists")
            self._records[record.approval_id] = record
            self._by_call.setdefault(key, []).append(record.approval_id)
            self._evict()
        logger.debug("approval %s created for tool %r", record.approval_id, request.tool)
        return record

    async def get(self, approval_id: str) -> ApprovalRecord:
        with self._lock:
            return self._get(approval_id)

    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._open((run_id, call_id))

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> ApprovalRecord:
        """Record the decision; only the first decide of a record succeeds.

        Raises:
            ApprovalNotFound: unknown id.
            ApprovalAlreadyDecided: the record is no longer pending.
            ToolUsageError: `edited_args` is not plain JSON (nothing is stored).
        """
        decision = stamped(decision)
        status: ApprovalStatus = "approved" if decision.approved else "denied"
        with self._lock:
            record = self._get(approval_id)
            if record.status != "pending":
                raise ApprovalAlreadyDecided(approval_id, status=record.status)
            record = record.model_copy(update={"status": status, "decision": decision})
            self._records[approval_id] = record
            self._decided[approval_id] = None  # pending records are never consumed
        logger.info("approval %s %s", approval_id, status)
        return record

    async def consume(self, approval_id: str) -> bool:
        """Mark a decided record used; True only for the caller that flipped it.

        A pending record is never consumed (False): only a decision can be used up.

        Raises:
            ApprovalNotFound: unknown id.
        """
        with self._lock:
            record = self._get(approval_id)
            if record.consumed or record.status == "pending":
                return False
            self._records[approval_id] = record.model_copy(update={"consumed": True})
            self._decided.pop(approval_id, None)
            self._consumed.append(approval_id)
            return True

    async def list(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]:
        """Records newest first, filtered by run and status.

        Raises:
            ConfigError: unknown `status` or a negative `limit`.
        """
        check_list(status, limit)
        found: builtins.list[ApprovalRecord] = []
        with self._lock:
            for record in reversed(self._records.values()):
                if len(found) >= limit:
                    break
                if run_id is not None and record.request.run_id != run_id:
                    continue
                if status is not None and record.status != status:
                    continue
                found.append(record)
        return found

    def _get(self, approval_id: str) -> ApprovalRecord:
        record = self._records.get(approval_id)
        if record is None:
            raise ApprovalNotFound(approval_id)
        return record

    def _open(self, key: tuple[str, str]) -> ApprovalRecord | None:
        # Only a call's newest record can be unconsumed: create refuses a second open one.
        # Caller holds _lock.
        ids = self._by_call.get(key)
        if not ids:
            return None
        record = self._records[ids[-1]]
        return None if record.consumed else record

    def _evict(self) -> None:
        # Caller holds _lock.
        while len(self._records) > self.max_records:
            victim = self._victim()
            record = self._records.pop(victim)
            self._decided.pop(victim, None)
            key = (record.request.run_id, record.request.call_id)
            ids = self._by_call[key]
            ids.remove(victim)
            if not ids:
                del self._by_call[key]
            if record.status == "pending":
                logger.warning(
                    "evicted pending approval %r (store full at %d records)",
                    victim,
                    self.max_records,
                )
            elif not record.consumed:  # a reviewer's decision is lost: the call asks again
                logger.warning(
                    "evicted decided approval %r before it was used (store full at %d records)",
                    victim,
                    self.max_records,
                )

    def _victim(self) -> str:
        if self._consumed:
            return self._consumed.popleft()
        if self._decided:
            return next(iter(self._decided))
        return next(iter(self._records))
