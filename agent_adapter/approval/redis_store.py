"""RedisApprovalStore: approvals in Redis, shared by every worker and host of a service."""

from __future__ import annotations

import builtins
import logging
import math
from collections.abc import Callable
from typing import Any

from agent_adapter.approval import redis_scripts
from agent_adapter.approval.persist import (
    ApprovalStoreCorrupt,
    check_list,
    encode,
    filled,
    key,
    read_row,
    shown,
    stamped,
)
from agent_adapter.approval.store import ApprovalRecord, ApprovalStatus
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime.loop_local import LoopLocal, close_client

logger = logging.getLogger(__name__)

DEFAULT_TTL_S = 7 * 24 * 3600.0
_FIELDS = ("record", "status", "consumed", "run_id", "call_id")
_ATTEMPTS = 3  # a record changes at most twice (decided, consumed) under a swap
_PAGE = 100
Row = builtins.list[Any]  # [record, status, consumed, run_id, call_id] as Redis returns them


class RedisApprovalStore:
    """ApprovalStore on Redis (`agent-adapter[redis]`), shared by every worker and host.

    `client_factory()` returns a `redis.asyncio.Redis` (one per event loop), e.g.
    `lambda: Redis.from_url(url)` (keep `decode_responses=False`). `create` and every
    status change are Lua scripts; `decide` and `consume` compare-and-set the stored record,
    so each decision is consumed once. Each record expires `ttl_s` seconds after it was
    created (`None`: never; then run Redis with `maxmemory-policy noeviction`). Backend
    errors propagate (fail closed). On Redis Cluster put a hash tag in `prefix`
    (`"{approvals}:"`): the scripts address record keys built from it.

    Raises:
        ConfigError: `prefix` is not a string ending in ":", or `ttl_s` is not a finite
            number > 0 or None.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        prefix: str = "aa:approval:",
        ttl_s: float | None = DEFAULT_TTL_S,
    ) -> None:
        if not callable(client_factory):
            raise ConfigError("client_factory must be a callable returning a redis client")
        if not isinstance(prefix, str):
            raise ConfigError(f"prefix must be a str, got {type(prefix).__name__}")
        if not prefix.endswith(":"):  # also refuses "": a bare keyspace collides with anything
            raise ConfigError(f"prefix must be non-empty and end in ':', got {prefix!r}")
        if ttl_s is not None and (
            isinstance(ttl_s, bool)
            or not isinstance(ttl_s, int | float)
            or not math.isfinite(ttl_s)
            or ttl_s <= 0
        ):
            raise ConfigError(f"ttl_s must be a finite number > 0 or None, got {ttl_s!r}")
        self.prefix = prefix
        self.ttl_s = ttl_s
        self._p = key(prefix)
        self._clients: LoopLocal[Any] = LoopLocal(client_factory, close=close_client)

    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        """Store a pending record (empty id / `created_at` filled in), or return the open one.

        Raises:
            ToolUsageError: another call's record already has this approval id, or the
                record holds a value JSON cannot carry.
        """
        record = ApprovalRecord(request=filled(request), status="pending")
        text = encode(record)
        req = record.request
        ttl_ms = 0 if self.ttl_s is None else max(1, math.ceil(self.ttl_s * 1000))
        keys = (
            self._open(req.run_id, req.call_id),
            self._rec(req.approval_id),
            self._p + b"all",
            self._p + b"run:" + key(req.run_id),
            *self._status_keys(),
            self._p + b"exp",
            self._p + b"seq",
        )
        args = (self._p + b"rec:", key(req.approval_id), key(req.run_id), key(req.call_id))
        reply = await self._eval("create", redis_scripts.CREATE, keys, (*args, text, ttl_ms))
        outcome = _bytes(reply[0]) if reply else None
        if outcome == b"open":
            return _row_record(reply[1], reply[2:7])
        if outcome == b"taken":
            raise ToolUsageError(f"approval {req.approval_id!r} already exists")
        if outcome != b"created":
            logger.error("approval %s: unexpected create script reply", req.approval_id)
            raise ApprovalStoreCorrupt("unexpected reply from the approval create script")
        logger.debug("approval %s created for tool %r", req.approval_id, req.tool)
        return _row_record(key(req.approval_id), [text, "pending", "0", req.run_id, req.call_id])

    async def get(self, approval_id: str) -> ApprovalRecord:
        """The record. Raises: ApprovalNotFound."""
        return (await self._read(approval_id))[1]

    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        """The unconsumed record of the call, or None."""
        reply = await self._eval(
            "find", redis_scripts.FIND, (self._open(run_id, call_id),), (self._p + b"rec:",)
        )
        if not reply:
            return None
        return _row_record(reply[0], reply[1:6])

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> ApprovalRecord:
        """Record the decision; only the first decide of a record succeeds.

        Raises:
            ApprovalNotFound: unknown or expired id.
            ApprovalAlreadyDecided: the record is no longer pending.
            ToolUsageError: `edited_args` is not plain JSON.
        """
        decision = stamped(decision)
        status: ApprovalStatus = "approved" if decision.approved else "denied"
        for _ in range(_ATTEMPTS):
            raw, record = await self._read(approval_id)
            if record.status != "pending":
                raise ApprovalAlreadyDecided(approval_id, status=record.status)
            record = record.model_copy(update={"status": status, "decision": decision})
            if await self._swap("decide", approval_id, raw, record):
                logger.info("approval %s %s", approval_id, status)
                return record
        raise self._unsettled(approval_id)

    async def consume(self, approval_id: str) -> bool:
        """Mark a decided record used; True only for the caller that flipped it.

        Raises:
            ApprovalNotFound: unknown or expired id.
        """
        for _ in range(_ATTEMPTS):
            raw, record = await self._read(approval_id)
            if record.consumed or record.status == "pending":
                return False
            if await self._swap(
                "consume", approval_id, raw, record.model_copy(update={"consumed": True})
            ):
                return True
        raise self._unsettled(approval_id)

    async def list(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]:
        """Records newest first, filtered by run and status (expired ones are skipped).

        Raises:
            ConfigError: unknown `status` or a negative `limit`.
        """
        check_list(status, limit)
        if run_id is not None:
            index = self._p + b"run:" + key(run_id)
        elif status is not None:
            index = self._p + b"status:" + status.encode()
        else:
            index = self._p + b"all"
        found: builtins.list[ApprovalRecord] = []
        start = 0
        # Only a run and status filter together skips records, so it reads full pages.
        size = _PAGE if run_id is not None and status is not None else max(1, min(limit, _PAGE))
        while len(found) < limit:
            page = await self._eval(
                "list", redis_scripts.PAGE, (index,), (self._p + b"rec:", start, start + size - 1)
            )
            for item in page:
                if item[1] is None:
                    continue  # expired since it was indexed
                record = _row_record(item[0], item[1:6])
                if status is None or record.status == status:
                    found.append(record)
                    if len(found) >= limit:
                        break
            if len(page) < size:
                break
            start += size
        return found

    async def aclose(self) -> None:
        """Close the running loop's client."""
        await self._clients.aclose()

    # --- Redis side --------------------------------------------------------------------------

    def _rec(self, approval_id: str) -> bytes:
        return self._p + b"rec:" + key(approval_id)

    def _open(self, run_id: str, call_id: str) -> bytes:
        run = key(run_id)  # length-prefixed: no run id / call id pair can collide
        return self._p + b"open:" + str(len(run)).encode() + b":" + run + b":" + key(call_id)

    def _status_keys(self) -> tuple[bytes, bytes, bytes]:
        status = self._p + b"status:"
        return status + b"pending", status + b"approved", status + b"denied"

    async def _read(self, approval_id: str) -> tuple[bytes, ApprovalRecord]:
        try:
            row = await self._clients.get().hmget(self._rec(approval_id), list(_FIELDS))
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis approval store get failed")
            raise
        if row is None or row[0] is None:
            raise ApprovalNotFound(approval_id)
        return _bytes(row[0]) or b"", _row_record(key(approval_id), row)

    async def _swap(
        self, op: str, approval_id: str, expected: bytes, record: ApprovalRecord
    ) -> bool:
        keys = (self._rec(approval_id), *self._status_keys())
        consumed = "1" if record.consumed else "0"
        args = (expected, encode(record), record.status, consumed, key(approval_id))
        return bool(await self._eval(op, redis_scripts.SWAP, keys, args))

    async def _eval(
        self, op: str, script: str, keys: tuple[bytes, ...], args: tuple[Any, ...]
    ) -> Any:
        client = self._clients.get()
        try:
            return await client.eval(script, len(keys), *keys, *args)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis approval store %s failed", op)
            raise

    def _unsettled(self, approval_id: str) -> ApprovalStoreCorrupt:
        logger.error("approval %s kept changing under a swap: failing closed", approval_id)
        return ApprovalStoreCorrupt(f"approval {approval_id!r} kept changing; try again")


def _row_record(approval_id: Any, row: Row) -> ApprovalRecord:
    """The record of a hash read as `_FIELDS`, checked against its id."""
    if len(row) != len(_FIELDS):
        logger.error("approval store row %s has the wrong shape", shown(approval_id))
        raise ApprovalStoreCorrupt(f"approval store row {shown(approval_id)} has the wrong shape")
    raw, status, consumed, run_id, call_id = row
    return read_row(
        raw,
        approval_id=_bytes(approval_id),
        run_id=_bytes(run_id),
        call_id=_bytes(call_id),
        status=_bytes(status),
        consumed=_bytes(consumed),
    )


def _bytes(value: Any) -> bytes | None:
    # Clients made with decode_responses=True hand back str; keep both readable.
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return key(value)
    return None
