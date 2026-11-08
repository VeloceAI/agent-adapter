"""DynamoDBApprovalStore: approvals in one DynamoDB table, for serverless and AWS services."""

from __future__ import annotations

import asyncio
import builtins
import logging
import math
import random
import time
from collections.abc import Callable
from typing import Any

from agent_adapter.approval import dynamodb_items as items
from agent_adapter.approval.dynamodb_items import ALIVE, NAMES, Item, b, n, s
from agent_adapter.approval.persist import (
    ApprovalStoreCorrupt,
    check_list,
    encode,
    filled,
    key,
    stamped,
)
from agent_adapter.approval.store import STATUSES, ApprovalRecord, ApprovalStatus
from agent_adapter.core.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    ConfigError,
    ToolUsageError,
    log_failure,
)
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime.loop_local import LoopLocal

logger = logging.getLogger(__name__)

DEFAULT_TTL_S = 7 * 24 * 3600.0
_ATTEMPTS = 5  # lost races retry: a record changes at most twice, a call's marker rarely
_BACKOFF_BASE_S = 0.02  # full jitter: sleep uniform(0, min(cap, base * 2**attempt))
_BACKOFF_CAP_S = 0.5
_PAGE = 100
_sleep = asyncio.sleep


class _LostRace(Exception):
    """A conditional write lost to another writer; the caller re-reads and retries."""


class DynamoDBApprovalStore:
    """ApprovalStore on one DynamoDB table (aioboto3 or aiobotocore), shared by every host.

    `client_factory()` returns an async DynamoDB client, or the context manager that
    `aioboto3.Session().client("dynamodb")` returns; one is made per event loop. `create`
    is one transaction of two conditional puts (the record and the call's open marker);
    `decide` and `consume` compare-and-set the stored record, so each decision is consumed
    once. Every read that decides is strongly consistent; `list` reads a GSI (eventually
    consistent). Items carry `expires_at` `ttl_s` seconds after creation (`None`: never);
    enable DynamoDB TTL on it. Expired items read as absent. Backend errors propagate.

    Raises:
        ConfigError: no callable `client_factory`, an empty `table`, or `ttl_s` not a
            finite number > 0 or None.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        table: str,
        ttl_s: float | None = DEFAULT_TTL_S,
    ) -> None:
        if not callable(client_factory):
            raise ConfigError("client_factory must be a callable returning a DynamoDB client")
        if not isinstance(table, str) or not table:
            raise ConfigError(f"table must be a non-empty str, got {table!r}")
        if ttl_s is not None and (
            isinstance(ttl_s, bool)
            or not isinstance(ttl_s, int | float)
            or not math.isfinite(ttl_s)
            or ttl_s <= 0
        ):
            raise ConfigError(f"ttl_s must be a finite number > 0 or None, got {ttl_s!r}")
        self.table = table
        self.ttl_s = ttl_s
        self._clients: LoopLocal[items.LoopClient] = LoopLocal(
            lambda: items.LoopClient(client_factory), close=items.close_loop_client
        )

    @staticmethod
    def table_definition(table: str) -> dict[str, Any]:
        """`CreateTable` arguments for the store's table (then enable TTL on `expires_at`)."""
        return items.table_definition(table)

    async def create(self, request: ApprovalRequest) -> ApprovalRecord:
        """Store a pending record (empty id / `created_at` filled in), or return the open one.

        Raises:
            ToolUsageError: another call's record already has this approval id, or the
                record holds a value JSON cannot carry.
        """
        record = ApprovalRecord(request=filled(request), status="pending")
        text = encode(record)
        req = record.request
        for attempt in range(_ATTEMPTS):
            await _backoff(attempt)
            now = time.time()
            marker = await self._get_item("create", items.open_pk(req.run_id, req.call_id))
            old: bytes | None = None
            if marker is not None:
                current = await self._open_record(marker, now, req.run_id, req.call_id)
                if current is not None:
                    return current
                old = items.raw(marker, "aid")
            taken = await self._get_item("create", items.rec_pk(req.approval_id))
            if taken is not None and items.alive(taken, now):
                raise ToolUsageError(f"approval {req.approval_id!r} already exists")
            expires = None if self.ttl_s is None else math.ceil(now + self.ttl_s)
            seq = await self._next_seq()
            try:
                await self._put_pair(record, text, seq, expires, old, now)
            except _LostRace:
                continue
            logger.debug("approval %s created for tool %r", req.approval_id, req.tool)
            return record
        raise _unsettled(req.approval_id)

    async def get(self, approval_id: str) -> ApprovalRecord:
        """The record. Raises: ApprovalNotFound (unknown or expired)."""
        return items.item_record(await self._live_item("get", approval_id))

    async def find(self, run_id: str, call_id: str) -> ApprovalRecord | None:
        """The unconsumed record of the call, or None."""
        marker = await self._get_item("find", items.open_pk(run_id, call_id))
        if marker is None:
            return None
        return await self._open_record(marker, time.time(), run_id, call_id)

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> ApprovalRecord:
        """Record the decision; only the first decide of a record succeeds.

        Raises:
            ApprovalNotFound: unknown or expired id.
            ApprovalAlreadyDecided: the record is no longer pending.
            ToolUsageError: `edited_args` is not plain JSON.
        """
        decision = stamped(decision)
        status: ApprovalStatus = "approved" if decision.approved else "denied"
        for attempt in range(_ATTEMPTS):
            await _backoff(attempt)
            item = await self._live_item("decide", approval_id)
            record = items.item_record(item)
            if record.status != "pending":
                raise ApprovalAlreadyDecided(approval_id, status=record.status)
            record = record.model_copy(update={"status": status, "decision": decision})
            if await self._swap("decide", item, record, "SET #st = :st, #rec = :new", ""):
                logger.info("approval %s %s", approval_id, status)
                return record
        raise _unsettled(approval_id)

    async def consume(self, approval_id: str) -> bool:
        """Mark a decided record used; True only for the caller that flipped it.

        Raises:
            ApprovalNotFound: unknown or expired id.
        """
        for attempt in range(_ATTEMPTS):
            await _backoff(attempt)
            item = await self._live_item("consume", approval_id)
            record = items.item_record(item)
            if record.consumed or record.status == "pending":
                return False
            used = record.model_copy(update={"consumed": True})
            if await self._swap(
                "consume", item, used, "SET #c = :one, #rec = :new", " AND #c = :zero"
            ):
                return True
        raise _unsettled(approval_id)

    async def list(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> builtins.list[ApprovalRecord]:
        """Records newest first, filtered by run and status (expired ones are skipped).

        Raises:
            ConfigError: unknown `status` or a negative `limit`.
        """
        check_list(status, limit)
        if limit == 0:
            return []
        if run_id is not None:
            found = await self._query(items.BY_RUN, "#run", items.run_key(run_id), status, limit)
            return found
        wanted = [status] if status is not None else sorted(STATUSES)
        merged: dict[str, tuple[int, ApprovalRecord]] = {}
        for each in wanted:
            for seq, record in await self._query_seq(items.BY_STATUS, "#st", each, None, limit):
                merged.setdefault(record.approval_id, (seq, record))
        ranked = sorted(merged.values(), key=lambda pair: pair[0], reverse=True)
        return [record for _, record in ranked[:limit]]

    async def aclose(self) -> None:
        """Close the running loop's client."""
        await self._clients.aclose()

    # --- DynamoDB side ---------------------------------------------------------------------

    async def _call(self, op: str, method: str, **kwargs: Any) -> Any:
        client = await self._clients.get().get()
        try:
            return await getattr(client, method)(**kwargs)
        except Exception as exc:
            if items.lost_race(exc):
                logger.debug("dynamodb approval store %s lost a conditional write race", op)
                raise _LostRace from exc
            cause = items.error_code(exc) or "-"
            reasons = ",".join(items.cancellation_reasons(exc)) or "-"
            log_failure(
                logger,
                logging.ERROR,
                exc,
                "dynamodb approval store %s failed: code=%s reasons=%s",
                op,
                cause,
                reasons,
            )
            raise

    async def _get_item(self, op: str, pk: str) -> Item | None:
        reply = await self._call(
            op, "get_item", TableName=self.table, Key={"pk": s(pk)}, ConsistentRead=True
        )
        found: Item | None = reply.get("Item") or None
        return found

    async def _live_item(self, op: str, approval_id: str) -> Item:
        item = await self._get_item(op, items.rec_pk(approval_id))
        if item is None:
            raise ApprovalNotFound(approval_id)
        if not items.alive(item, time.time()):  # past expires_at, before DynamoDB deletes it
            raise ApprovalNotFound(approval_id, expired=True, tool=_tool_of(item))
        return item

    async def _open_record(
        self, marker: Item, now: float, run_id: str, call_id: str
    ) -> ApprovalRecord | None:
        aid = items.raw(marker, "aid")
        if not isinstance(aid, bytes) or not items.alive(marker, now):
            return None
        item = await self._get_item("find", items.rec_pk(aid.decode("utf-8", "surrogatepass")))
        if item is None or not items.alive(item, now):
            return None
        record = items.item_record(item)
        if (record.request.run_id, record.request.call_id) != (run_id, call_id):
            logger.error("approval marker of a call points at another call: failing closed")
            raise ApprovalStoreCorrupt("approval marker points at another call's record")
        return None if record.consumed else record

    async def _next_seq(self) -> int:
        reply = await self._call(
            "create",
            "update_item",
            TableName=self.table,
            Key={"pk": s(items.SEQ_PK)},
            UpdateExpression="ADD #n :one",
            ExpressionAttributeNames={"#n": "n"},
            ExpressionAttributeValues={":one": n(1)},
            ReturnValues="UPDATED_NEW",
        )
        return items.seq_of({"seq": reply["Attributes"]["n"]})

    async def _put_pair(
        self,
        record: ApprovalRecord,
        text: str,
        seq: int,
        expires: int | None,
        old: bytes | None,
        now: float,
    ) -> None:
        req = record.request
        marker: Item = {
            "pk": s(items.open_pk(req.run_id, req.call_id)),
            "aid": b(key(req.approval_id)),
        }
        if expires is not None:
            marker["expires_at"] = n(expires)
        if old is None:
            marker_check: dict[str, Any] = {"ConditionExpression": "attribute_not_exists(#pk)"}
        else:
            marker_check = {
                "ConditionExpression": "#aid = :old",
                "ExpressionAttributeValues": {":old": b(old)},
            }
        record_check = {
            "ConditionExpression": "attribute_not_exists(#pk) OR #exp <= :now",
            "ExpressionAttributeValues": {":now": n(now)},
        }
        await self._call(
            "create",
            "transact_write_items",
            TransactItems=[
                {"Put": self._put(marker, marker_check)},
                {"Put": self._put(items.record_item(record, text, seq, expires), record_check)},
            ],
        )

    def _put(self, item: Item, check: dict[str, Any]) -> dict[str, Any]:
        used = {name: NAMES[name] for name in NAMES if name in check["ConditionExpression"]}
        return {"TableName": self.table, "Item": item, "ExpressionAttributeNames": used, **check}

    async def _swap(
        self, op: str, item: Item, record: ApprovalRecord, update: str, extra: str
    ) -> bool:
        condition = f"#rec = :old AND {ALIVE}{extra}"
        values: dict[str, Any] = {
            ":old": s(str(items.raw(item, "record"))),
            ":new": s(encode(record)),
            ":now": n(time.time()),
        }
        if op == "decide":
            values[":st"] = s(record.status)
        else:
            values[":one"], values[":zero"] = n(1), n(0)
        used = {name: NAMES[name] for name in NAMES if name in update + condition}
        try:
            await self._call(
                op,
                "update_item",
                TableName=self.table,
                Key={"pk": item["pk"]},
                UpdateExpression=update,
                ConditionExpression=condition,
                ExpressionAttributeNames=used,
                ExpressionAttributeValues=values,
            )
        except _LostRace:
            return False
        return True

    async def _query(
        self, index: str, name: str, value: str, status: str | None, limit: int
    ) -> builtins.list[ApprovalRecord]:
        return [r for _, r in await self._query_seq(index, name, value, status, limit)]

    async def _query_seq(
        self, index: str, name: str, value: str, status: str | None, limit: int
    ) -> builtins.list[tuple[int, ApprovalRecord]]:
        found: builtins.list[tuple[int, ApprovalRecord]] = []
        start: Item | None = None
        while len(found) < limit:
            page: dict[str, Any] = {
                "TableName": self.table,
                "IndexName": index,
                "KeyConditionExpression": f"{name} = :v",
                "ExpressionAttributeNames": {name: NAMES[name]},
                "ExpressionAttributeValues": {":v": s(value)},
                "ScanIndexForward": False,
                "Limit": _PAGE,
            }
            if start is not None:
                page["ExclusiveStartKey"] = start
            reply = await self._call("list", "query", **page)
            now = time.time()
            for item in reply.get("Items", []):
                if not items.alive(item, now):
                    continue
                record = items.item_record(item)
                if status is None or record.status == status:
                    found.append((items.seq_of(item), record))
                    if len(found) >= limit:
                        break
            start = reply.get("LastEvaluatedKey")
            if not start:
                break
        return found


async def _backoff(attempt: int) -> None:
    """Spread retries after a lost race so racing writers stop colliding (none before try 0)."""
    if attempt > 0:
        ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2 ** (attempt - 1))
        await _sleep(random.uniform(0, ceiling))


def _unsettled(approval_id: str) -> ApprovalStoreCorrupt:
    logger.error("approval %s kept changing under a conditional write: failing closed", approval_id)
    return ApprovalStoreCorrupt(f"approval {approval_id!r} kept changing; try again")


def _tool_of(item: Item) -> str:
    """The tool of an expired item, or "" when the item cannot be read."""
    try:
        return items.item_record(item).request.tool
    except Exception as exc:  # an unreadable expired item still counts as expired
        log_failure(logger, logging.WARNING, exc, "expired approval item unreadable")
        return ""
