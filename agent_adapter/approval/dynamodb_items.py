"""DynamoDB item layout of DynamoDBApprovalStore: keys, attributes, the table and the client."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from agent_adapter.approval.persist import ApprovalStoreCorrupt, key, read_row, shown
from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

Item = dict[str, dict[str, Any]]  # attribute name -> typed value, e.g. {"S": "..."}

BY_RUN = "by_run"  # GSI: run_key, seq
BY_STATUS = "by_status"  # GSI: st, seq
SEQ_PK = "meta#seq"
# Short attribute names; `st` avoids the reserved word STATUS.
NAMES = {
    "#pk": "pk",
    "#aid": "aid",
    "#st": "st",
    "#c": "consumed",
    "#rec": "record",
    "#exp": "expires_at",
    "#run": "run_key",
    "#n": "n",
}
ALIVE = "(attribute_not_exists(#exp) OR #exp > :now)"
CANCELLED = "TransactionCanceledException"
RACE_CODES = frozenset({"ConditionalCheckFailedException", "TransactionConflictException"})
RACE_REASONS = frozenset({"ConditionalCheckFailed", "TransactionConflict"})
_NO_REASON = frozenset({"", "None"})  # "None": that item of the transaction was fine
_REASONS_IN_MESSAGE = re.compile(r"\[([A-Za-z, ]*)\]\s*$")


def esc(text: str) -> str:
    """`text` as ASCII for a string key; injective, and lone surrogates survive."""
    return json.dumps(text, ensure_ascii=True)[1:-1]


def rec_pk(approval_id: str) -> str:
    return "rec#" + esc(approval_id)


def open_pk(run_id: str, call_id: str) -> str:
    run = esc(run_id)  # length-prefixed: no run id / call id pair can collide
    return f"open#{len(run)}:{run}:{esc(call_id)}"


def run_key(run_id: str) -> str:
    return "run#" + esc(run_id)


def s(value: str) -> dict[str, str]:
    return {"S": value}


def n(value: int | float) -> dict[str, str]:
    return {"N": str(value)}


def b(value: bytes) -> dict[str, bytes]:
    return {"B": value}


def record_item(record: ApprovalRecord, text: str, seq: int, expires_at: int | None) -> Item:
    """The item of a new pending record."""
    req = record.request
    item: Item = {
        "pk": s(rec_pk(req.approval_id)),
        "aid": b(key(req.approval_id)),
        "run": b(key(req.run_id)),
        "call": b(key(req.call_id)),
        "run_key": s(run_key(req.run_id)),
        "st": s("pending"),
        "consumed": n(0),
        "seq": n(seq),
        "digest": s(record.args_digest),
        "record": s(text),
    }
    if expires_at is not None:
        item["expires_at"] = n(expires_at)
    return item


def alive(item: Item, now: float) -> bool:
    """False once the item's TTL has passed (DynamoDB deletes it later, maybe days later)."""
    expires = item.get("expires_at")
    return expires is None or float(expires.get("N", "0")) > now


def raw(item: Item, name: str) -> Any:
    """The plain value of attribute `name` (None when absent or of an unknown type)."""
    value = item.get(name) or {}
    for kind in ("S", "B", "N"):
        if kind in value:
            found = value[kind]
            return bytes(found) if kind == "B" and isinstance(found, bytearray) else found
    return None


def item_record(item: Item) -> ApprovalRecord:
    """The record of an item, checked against its key attributes."""
    return read_row(
        raw(item, "record"),
        approval_id=raw(item, "aid"),
        run_id=raw(item, "run"),
        call_id=raw(item, "call"),
        status=raw(item, "st"),
        consumed=raw(item, "consumed"),
    )


def seq_of(item: Item) -> int:
    """The creation counter of an item."""
    try:
        return int(raw(item, "seq"))
    except (TypeError, ValueError) as exc:
        where = shown(raw(item, "aid"))
        logger.error("approval store item %s has no valid seq", where)
        raise ApprovalStoreCorrupt(f"approval store item {where} has no valid seq") from exc


def error_code(exc: BaseException) -> str:
    """The botocore error code of `exc` ('' for anything else)."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            return str(error.get("Code", ""))
    return ""


def cancellation_reasons(exc: BaseException) -> list[str]:
    """The per-item codes of a cancelled transaction, from the response or its message."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return []
    reasons = response.get("CancellationReasons")
    if isinstance(reasons, list):
        return [str(r.get("Code", "")) if isinstance(r, dict) else "" for r in reasons]
    error = response.get("Error")
    message = str(error.get("Message", "")) if isinstance(error, dict) else ""
    found = _REASONS_IN_MESSAGE.search(message)
    return [part.strip() for part in found.group(1).split(",")] if found else []


def lost_race(exc: BaseException) -> bool:
    """True when `exc` is a lost compare-and-set, not throttling, validation or an outage."""
    code = error_code(exc)
    if code in RACE_CODES:
        return True
    if code != CANCELLED:
        return False
    reasons = [r for r in cancellation_reasons(exc) if r not in _NO_REASON]
    return bool(reasons) and all(r in RACE_REASONS for r in reasons)


def table_definition(table: str) -> dict[str, Any]:
    """`CreateTable` arguments: key `pk`, GSIs `by_run` and `by_status`, on-demand billing."""
    projection = {"ProjectionType": "ALL"}
    return {
        "TableName": table,
        "BillingMode": "PAY_PER_REQUEST",
        "AttributeDefinitions": [
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "run_key", "AttributeType": "S"},
            {"AttributeName": "st", "AttributeType": "S"},
            {"AttributeName": "seq", "AttributeType": "N"},
        ],
        "KeySchema": [{"AttributeName": "pk", "KeyType": "HASH"}],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": BY_RUN,
                "KeySchema": [
                    {"AttributeName": "run_key", "KeyType": "HASH"},
                    {"AttributeName": "seq", "KeyType": "RANGE"},
                ],
                "Projection": projection,
            },
            {
                "IndexName": BY_STATUS,
                "KeySchema": [
                    {"AttributeName": "st", "KeyType": "HASH"},
                    {"AttributeName": "seq", "KeyType": "RANGE"},
                ],
                "Projection": projection,
            },
        ],
    }


class LoopClient:
    """One event loop's DynamoDB client, entered on first use when the factory gives a context."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._client: Any = None
        self._context: Any = None
        self._lock = asyncio.Lock()

    async def get(self) -> Any:
        """The client; ConfigError when the factory gives neither a client nor a context."""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                made = self._factory()
                if hasattr(made, "get_item"):
                    self._client = made
                elif hasattr(made, "__aenter__"):
                    self._client = await made.__aenter__()
                    self._context = made
                else:
                    raise ConfigError("client_factory must return a DynamoDB client or context")
        return self._client

    async def aclose(self) -> None:
        """Exit the entered context, or close a plain client."""
        client, context = self._client, self._context
        self._client = self._context = None
        if context is not None:
            await context.__aexit__(None, None, None)
        elif client is not None and hasattr(client, "close"):
            await client.close()


async def close_loop_client(holder: LoopClient) -> None:
    """A `LoopLocal` close for `LoopClient`."""
    await holder.aclose()
