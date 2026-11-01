"""Async DynamoDB clients for DynamoDBApprovalStore tests: an in-memory fake and a moto shim.

The fake evaluates the condition and update expressions the store sends (a small subset of
DynamoDB's grammar) and applies each request atomically. `MotoClient` runs every call on a
real boto3 client under moto, so the conformance suite also checks DynamoDB's own rules.
"""

from __future__ import annotations

import asyncio
import copy
import re
import threading
from typing import Any

from agent_adapter.approval.dynamodb_items import table_definition

Item = dict[str, dict[str, Any]]
_TOKEN = re.compile(r"\s*(<=|>=|<>|[=<>(),]|[#:]?\w+)")


class FakeClientError(Exception):
    """Shaped like botocore's ClientError: `response["Error"]["Code"]`, cancellation reasons."""

    def __init__(self, code: str, reasons: list[str] | None = None) -> None:
        super().__init__(code)
        self.response: dict[str, Any] = {"Error": {"Code": code, "Message": code}}
        if reasons is not None:
            self.response["CancellationReasons"] = [{"Code": r} for r in reasons]


def _value(attr: dict[str, Any] | None) -> Any:
    if attr is None:
        return None
    if "N" in attr:
        return float(attr["N"])
    return attr.get("S", attr.get("B"))


class _Condition:
    """Recursive descent over `or`, `and`, parentheses, attribute_not_exists and comparisons."""

    def __init__(self, text: str, names: dict[str, str], values: dict[str, Any], item: Item):
        self.tokens = _TOKEN.findall(text)
        self.names, self.values, self.item, self.at = names, values, item, 0

    def run(self) -> bool:
        result = self._or()
        assert self.at == len(self.tokens), f"unparsed: {self.tokens[self.at :]}"
        return result

    def _take(self) -> str:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def _peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def _or(self) -> bool:
        result = self._and()
        while self._peek() == "OR":
            self._take()
            result = self._and() or result
        return result

    def _and(self) -> bool:
        result = self._primary()
        while self._peek() == "AND":
            self._take()
            result = self._primary() and result
        return result

    def _primary(self) -> bool:
        token = self._take()
        if token == "(":
            result = self._or()
            assert self._take() == ")"
            return result
        if token == "attribute_not_exists":
            assert self._take() == "("
            name = self.names[self._take()]
            assert self._take() == ")"
            return name not in self.item
        left, op, right = self._operand(token), self._take(), self._operand(self._take())
        if left is None or right is None:
            return False
        return {
            "=": left == right,
            "<>": left != right,
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }[op]

    def _operand(self, token: str) -> Any:
        if token.startswith(":"):
            return _value(self.values[token])
        return _value(self.item.get(self.names[token]))


class FakeDynamoDB:
    """One table with GSIs `by_run` and `by_status`; one request at a time, like the service."""

    def __init__(self, *, broken: bool = False) -> None:
        self.items: dict[str, Item] = {}
        self.broken = broken
        self.closed = False
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self._gsi = {
            g["IndexName"]: g["KeySchema"][0]["AttributeName"]
            for g in table_definition("t")["GlobalSecondaryIndexes"]
        }

    def _enter(self, method: str) -> None:
        self.calls.append(method)
        if self.broken:
            raise FakeClientError("InternalServerError")

    async def get_item(self, *, TableName: str, Key: Item, ConsistentRead: bool) -> Any:
        await asyncio.sleep(0)
        with self._lock:
            self._enter("get_item")
            found = self.items.get(Key["pk"]["S"])
            return {"Item": copy.deepcopy(found)} if found else {}

    async def update_item(self, *, TableName: str, Key: Item, **kw: Any) -> Any:
        await asyncio.sleep(0)
        with self._lock:
            self._enter("update_item")
            pk = Key["pk"]["S"]
            item = copy.deepcopy(self.items.get(pk, {"pk": {"S": pk}}))
            names, values = kw.get("ExpressionAttributeNames", {}), kw["ExpressionAttributeValues"]
            check = kw.get("ConditionExpression")
            if check and not _Condition(check, names, values, self.items.get(pk, {})).run():
                raise FakeClientError("ConditionalCheckFailedException")
            _apply(kw["UpdateExpression"], names, values, item)
            self.items[pk] = item
            return {"Attributes": copy.deepcopy(item)}

    async def transact_write_items(self, *, TransactItems: list[dict[str, Any]]) -> Any:
        await asyncio.sleep(0)
        with self._lock:
            self._enter("transact_write_items")
            puts = [entry["Put"] for entry in TransactItems]
            reasons = []
            for put in puts:
                current = self.items.get(put["Item"]["pk"]["S"], {})
                names = put.get("ExpressionAttributeNames", {})
                values = put.get("ExpressionAttributeValues", {})
                passed = _Condition(put["ConditionExpression"], names, values, current).run()
                reasons.append("None" if passed else "ConditionalCheckFailed")
            if any(r != "None" for r in reasons):
                raise FakeClientError("TransactionCanceledException", reasons)
            for put in puts:
                self.items[put["Item"]["pk"]["S"]] = copy.deepcopy(put["Item"])
            return {}

    async def query(self, **kw: Any) -> Any:
        await asyncio.sleep(0)
        with self._lock:
            self._enter("query")
            attribute = self._gsi[kw["IndexName"]]
            wanted = kw["ExpressionAttributeValues"][":v"]["S"]
            rows = [i for i in self.items.values() if i.get(attribute, {}).get("S") == wanted]
            rows.sort(key=lambda i: float(i["seq"]["N"]), reverse=not kw["ScanIndexForward"])
            start = kw.get("ExclusiveStartKey")
            if start is not None:
                keys = [i["pk"]["S"] for i in rows]
                rows = rows[keys.index(start["pk"]["S"]) + 1 :]
            page = rows[: kw["Limit"]]
            reply: dict[str, Any] = {"Items": copy.deepcopy(page)}
            if len(rows) > kw["Limit"]:
                last = page[-1]
                reply["LastEvaluatedKey"] = {k: last[k] for k in ("pk", attribute, "seq")}
            return reply

    async def close(self) -> None:
        self.closed = True


def _apply(update: str, names: dict[str, str], values: dict[str, Any], item: Item) -> None:
    if update.startswith("ADD "):
        name, value = update[4:].split()
        old = float(item.get(names[name], {"N": "0"})["N"])
        item[names[name]] = {"N": str(int(old + float(values[value]["N"])))}
        return
    assert update.startswith("SET ")
    for part in update[4:].split(","):
        name, value = (p.strip() for p in part.split("="))
        item[names[name]] = copy.deepcopy(values[value])


class MotoClient:
    """A boto3 DynamoDB client under moto, awaited like aiobotocore (one call at a time)."""

    _lock = threading.Lock()

    def __init__(self, client: Any) -> None:
        self.client = client

    def __getattr__(self, method: str) -> Any:
        call = getattr(self.client, method)

        async def run(**kwargs: Any) -> Any:
            await asyncio.sleep(0)
            with MotoClient._lock:
                return call(**kwargs)

        return run
