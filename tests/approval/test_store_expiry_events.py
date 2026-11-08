"""A store TTL that ran out says so, and `executor.resume` turns it into `approval.expired`."""

from __future__ import annotations

from typing import Any

import pytest

from agent_adapter.approval import DeferredApproval, dynamodb_store
from agent_adapter.approval.dynamodb_store import DynamoDBApprovalStore
from agent_adapter.approval.redis_store import RedisApprovalStore
from agent_adapter.core.errors import ApprovalNotFound
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec import Executor
from agent_adapter.tracing.sinks import EventLog
from tests.approval.fake_dynamodb import FakeDynamoDB
from tests.approval.fake_redis_store import FakeApprovalRedis
from tests.approval.test_store_dynamodb import Clock


def req() -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="run_1", call_id="call_1", tool="pay", arguments={"amount": 5}
    )


async def test_redis_tells_expired_from_unknown() -> None:
    fake = FakeApprovalRedis()
    store = RedisApprovalStore(lambda: fake, prefix="t:", ttl_s=60)
    record = await store.create(req())
    fake.clock.advance(61)
    with pytest.raises(ApprovalNotFound) as gone:
        await store.get(record.approval_id)
    assert gone.value.expired and "expired" in str(gone.value)
    with pytest.raises(ApprovalNotFound) as unknown:
        await store.get("apr_never")
    assert not unknown.value.expired


async def test_redis_expiry_lookup_failing_counts_as_unknown() -> None:
    fake = FakeApprovalRedis()
    store = RedisApprovalStore(lambda: fake, prefix="t:", ttl_s=60)

    async def broken(*args: Any) -> None:
        raise ConnectionError("down")

    fake.zscore = broken  # type: ignore[method-assign]
    with pytest.raises(ApprovalNotFound) as gone:
        await store.get("apr_x")
    assert not gone.value.expired


async def test_dynamodb_tells_expired_and_names_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock(1_000_000.0)
    monkeypatch.setattr(dynamodb_store, "time", clock)
    store = DynamoDBApprovalStore(FakeDynamoDB, table="t", ttl_s=60)
    record = await store.create(req())
    clock.now += 61
    with pytest.raises(ApprovalNotFound) as gone:
        await store.get(record.approval_id)
    assert gone.value.expired and gone.value.tool == "pay"


async def test_resume_of_an_expired_approval_sends_approval_expired() -> None:
    fake = FakeApprovalRedis()
    store = RedisApprovalStore(lambda: fake, prefix="t:", ttl_s=60)
    log = EventLog()
    executor = Executor(Registry(), approval=DeferredApproval(store), sink=log)
    record = await store.create(req())
    fake.clock.advance(61)
    with pytest.raises(ApprovalNotFound):
        await executor.resume(record.approval_id, True, decided_by="ann")
    (event,) = log.of_type("approval.expired")
    assert dict(event.data) == {"tool": "", "approval_id": record.approval_id, "outcome": "ttl"}
    with pytest.raises(ApprovalNotFound):
        await executor.resume("apr_unknown", True, decided_by="ann")
    assert len(log.of_type("approval.expired")) == 1  # unknown ids are not expiries
