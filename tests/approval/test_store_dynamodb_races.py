"""DynamoDBApprovalStore: lost races versus real failures, backoff, and the create marker check."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agent_adapter.approval import dynamodb_items, dynamodb_store
from agent_adapter.approval.dynamodb_store import DynamoDBApprovalStore
from agent_adapter.approval.persist import ApprovalStoreCorrupt
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from tests.approval.fake_dynamodb import FakeClientError, FakeDynamoDB

YES = ApprovalDecision(approved=True, decided_by="ana")
CANCELLED = "TransactionCanceledException"


def req(call_id: str = "call_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="run_1", call_id=call_id, tool="pay", arguments={"amount": 5}
    )


def store_on(fake: FakeDynamoDB) -> DynamoDBApprovalStore:
    return DynamoDBApprovalStore(lambda: fake, table="t")


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []

    async def sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(dynamodb_store, "_sleep", sleep)
    return slept


@pytest.mark.parametrize(
    ("error", "race"),
    [
        (FakeClientError("ConditionalCheckFailedException"), True),
        (FakeClientError("TransactionConflictException"), True),
        (FakeClientError(CANCELLED, ["None", "ConditionalCheckFailed"]), True),
        (FakeClientError(CANCELLED, ["TransactionConflict", "None"]), True),
        (FakeClientError(CANCELLED, ["None", "ThrottlingError"]), False),
        (FakeClientError(CANCELLED, ["ConditionalCheckFailed", "ValidationError"]), False),
        (FakeClientError(CANCELLED, ["ProvisionedThroughputExceeded", "None"]), False),
        (FakeClientError(CANCELLED, ["None", "None"]), False),
        (FakeClientError(CANCELLED), False),
        (FakeClientError("ThrottlingException"), False),
        (ValueError("x"), False),
    ],
)
def test_only_condition_and_conflict_reasons_are_lost_races(error: Exception, race: bool) -> None:
    assert dynamodb_items.lost_race(error) is race


def test_cancellation_reasons_fall_back_to_the_message() -> None:
    error = FakeClientError(CANCELLED)
    error.response["Error"]["Message"] = "Transaction cancelled, ... [None, ConditionalCheckFailed]"
    assert dynamodb_items.cancellation_reasons(error) == ["None", "ConditionalCheckFailed"]
    assert dynamodb_items.lost_race(error) is True


async def test_a_throttled_transaction_is_logged_and_raised_not_retried(caplog, sleeps) -> None:
    fake = FakeDynamoDB()
    store = store_on(fake)

    async def throttled(**kwargs: Any) -> Any:
        fake.calls.append("transact_write_items")
        raise FakeClientError(CANCELLED, ["None", "ThrottlingError"])

    fake.transact_write_items = throttled  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR), pytest.raises(FakeClientError):
        await store.create(req())
    assert fake.calls.count("transact_write_items") == 1
    assert "dynamodb approval store create failed" in caplog.text
    assert "ThrottlingError" in caplog.text
    assert "kept changing" not in caplog.text and sleeps == []


async def test_a_throttled_update_is_raised_not_reported_as_a_lost_race(caplog) -> None:
    fake = FakeDynamoDB()
    store = store_on(fake)
    record = await store.create(req())

    async def throttled(**kwargs: Any) -> Any:
        raise FakeClientError("ProvisionedThroughputExceededException")

    fake.update_item = throttled  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR), pytest.raises(FakeClientError):
        await store.decide(record.approval_id, YES)
    assert "ProvisionedThroughputExceededException" in caplog.text


async def test_lost_races_back_off_with_growing_jittered_delays(monkeypatch, sleeps) -> None:
    fake = FakeDynamoDB()
    store = store_on(fake)
    record = await store.create(req())

    async def lost(*args: Any) -> bool:
        return False

    monkeypatch.setattr(store, "_swap", lost)
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.decide(record.approval_id, YES)
    assert len(sleeps) == dynamodb_store._ATTEMPTS - 1
    for attempt, delay in enumerate(sleeps):
        cap = min(dynamodb_store._BACKOFF_CAP_S, dynamodb_store._BACKOFF_BASE_S * 2**attempt)
        assert 0 <= delay <= cap


async def test_a_lost_create_race_is_logged_at_debug_and_retried(caplog, sleeps) -> None:
    fake = FakeDynamoDB()
    store = store_on(fake)
    real = fake.transact_write_items
    calls = 0

    async def once_lost(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FakeClientError(CANCELLED, ["ConditionalCheckFailed", "None"])
        return await real(**kwargs)

    fake.transact_write_items = once_lost  # type: ignore[method-assign]
    with caplog.at_level(logging.DEBUG, logger=dynamodb_store.__name__):
        record = await store.create(req())
    assert await store.find("run_1", "call_1") == record
    assert "lost a conditional write race" in caplog.text and len(sleeps) == 1


async def test_create_refuses_a_marker_pointing_at_another_call() -> None:
    fake = FakeDynamoDB()
    store = store_on(fake)
    other = await store.create(req("other"))
    marker = dynamodb_items.open_pk("run_1", "call_1")
    fake.items[marker] = {"pk": {"S": marker}, "aid": {"B": other.approval_id.encode()}}
    with pytest.raises(ApprovalStoreCorrupt, match="another call"):
        await store.create(req())


def test_cancellation_reasons_of_other_exceptions_are_empty() -> None:
    assert dynamodb_items.cancellation_reasons(ValueError("x")) == []
    error = FakeClientError(CANCELLED)
    error.response["Error"] = "not a dict"
    assert dynamodb_items.cancellation_reasons(error) == []
