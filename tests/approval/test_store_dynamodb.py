"""DynamoDBApprovalStore: configuration, client handling, TTL and failing closed on bad items."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agent_adapter.approval import dynamodb_items, dynamodb_store
from agent_adapter.approval.dynamodb_store import DEFAULT_TTL_S, DynamoDBApprovalStore
from agent_adapter.approval.persist import ApprovalStoreCorrupt
from agent_adapter.core.errors import ApprovalNotFound, ConfigError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from tests.approval.fake_dynamodb import FakeClientError, FakeDynamoDB

YES = ApprovalDecision(approved=True, decided_by="ana")


def req(call_id: str = "call_1", run_id: str = "run_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id=run_id, call_id=call_id, tool="pay", arguments={"amount": 5}
    )


def store_on(fake: Any, **options: Any) -> DynamoDBApprovalStore:
    return DynamoDBApprovalStore(lambda: fake, table="t", **options)


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def fake() -> FakeDynamoDB:
    return FakeDynamoDB()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    found = Clock(1_000_000.0)
    monkeypatch.setattr(dynamodb_store, "time", found)
    return found


def test_client_factory_must_be_callable() -> None:
    with pytest.raises(ConfigError, match="client_factory"):
        DynamoDBApprovalStore(FakeDynamoDB(), table="t")


@pytest.mark.parametrize("bad", ["", None, 3])
def test_table_must_be_a_non_empty_string(bad: Any) -> None:
    with pytest.raises(ConfigError, match="table"):
        DynamoDBApprovalStore(FakeDynamoDB, table=bad)


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan"), "60"])
def test_ttl_must_be_positive_finite_or_none(bad: Any) -> None:
    with pytest.raises(ConfigError, match="ttl_s"):
        DynamoDBApprovalStore(FakeDynamoDB, table="t", ttl_s=bad)


def test_table_definition_keys_and_indexes() -> None:
    table = DynamoDBApprovalStore.table_definition("approvals")
    assert table["TableName"] == "approvals" and table["BillingMode"] == "PAY_PER_REQUEST"
    assert table["KeySchema"] == [{"AttributeName": "pk", "KeyType": "HASH"}]
    assert {g["IndexName"] for g in table["GlobalSecondaryIndexes"]} == {"by_run", "by_status"}


async def test_a_context_factory_is_entered_once_and_exited_on_aclose(fake) -> None:
    class Context:
        entered = exited = 0

        async def __aenter__(self) -> FakeDynamoDB:
            Context.entered += 1
            return fake

        async def __aexit__(self, *args: Any) -> None:
            Context.exited += 1

    store = DynamoDBApprovalStore(Context, table="t")
    await store.create(req())
    assert await store.find("run_1", "call_1") is not None
    await store.aclose()
    assert (Context.entered, Context.exited) == (1, 1)


async def test_a_plain_client_is_closed_on_aclose(fake) -> None:
    store = store_on(fake)
    await store.list()
    await store.aclose()
    assert fake.closed


async def test_a_factory_returning_something_else_is_a_config_error() -> None:
    store = DynamoDBApprovalStore(object, table="t")
    with pytest.raises(ConfigError, match="client_factory"):
        await store.get("x")


async def test_records_expire_ttl_after_creation(fake, clock) -> None:
    store = store_on(fake, ttl_s=60)
    record = await store.create(req())
    stored = fake.items[dynamodb_items.rec_pk(record.approval_id)]
    assert stored["expires_at"] == {"N": str(1_000_060)}
    clock.now += 61
    with pytest.raises(ApprovalNotFound):
        await store.get(record.approval_id)
    with pytest.raises(ApprovalNotFound):
        await store.decide(record.approval_id, YES)
    assert await store.find("run_1", "call_1") is None
    assert await store.list() == []
    again = await store.create(req())  # the expired marker is replaced
    assert again.approval_id != record.approval_id
    assert await store.find("run_1", "call_1") == again


async def test_default_ttl_and_no_ttl(fake, clock) -> None:
    assert store_on(fake).ttl_s == DEFAULT_TTL_S
    record = await store_on(fake, ttl_s=None).create(req())
    assert "expires_at" not in fake.items[dynamodb_items.rec_pk(record.approval_id)]


async def test_a_marker_pointing_at_another_call_fails_closed(fake) -> None:
    store = store_on(fake)
    other = await store.create(req("other"))
    marker = dynamodb_items.open_pk("run_1", "call_1")
    fake.items[marker] = {"pk": {"S": marker}, "aid": {"B": other.approval_id.encode()}}
    with pytest.raises(ApprovalStoreCorrupt, match="another call"):
        await store.find("run_1", "call_1")


async def test_a_tampered_item_fails_closed(fake) -> None:
    store = store_on(fake)
    record = await store.create(req())
    fake.items[dynamodb_items.rec_pk(record.approval_id)]["st"] = {"S": "approved"}
    with pytest.raises(ApprovalStoreCorrupt, match="disagrees"):
        await store.get(record.approval_id)


def test_an_item_without_a_valid_seq_fails_closed() -> None:
    with pytest.raises(ApprovalStoreCorrupt, match="seq"):
        dynamodb_items.seq_of({"aid": {"B": b"apr_1"}, "seq": {"S": "x"}})


async def test_backend_errors_are_logged_and_propagate(caplog) -> None:
    store = store_on(FakeDynamoDB(broken=True))
    with caplog.at_level(logging.ERROR), pytest.raises(FakeClientError):
        await store.get("x")
    assert "dynamodb approval store get failed: code=InternalServerError" in caplog.text


async def test_a_swap_that_keeps_losing_fails_closed(fake, monkeypatch) -> None:
    store = store_on(fake)
    record = await store.create(req())

    async def lost(*args: Any) -> bool:
        return False

    monkeypatch.setattr(store, "_swap", lost)
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.decide(record.approval_id, YES)


async def test_consume_that_keeps_losing_fails_closed(fake, monkeypatch) -> None:
    store = store_on(fake)
    record = await store.create(req())
    await store.decide(record.approval_id, YES)

    async def lost(*args: Any) -> bool:
        return False

    monkeypatch.setattr(store, "_swap", lost)
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.consume(record.approval_id)


async def test_a_create_that_keeps_losing_the_transaction_fails_closed(fake, monkeypatch) -> None:
    store = store_on(fake)

    async def lost(*args: Any) -> None:
        raise dynamodb_store._LostRace

    monkeypatch.setattr(store, "_put_pair", lost)
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.create(req())


async def test_list_pages_and_merges_statuses_newest_first(fake) -> None:
    store = store_on(fake)
    made = [await store.create(req(f"c{i}")) for i in range(130)]
    for record in made[::2]:
        await store.decide(record.approval_id, YES)
    newest = [r.approval_id for r in reversed(made)]
    assert [r.approval_id for r in await store.list(limit=150)] == newest
    assert len(await store.list(status="approved", limit=200)) == 65


def test_error_code_of_other_exceptions_is_empty() -> None:
    assert dynamodb_items.error_code(ValueError("x")) == ""
    assert dynamodb_items.error_code(FakeClientError("Throttling")) == "Throttling"


def test_raw_reads_bytearray_and_unknown_types() -> None:
    item = {"a": {"B": bytearray(b"x")}, "b": {"BOOL": True}}
    assert dynamodb_items.raw(item, "a") == b"x"
    assert dynamodb_items.raw(item, "b") is None


async def test_a_marker_whose_record_is_gone_is_replaced(fake) -> None:
    store = store_on(fake)
    marker = dynamodb_items.open_pk("run_1", "call_1")
    fake.items[marker] = {"pk": {"S": marker}, "aid": {"B": b"apr_deleted"}}
    assert await store.find("run_1", "call_1") is None
    record = await store.create(req())
    assert await store.find("run_1", "call_1") == record
