"""RedisApprovalStore: configuration, TTL, client handling and failing closed on bad hashes."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agent_adapter.approval import redis_scripts
from agent_adapter.approval.persist import ApprovalStoreCorrupt, encode
from agent_adapter.approval.redis_store import DEFAULT_TTL_S, RedisApprovalStore
from agent_adapter.core.errors import ApprovalNotFound, ConfigError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.runtime import run_sync
from tests.approval.fake_redis_store import FakeApprovalRedis

YES = ApprovalDecision(approved=True, decided_by="ana")
P = b"t:"


def req(call_id: str = "call_1") -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="", run_id="run_1", call_id=call_id, tool="pay", arguments={"amount": 5}
    )


def store_on(fake: Any, **options: Any) -> RedisApprovalStore:
    return RedisApprovalStore(lambda: fake, prefix="t:", **options)


@pytest.fixture
def fake() -> FakeApprovalRedis:
    return FakeApprovalRedis()


def test_client_factory_must_be_callable():
    with pytest.raises(ConfigError, match="client_factory"):
        RedisApprovalStore(FakeApprovalRedis())


def test_prefix_must_be_a_string():
    with pytest.raises(ConfigError, match="prefix"):
        RedisApprovalStore(FakeApprovalRedis, prefix=b"x:")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "aa", "aa:approval", "aa: "])
def test_prefix_must_be_non_empty_and_end_in_a_colon(bad):
    with pytest.raises(ConfigError, match="prefix"):
        RedisApprovalStore(FakeApprovalRedis, prefix=bad)


def test_the_swap_script_compares_before_it_writes():
    # The Python fake mirrors the scripts; this pins the compare that makes consume exactly-once
    # (the redis-lua conformance run proves it under real Lua with a yielding client).
    compare = "if cur[1] ~= ARGV[1] then return 0 end"
    assert compare in redis_scripts.SWAP
    assert redis_scripts.SWAP.index(compare) < redis_scripts.SWAP.index("redis.call('HSET'")


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan"), True, "5"])
def test_ttl_must_be_a_finite_positive_number_or_none(bad):
    with pytest.raises(ConfigError, match="ttl_s"):
        RedisApprovalStore(FakeApprovalRedis, ttl_s=bad)


def test_defaults():
    store = RedisApprovalStore(FakeApprovalRedis)
    assert (store.prefix, store.ttl_s) == ("aa:approval:", DEFAULT_TTL_S)


async def test_records_expire_ttl_after_creation_and_leave_the_indexes(fake):
    store = store_on(fake, ttl_s=60)
    old = await store.create(req())
    fake.clock.advance(50)
    await store.decide(old.approval_id, YES)  # a decision does not extend the record's life
    fake.clock.advance(11)
    with pytest.raises(ApprovalNotFound):
        await store.get(old.approval_id)
    assert await store.find("run_1", "call_1") is None
    assert await store.list() == [] and await store.list(status="approved") == []
    new = await store.create(req())  # the call reopens; expired ids leave every index
    assert new.approval_id != old.approval_id
    for index in (b"all", b"status:pending", b"status:approved", b"exp"):
        assert old.approval_id.encode() not in fake.zsets.get(P + index, {})
    assert [r.approval_id for r in await store.list(run_id="run_1")] == [new.approval_id]


async def test_without_a_ttl_records_never_expire(fake):
    store = store_on(fake, ttl_s=None)
    record = await store.create(req())
    fake.clock.advance(10 * DEFAULT_TTL_S)
    assert await store.get(record.approval_id) == record
    assert fake.expires == {} and P + b"exp" not in fake.zsets


def _counting() -> tuple[RedisApprovalStore, list[FakeApprovalRedis]]:
    made: list[FakeApprovalRedis] = []

    def factory() -> FakeApprovalRedis:
        made.append(FakeApprovalRedis())
        return made[-1]

    return RedisApprovalStore(factory), made


def test_one_client_per_event_loop_closed_with_its_loop():
    store, made = _counting()
    run_sync(store.list())
    run_sync(store.list())
    assert len(made) == 2 and all(client.closed for client in made)


async def test_aclose_closes_the_running_loops_client():
    store, made = _counting()
    await store.list()
    await store.list()
    await store.aclose()
    assert len(made) == 1 and made[0].closed


class _Bare(FakeApprovalRedis):
    """A client without `aclose` (closing it is a no-op)."""

    aclose = None  # type: ignore[assignment]


async def test_a_client_without_aclose_is_simply_dropped():
    store = store_on(_Bare())
    await store.list()
    await store.aclose()


class _Text(FakeApprovalRedis):
    """A client made with decode_responses=True: every reply is str."""

    async def hmget(self, name: bytes, keys: list[str]) -> list[Any]:
        return _text(await super().hmget(name, keys))

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        return _text(await super().eval(script, numkeys, *args))


def _text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, list):
        return [_text(item) for item in value]
    return value


async def test_a_decode_responses_client_still_works():
    store = store_on(_Text())
    record = await store.create(req())
    assert await store.create(req()) == record
    await store.decide(record.approval_id, YES)
    assert await store.find("run_1", "call_1") == await store.get(record.approval_id)
    assert await store.consume(record.approval_id) is True
    assert [r.approval_id for r in await store.list(status="approved")] == [record.approval_id]


async def test_backend_errors_propagate_and_are_logged_without_detail(caplog):
    store = store_on(FakeApprovalRedis(broken=True))
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ConnectionError):
            await store.create(req())
        with pytest.raises(ConnectionError):
            await store.get("apr_1")
    assert "redis approval store create failed (ConnectionError)" in caplog.text
    assert "redis approval store get failed (ConnectionError)" in caplog.text
    assert "redis down" not in caplog.text  # exception text stays at DEBUG


# --- corrupt or foreign hashes fail closed --------------------------------------------------


async def _decided(fake: FakeApprovalRedis) -> tuple[RedisApprovalStore, str, bytes]:
    store = store_on(fake)
    record = await store.create(req())
    await store.decide(record.approval_id, YES)
    return store, record.approval_id, P + b"rec:" + record.approval_id.encode()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (b"record", b"{not json"),
        (b"record", b'"a string"'),
        (b"status", b"pending"),
        (b"consumed", b"maybe"),
        (b"run_id", b"run_2"),
        (b"call_id", b"\xff\xfe"),
    ],
)
async def test_unreadable_hashes_raise_a_typed_error_and_are_never_consumed(fake, field, value):
    store, approval_id, rec = await _decided(fake)
    fake.hashes[rec][field] = value
    for call in (store.get(approval_id), store.consume(approval_id), store.list()):
        with pytest.raises(ApprovalStoreCorrupt):
            await call
    assert fake.hashes[rec][b"consumed"] != b"1"


async def test_a_hash_without_its_fields_fails_closed(fake):
    store, _, rec = await _decided(fake)
    del fake.hashes[rec][b"run_id"]
    with pytest.raises(ApprovalStoreCorrupt, match="disagrees"):
        await store.find("run_1", "call_1")
    with pytest.raises(ApprovalStoreCorrupt):
        await store.create(req())  # the call's open hash is unreadable: nothing new is stored


class _Odd(FakeApprovalRedis):
    """Script replies no RedisApprovalStore script returns."""

    def __init__(self, reply: Any) -> None:
        super().__init__()
        self.reply = reply

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        return self.reply


@pytest.mark.parametrize("reply", [[], [b"weird"], None])
async def test_an_unexpected_create_reply_fails_closed(reply):
    with pytest.raises(ApprovalStoreCorrupt, match="unexpected reply"):
        await store_on(_Odd(reply)).create(req())


async def test_a_row_of_the_wrong_shape_fails_closed():
    with pytest.raises(ApprovalStoreCorrupt, match="wrong shape"):
        await store_on(_Odd([b"apr_1", b"{}"])).find("run_1", "call_1")


async def test_a_record_that_is_not_text_fails_closed():
    row = [b"apr_1", 5, b"pending", b"0", b"run_1", b"call_1"]
    with pytest.raises(ApprovalStoreCorrupt, match="is unreadable"):
        await store_on(_Odd(row)).find("run_1", "call_1")


class _Restless(FakeApprovalRedis):
    """Every compare-and-set loses, as if another writer kept changing the record."""

    def _swap(self, keys: list[bytes], argv: list[bytes]) -> int:
        return 0


async def test_a_record_that_keeps_changing_fails_closed(caplog):
    fake = _Restless()
    store = store_on(fake)
    record = await store.create(req())
    with pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.decide(record.approval_id, YES)
    rec = P + b"rec:" + record.approval_id.encode()
    decided = (await store.get(record.approval_id)).model_copy(
        update={"status": "approved", "decision": YES}
    )
    fake.hashes[rec].update({b"record": encode(decided).encode(), b"status": b"approved"})
    with caplog.at_level(logging.ERROR), pytest.raises(ApprovalStoreCorrupt, match="kept changing"):
        await store.consume(record.approval_id)
    assert "kept changing under a swap" in caplog.text
