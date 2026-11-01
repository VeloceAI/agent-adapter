"""Cross-worker `Idempotency-Key` replay through an `IdempotencyStore` (fix round 1)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from agent_adapter.approval.store import MemoryApprovalStore
from agent_adapter.cache.memory import MemoryCache
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.registry import Registry
from agent_adapter.server.idempotency import (
    IdempotencyStore,
    MemoryIdempotencyStore,
    RedisIdempotencyStore,
    Stored,
)
from tests.fakes.cache import BrokenCache, DictCache
from tests.fakes.clock import FakeClock
from tests.server.fake_redis import FakeRedis
from tests.server.helpers import call, invoke, make_registry, make_service

KEY = {"Idempotency-Key": "k-1"}
EMAIL = {"to": "a@example.com", "body": "hi"}


def blocking_registry(log: list[str], gate: asyncio.Event) -> Registry:
    reg = Registry()

    @reg.tool
    async def pay(amount: int) -> str:
        """Pay (waits for the gate)."""
        log.append(f"pay {amount}")
        await gate.wait()
        return "paid"

    return reg


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> IdempotencyStore:
    if request.param == "memory":
        return MemoryIdempotencyStore()
    fake = FakeRedis()
    return RedisIdempotencyStore(lambda: fake)


async def test_two_workers_sharing_a_store_run_a_call_once(store: IdempotencyStore):
    log: list[str] = []
    gate = asyncio.Event()
    registry = blocking_registry(log, gate)
    one = make_service(registry, idempotency=store)
    two = make_service(registry, idempotency=store)
    first = asyncio.ensure_future(invoke(one, "pay", {"amount": 5}, headers=KEY))
    await asyncio.sleep(0.01)
    racing = await invoke(two, "pay", {"amount": 5}, headers=KEY)
    assert racing.status == 409 and racing.body["type"].endswith(":request-in-progress")
    other = await invoke(two, "pay", {"amount": 6}, headers=KEY)
    assert other.status == 400  # same key, another request: refused even while in flight
    gate.set()
    assert (await first).status == 200
    replay = await invoke(two, "pay", {"amount": 5}, headers=KEY)
    assert (replay.status, replay.headers["Idempotent-Replayed"]) == (200, "true")
    assert log == ["pay 5"]


async def test_a_consumed_approval_is_never_parked_again(store: IdempotencyStore):
    log: list[str] = []
    approvals = MemoryApprovalStore()
    registry = make_registry(log)
    one = make_service(registry, idempotency=store, approvals=approvals)
    two = make_service(registry, idempotency=store, approvals=approvals)
    parked = await invoke(one, "send_email", EMAIL, headers=KEY)
    assert parked.status == 202
    decided = await call(
        two, "POST", f"/approvals/{parked.body['approval_id']}/decide", body={"approved": True}
    )
    assert decided.status == 200
    again = await invoke(one, "send_email", EMAIL, headers=KEY)
    assert (again.status, again.headers["Idempotent-Replayed"]) == (200, "true")
    assert again.body["result"]["sent"] is True
    assert log == ["send a@example.com"]
    assert len(await approvals.list()) == 1  # no second pending record


async def test_store_semantics(store: IdempotencyStore):
    first = await store.claim("k", "fp", 30)
    assert first.state == "new" and first.token
    busy = await store.claim("k", "fp2", 30)
    assert (busy.state, busy.fingerprint) == ("in_progress", "fp")
    stored = Stored("fp", 200, {"a": 1}, headers={"X": "1"})
    assert not await store.complete("k", "wrong-token", stored, 60)
    assert await store.complete("k", first.token, stored, 60)
    done = await store.claim("k", "fp", 30)
    assert (done.state, done.stored) == ("done", stored)
    assert not await store.update("k", Stored("other", 200, {}), 60)
    assert await store.update("k", Stored("fp", 201, {"b": 2}), 60)
    assert (await store.claim("k", "fp", 30)).stored == Stored("fp", 201, {"b": 2})
    assert not await store.update("missing", stored, 60)
    lease = await store.claim("k2", "fp", 30)
    await store.release("k2", "wrong-token")
    assert (await store.claim("k2", "fp", 30)).state == "in_progress"
    await store.release("k2", lease.token)
    assert (await store.claim("k2", "fp", 30)).state == "new"


async def test_memory_leases_and_entries_expire():
    clock = FakeClock()
    store = MemoryIdempotencyStore(clock=clock)
    lost = await store.claim("k", "fp", 10)
    clock.advance(11)
    retaken = await store.claim("k", "fp", 10)
    assert retaken.state == "new" and retaken.token != lost.token
    assert not await store.complete("k", lost.token, Stored("fp", 200, {}), 60)
    assert await store.complete("k", retaken.token, Stored("fp", 200, {}), 60)
    clock.advance(61)
    assert (await store.claim("k", "fp", 10)).state == "new"


async def test_memory_store_evicts_done_entries_first(caplog: pytest.LogCaptureFixture):
    store = MemoryIdempotencyStore(max_entries=2)
    a = await store.claim("a", "fp", 30)
    await store.complete("a", a.token, Stored("fp", 200, {}), 60)
    await store.claim("b", "fp", 30)
    await store.claim("c", "fp", 30)  # evicts the done "a", keeps in-flight "b"
    assert (await store.claim("a", "fp", 30)).state == "new"
    assert (await store.claim("b", "fp", 30)).state == "in_progress"
    await store.claim("d", "fp", 30)  # only leases left: kept, over the cap
    assert "idempotency leases exceed" in caplog.text


async def test_redis_values_and_expiry():
    fake = FakeRedis()
    store = RedisIdempotencyStore(lambda: fake, prefix="p:")
    claim = await store.claim("k", "fp", 1.5)
    assert fake.data["p:k"][0].startswith(b"i:") and fake.data["p:k"][1] == 1500
    await store.complete("k", claim.token, Stored("fp", 200, {"a": 1}), 2)
    assert fake.data["p:k"][0].startswith(b"d:fp:")
    fake.data["p:k"] = (b"garbage", None)
    unreadable = await store.claim("k", "fp", 1)
    assert (unreadable.state, unreadable.fingerprint) == ("in_progress", "")


async def test_redis_claim_rereads_a_key_that_vanished_and_fails_closed():
    for vanish, fingerprint in ((1, "fp0"), (3, "")):
        fake = FakeRedis(vanish=vanish)
        fake.data["k"] = (b"i:token:fp0", 1000)
        store = RedisIdempotencyStore(lambda fake=fake: fake)  # type: ignore[misc]
        claim = await store.claim("k", "fp", 1)
        assert (claim.state, claim.fingerprint) == ("in_progress", fingerprint)
        await store.aclose()
        assert fake.closed


async def test_a_failing_store_fails_closed_with_503(caplog: pytest.LogCaptureFixture):
    log: list[str] = []
    store = RedisIdempotencyStore(lambda: FakeRedis(broken=True))
    service = make_service(make_registry(log), idempotency=store)
    reply = await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)
    assert reply.status == 503 and reply.headers["Retry-After"]
    assert log == [] and "idempotency store failed: ConnectionError" in caplog.text
    assert (await invoke(service, "add", {"a": 1, "b": 1})).status == 200  # no key: unaffected


async def test_a_lost_lease_or_a_failing_completion_still_answers(
    caplog: pytest.LogCaptureFixture,
):
    class Losing(MemoryIdempotencyStore):
        async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
            return False

    service = make_service(idempotency=Losing())
    assert (await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)).status == 200
    assert "lease was lost" in caplog.text

    class Failing(MemoryIdempotencyStore):
        async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
            raise ConnectionError("down")

        async def release(self, key: str, token: str) -> None:
            raise ConnectionError("down")

        async def update(self, key: str, stored: Stored, ttl_s: float) -> bool:
            raise ConnectionError("down")

    service = make_service(idempotency=Failing())
    assert (await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)).status == 200
    charge = {"Idempotency-Key": "k-charge"}
    await invoke(service, "charge", {"amount": 1}, extra={"run_id": "r"})
    assert (
        await invoke(service, "charge", {"amount": 1}, extra={"run_id": "r"}, headers=charge)
    ).status == 409  # release fails: logged only
    parked = await invoke(service, "send_email", EMAIL, headers={"Idempotency-Key": "k-m"})
    decided = await call(
        service,
        "POST",
        f"/approvals/{parked.body['approval_id']}/decide",
        body={"approved": True},
    )
    assert decided.status == 200


async def test_refusals_before_execution_are_not_replayed():
    log: list[str] = []
    service = make_service(make_registry(log))
    await invoke(service, "charge", {"amount": 1}, extra={"run_id": "r"})
    key = {"Idempotency-Key": "k-again"}
    limited = await invoke(service, "charge", {"amount": 1}, extra={"run_id": "r"}, headers=key)
    again = await invoke(service, "charge", {"amount": 1}, extra={"run_id": "r"}, headers=key)
    assert (limited.status, again.status) == (409, 409)
    assert "Idempotent-Replayed" not in again.headers  # evaluated again, not replayed


def test_a_plain_cache_port_is_wrapped_with_a_warning(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    make_service(idempotency=MemoryCache())
    assert "best effort" not in caplog.text
    make_service(idempotency=DictCache())
    assert "best effort" in caplog.text
    with pytest.raises(ToolUsageError, match="idempotency"):
        make_service(idempotency=object())


async def test_the_cache_port_wrapper_still_replays_and_refuses_duplicates():
    log: list[str] = []
    gate = asyncio.Event()
    service = make_service(blocking_registry(log, gate), idempotency=MemoryCache())
    first = asyncio.ensure_future(invoke(service, "pay", {"amount": 1}, headers=KEY))
    await asyncio.sleep(0.01)
    assert (await invoke(service, "pay", {"amount": 1}, headers=KEY)).status == 409
    assert (await invoke(service, "pay", {"amount": 2}, headers=KEY)).status == 400
    gate.set()
    await first
    assert (await invoke(service, "pay", {"amount": 1}, headers=KEY)).headers[
        "Idempotent-Replayed"
    ] == "true"
    assert log == ["pay 1"]


async def test_the_cache_port_wrapper_treats_a_broken_cache_as_a_miss():
    service = make_service(idempotency=BrokenCache())
    assert (await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)).status == 200
    assert (await invoke(service, "add", {"a": 1, "b": 1}, headers=KEY)).status == 200
    parked = await invoke(service, "send_email", EMAIL, headers={"Idempotency-Key": "k-m"})
    decided = await call(
        service,
        "POST",
        f"/approvals/{parked.body['approval_id']}/decide",
        body={"approved": True},
    )
    assert decided.status == 200


def test_stores_are_idempotency_stores():
    assert isinstance(MemoryIdempotencyStore(), IdempotencyStore)
    assert isinstance(RedisIdempotencyStore(FakeRedis), IdempotencyStore)
    assert not isinstance(MemoryCache(), IdempotencyStore)
    _: Any = None


async def test_redis_unreadable_done_entries_and_clients_without_aclose():
    class Bare(FakeRedis):
        aclose = None  # type: ignore[assignment]

    fake = Bare()
    fake.data["k"] = (b"d:fp:not json", None)
    store = RedisIdempotencyStore(lambda: fake)
    assert (await store.claim("k", "fp", 1)).state == "in_progress"
    await store.release("k", "")  # no token: nothing to release
    assert not await store.complete("k", "", Stored("fp", 200, {}), 1)
    await store.aclose()


class RecordingLeases(MemoryIdempotencyStore):
    def __init__(self) -> None:
        super().__init__()
        self.leases: list[float] = []

    async def claim(self, key: str, fingerprint: str, lease_s: float) -> Any:
        self.leases.append(lease_s)
        return await super().claim(key, fingerprint, lease_s)


async def test_a_lease_covers_every_attempt_of_a_slow_tool():
    reg = Registry()

    @reg.tool(timeout=400.0, retries=1)
    async def slow() -> str:
        """Up to two attempts of 400 s each."""
        return "done"

    @reg.tool(timeout=1.0)
    async def quick() -> str:
        """Bounded well under the default lease."""
        return "done"

    @reg.tool
    async def unbounded() -> str:
        """No timeout: the default lease applies."""
        return "done"

    store = RecordingLeases()
    service = make_service(reg, idempotency=store)
    for i, name in enumerate(("slow", "quick", "unbounded")):
        await invoke(service, name, {}, headers={"Idempotency-Key": f"k{i}"})
    assert store.leases == [830.0, 300.0, 300.0]  # max(default, timeout x attempts + 30 s)
