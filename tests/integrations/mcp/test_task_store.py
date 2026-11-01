"""TaskStore backends: memory and Redis keep states and signals until their TTL."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.mcp._task_store import (
    MemoryTaskStore,
    RedisTaskStore,
    TaskStore,
    TaskStoreCorrupt,
)
from tests.fakes.clock import FakeClock


def fake_redis() -> tuple[RedisTaskStore, Any]:
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    return RedisTaskStore(lambda: fakeredis.FakeAsyncRedis(server=server), prefix="p:"), server


def memory() -> TaskStore:
    return MemoryTaskStore()


def redis() -> TaskStore:
    return fake_redis()[0]


@pytest.mark.parametrize("make", [memory, redis], ids=["memory", "redis"])
async def test_a_store_keeps_states_and_signals(make: Callable[[], TaskStore]) -> None:
    store = make()
    assert isinstance(store, TaskStore)
    assert await store.get("t") is None
    await store.put("t", {"task": {"status": "working"}, "n": 1.5}, 60)
    await store.put("t", {"task": {"status": "completed"}}, 60)
    assert await store.get("t") == {"task": {"status": "completed"}}
    assert await store.flagged("t", ["cancel", "answered:a"]) == set()
    await store.flag("t", "cancel", 60)
    assert await store.flagged("t", ["cancel", "answered:a"]) == {"cancel"}
    assert await store.flagged("t", []) == set()
    await store.delete("t")
    assert await store.get("t") is None
    if isinstance(store, RedisTaskStore):
        await store.aclose()


def test_memory_entries_expire_and_the_oldest_go_past_the_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import asyncio

    clock = FakeClock()
    store = MemoryTaskStore(max_entries=2, clock=clock)

    async def scenario() -> None:
        await store.put("a", {"x": 1}, 10)
        await store.flag("a", "cancel", 1)
        clock.advance(2)
        assert await store.flagged("a", ["cancel"]) == set()
        await store.put("b", {"x": 2}, 10)
        await store.put("c", {"x": 3}, 10)
        assert await store.get("a") is None and await store.get("c") == {"x": 3}
        clock.advance(20)
        assert await store.get("c") is None

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert "dropped the oldest" in caplog.text


def test_memory_store_copies_states_and_refuses_bad_limits() -> None:
    import asyncio

    store = MemoryTaskStore()
    state: dict[str, Any] = {"task": {"status": "working"}}

    async def scenario() -> None:
        await store.put("t", state, 10)
        state["task"]["status"] = "changed"
        got = await store.get("t")
        assert got == {"task": {"status": "working"}}

    asyncio.run(scenario())
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ConfigError):
            MemoryTaskStore(max_entries=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        asyncio.run(store.put("t", {"x": float("nan")}, 10))


async def test_redis_keys_use_the_prefix_and_a_ttl() -> None:
    store, server = fake_redis()
    await store.put("t", {"a": 1}, 1.5)
    await store.flag("t", "cancel", 2)
    import fakeredis

    client = fakeredis.FakeAsyncRedis(server=server)
    assert await client.get("p:t:t") == b'{"a":1}'
    assert 0 < await client.pttl("p:t:t") <= 1500
    assert await client.get("p:f:t:cancel") == b"1"
    await client.aclose()
    await store.aclose()


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]"])
async def test_an_unreadable_redis_state_is_corrupt(
    raw: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    store, server = fake_redis()
    import fakeredis

    client = fakeredis.FakeAsyncRedis(server=server)
    await client.set("p:t:t", raw)
    with caplog.at_level(logging.ERROR), pytest.raises(TaskStoreCorrupt):
        await store.get("t")
    assert "unreadable state" in caplog.text and "not json" not in caplog.text
    await client.aclose()
    await store.aclose()


def test_a_redis_store_needs_a_factory() -> None:
    with pytest.raises(ConfigError, match="client_factory"):
        RedisTaskStore("redis://x")  # type: ignore[arg-type]


@pytest.mark.parametrize("ttl", [float("inf"), float("nan"), -1.0, 0.0])
async def test_a_store_refuses_a_ttl_that_is_not_finite_and_positive(ttl: float) -> None:
    from agent_adapter.integrations.mcp._task_store import _ms

    with pytest.raises(ConfigError, match="ttl_s"):
        _ms(ttl)
    with pytest.raises(ConfigError, match="ttl_s"):
        await MemoryTaskStore().put("t", {}, ttl)


def test_an_outage_log_warns_once_then_reports_the_recovery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agent_adapter.integrations.mcp._task_store import OutageLog

    outage = OutageLog(logging.getLogger("test.outage"))
    with caplog.at_level(logging.INFO, logger="test.outage"):
        for _ in range(5):
            outage.failed("t1", ConnectionError("x"), "store unread")
        outage.ok("t1", "store back")
        outage.ok("t1", "store back")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert caplog.text.count("store back") == 1
