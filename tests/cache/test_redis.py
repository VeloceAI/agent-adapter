import asyncio
import logging
import sys

import pytest

redis = pytest.importorskip("redis")
from redis import exceptions as redis_exc  # noqa: E402

from agent_adapter.cache.keys import digest  # noqa: E402
from agent_adapter.cache.redis import RedisCache  # noqa: E402
from agent_adapter.core.errors import ConfigError, MissingDependency  # noqa: E402
from agent_adapter.core.ports import CachePort  # noqa: E402
from agent_adapter.runtime import run_sync  # noqa: E402


class FakeRedis:
    """The `redis.asyncio.Redis` subset RedisCache uses: get, set(ex=), delete, aclose."""

    def __init__(self, *, error: Exception | None = None, decode: bool = False) -> None:
        self.data: dict[str, bytes] = {}
        self.calls: list[tuple] = []
        self.error = error
        self.decode = decode
        self.closed = False
        self.loop = asyncio.get_running_loop()

    def _check(self):
        assert asyncio.get_running_loop() is self.loop, "client used from another loop"
        if self.error is not None:
            raise self.error

    async def get(self, key):
        self.calls.append(("get", key))
        self._check()
        value = self.data.get(key)
        return value.decode() if self.decode and value is not None else value

    async def set(self, key, value, ex=None):
        self.calls.append(("set", key, value, ex))
        self._check()
        self.data[key] = value
        return True

    async def delete(self, *keys):
        self.calls.append(("delete", *keys))
        self._check()
        for key in keys:
            self.data.pop(key, None)
        return 1

    async def aclose(self):
        self.closed = True


class Factory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.made: list[FakeRedis] = []

    def __call__(self):
        client = FakeRedis(**self.kwargs)
        self.made.append(client)
        return client


def test_redis_cache_is_a_cache_port():
    assert isinstance(RedisCache(client_factory=Factory()), CachePort)


async def test_round_trip_with_prefix_and_ceiled_ttl():
    factory = Factory()
    cache = RedisCache(client_factory=factory, prefix="app:")
    await cache.set("k", b"v", ttl=1.2)
    await cache.set("n", b"forever")
    assert await cache.get("k") == b"v"
    await cache.delete("k")
    assert await cache.get("k") is None
    client = factory.made[0]
    assert ("set", "app:k", b"v", 2) in client.calls
    assert ("set", "app:n", b"forever", None) in client.calls
    assert ("delete", "app:k") in client.calls
    assert len(factory.made) == 1


async def test_non_positive_ttl_deletes_the_key():
    factory = Factory()
    cache = RedisCache(client_factory=factory)
    await cache.set("k", b"v")
    await cache.set("k", b"w", ttl=0)
    assert await cache.get("k") is None
    assert factory.made[0].calls[1] == ("delete", "k")


async def test_string_values_from_a_decoding_client_are_encoded():
    factory = Factory(decode=True)
    cache = RedisCache(client_factory=factory)
    await cache.set("k", "é".encode())
    assert await cache.get("k") == "é".encode()


@pytest.mark.parametrize(
    "error",
    [
        redis_exc.ConnectionError("down"),
        redis_exc.TimeoutError("slow"),
        redis_exc.ResponseError("x"),
    ],
)
async def test_redis_errors_are_logged_misses_and_no_ops(error, caplog):
    cache = RedisCache(client_factory=Factory(error=error))
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.redis"):
        assert await cache.get("k") is None
        await cache.set("k", b"v", ttl=5)
        await cache.delete("k")
    assert len(caplog.records) == 3
    assert all(r.levelno == logging.WARNING for r in caplog.records)
    assert "redis get failed" in caplog.records[0].getMessage()


async def test_other_exceptions_propagate():
    cache = RedisCache(client_factory=Factory(error=KeyError("bug")))
    with pytest.raises(KeyError):
        await cache.get("k")


async def test_unexpected_value_type_is_a_miss(caplog):
    class OddRedis(FakeRedis):
        async def get(self, key):
            return 42

    cache = RedisCache(client_factory=lambda: OddRedis())
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.redis"):
        assert await cache.get("k") is None
    assert "unexpected int" in caplog.text


def test_one_client_per_event_loop_closed_with_its_loop():
    factory = Factory()
    cache = RedisCache(client_factory=factory)

    async def work():
        await cache.set("k", b"v")
        return await cache.get("k")

    assert run_sync(work) == b"v"
    assert run_sync(work) == b"v"
    assert len(factory.made) == 2
    assert factory.made[0].loop is not factory.made[1].loop
    assert all(client.closed for client in factory.made)


async def test_aclose_closes_this_loops_client_and_tolerates_errors(caplog):
    class FailingClose(FakeRedis):
        async def aclose(self):
            raise redis_exc.ConnectionError("gone")

    factory = Factory()
    cache = RedisCache(client_factory=factory)
    await cache.aclose()  # nothing created yet
    await cache.get("k")
    await cache.aclose()
    assert factory.made[0].closed
    await cache.get("k")
    assert len(factory.made) == 2  # a fresh client after aclose

    broken = RedisCache(client_factory=lambda: FailingClose())
    await broken.get("k")
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.redis"):
        await broken.aclose()
    assert "closing the redis client failed" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"url": "redis://x", "client_factory": Factory()}],
)
def test_exactly_one_of_url_or_client_factory(kwargs):
    with pytest.raises(ConfigError, match="exactly one"):
        RedisCache(**kwargs)


@pytest.mark.parametrize(("name", "value"), [("retries", -1), ("socket_timeout", 0)])
def test_bad_retry_settings_are_rejected(name, value):
    with pytest.raises(ConfigError, match=name):
        RedisCache("redis://localhost", **{name: value})


async def test_url_client_is_built_with_timeouts_and_retries(monkeypatch):
    import redis.asyncio as aioredis

    seen = {}

    def from_url(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return FakeRedis()

    monkeypatch.setattr(aioredis.Redis, "from_url", staticmethod(from_url))
    cache = RedisCache("redis://user:secret@localhost:6379/0", socket_timeout=2.5, retries=4)
    await cache.set("k", b"v")
    assert seen["url"] == "redis://user:secret@localhost:6379/0"
    assert seen["decode_responses"] is False
    assert seen["socket_timeout"] == 2.5
    assert seen["socket_connect_timeout"] == 2.5
    retry = seen["retry"]
    assert retry.get_retries() == 4
    assert "secret" not in repr(cache)


def test_missing_redis_raises_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setitem(sys.modules, "redis.exceptions", None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[redis\]"):
        RedisCache(client_factory=Factory())


async def test_logs_name_a_key_digest_never_the_key(caplog):
    cache = RedisCache(client_factory=Factory(error=redis_exc.ConnectionError("down")))
    key = "aa:tool:v1:thread=secret-thread:search:abc"
    with caplog.at_level(logging.WARNING, logger="agent_adapter.cache.redis"):
        await cache.get(key)
        await cache.set(key, b"v")
        await cache.delete(key)
    assert "secret-thread" not in caplog.text
    assert caplog.text.count(digest(key)[:8]) == 3


@pytest.mark.parametrize("ttl", [float("inf"), float("nan")])
async def test_non_finite_ttl_means_no_expiry(ttl):
    factory = Factory()
    cache = RedisCache(client_factory=factory)
    await cache.set("k", b"v", ttl=ttl)
    assert factory.made[0].calls == [("set", "k", b"v", None)]


@pytest.mark.parametrize("url", ["http://localhost", "redis://localhost:notaport"])
def test_malformed_url_is_a_config_error_without_the_url(url):
    with pytest.raises(ConfigError, match="malformed") as info:
        RedisCache(url)
    assert "localhost" not in str(info.value)


async def test_negative_infinite_ttl_deletes_the_key():
    factory = Factory()
    cache = RedisCache(client_factory=factory)
    await cache.set("k", b"v", ttl=float("-inf"))
    assert factory.made[0].calls == [("delete", "k")]
