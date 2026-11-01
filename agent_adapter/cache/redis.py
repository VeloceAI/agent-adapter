"""Redis CachePort (`redis.asyncio`): one client per event loop; outages are cache misses."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Any

from agent_adapter.cache.keys import digest
from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.runtime.loop_local import LoopLocal

logger = logging.getLogger(__name__)

_INSTALL = 'pip install "agent-adapter[redis]"'


class RedisCache:
    """Byte cache on Redis; `RedisError` is logged at WARNING and treated as a miss.

    Raises:
        ConfigError: not exactly one of `url` / `client_factory`, a malformed URL, or bad
            retry settings.
        MissingDependency: the `redis` package is not installed.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
        prefix: str = "",
        socket_timeout: float = 5.0,
        retries: int = 3,
    ) -> None:
        if (url is None) == (client_factory is None):
            raise ConfigError("RedisCache needs exactly one of url or client_factory")
        if retries < 0:
            raise ConfigError(f"RedisCache retries must be >= 0, got {retries}")
        if socket_timeout <= 0:
            raise ConfigError(f"RedisCache socket_timeout must be > 0, got {socket_timeout}")
        self._error: type[Exception] = _redis_error()
        self.prefix = prefix
        factory = client_factory
        if factory is None:
            assert url is not None
            _check_url(url)
            factory = partial(_client_from_url, url, socket_timeout, retries)
        self._clients: LoopLocal[Any] = LoopLocal(factory, close=self._close_client)

    async def get(self, key: str) -> bytes | None:
        """The value, or None on a miss or a Redis error."""
        try:
            value = await self._clients.get().get(self.prefix + key)
        except self._error as exc:
            logger.warning("redis get failed for key #%s: %s", _tag(key), type(exc).__name__)
            return None
        if value is None or isinstance(value, bytes):
            return value
        if isinstance(value, str):  # a host client_factory with decode_responses=True
            return value.encode("utf-8")
        logger.warning(
            "redis get for key #%s returned unexpected %s", _tag(key), type(value).__name__
        )
        return None

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        """Store with `ex=ceil(ttl)`; `ttl <= 0` deletes the key; `inf` / `nan` never expire."""
        if ttl is not None and (math.isnan(ttl) or ttl == math.inf):
            ttl = None
        if ttl is not None and ttl <= 0:
            await self.delete(key)
            return
        ex = None if ttl is None else math.ceil(ttl)
        try:
            await self._clients.get().set(self.prefix + key, value, ex=ex)
        except self._error as exc:
            logger.warning("redis set failed for key #%s: %s", _tag(key), type(exc).__name__)

    async def delete(self, key: str) -> None:
        """Remove `key`; a Redis error is logged and ignored."""
        try:
            await self._clients.get().delete(self.prefix + key)
        except self._error as exc:
            logger.warning("redis delete failed for key #%s: %s", _tag(key), type(exc).__name__)

    async def aclose(self) -> None:
        """Close the running loop's client (a new one is made on next use)."""
        await self._clients.aclose()

    async def _close_client(self, client: Any) -> None:
        try:
            await client.aclose()
        except self._error as exc:
            logger.warning("closing the redis client failed: %s", type(exc).__name__)


def _tag(key: str) -> str:
    """A short digest naming `key` in logs (keys can carry run and thread ids)."""
    return digest(key)[:8]


def _check_url(url: str) -> None:
    from redis.asyncio.connection import parse_url

    try:
        parse_url(url)
    except ValueError:
        raise ConfigError(
            "RedisCache url is malformed (scheme redis://, rediss:// or unix://)"
        ) from None


def _redis_error() -> type[Exception]:
    try:
        from redis.exceptions import RedisError
    except ImportError as exc:
        raise MissingDependency(f"RedisCache needs redis: {_INSTALL}") from exc
    error: type[Exception] = RedisError
    return error


def _client_from_url(url: str, socket_timeout: float, retries: int) -> Any:
    from redis.asyncio import Redis
    from redis.asyncio.retry import Retry
    from redis.backoff import ExponentialWithJitterBackoff

    return Redis.from_url(
        url,
        decode_responses=False,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
        health_check_interval=30,
        # Retries ConnectionError / TimeoutError only, with jittered backoff up to 1 s.
        retry=Retry(ExponentialWithJitterBackoff(cap=1.0, base=0.01), retries=retries),
    )
