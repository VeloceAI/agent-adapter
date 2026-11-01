"""CachePort fakes: a dict-backed cache with TTL and one that fails on every call."""

from __future__ import annotations

from collections.abc import Callable

from tests.fakes.clock import FakeClock


class DictCache:
    """CachePort over a dict; TTL measured with `clock`."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self.clock = clock or FakeClock()
        self.data: dict[str, tuple[bytes, float | None]] = {}
        self.calls: list[tuple[str, str]] = []

    async def get(self, key: str) -> bytes | None:
        self.calls.append(("get", key))
        item = self.data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and self.clock() >= expires:
            del self.data[key]
            return None
        return value

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        self.calls.append(("set", key))
        self.data[key] = (value, None if ttl is None else self.clock() + ttl)

    async def delete(self, key: str) -> None:
        self.calls.append(("delete", key))
        self.data.pop(key, None)


class BrokenCache:
    """CachePort that raises on every call (tests the callers' guards)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or ConnectionError("cache down")
        self.calls = 0

    async def get(self, key: str) -> bytes | None:
        self.calls += 1
        raise self.error

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        self.calls += 1
        raise self.error

    async def delete(self, key: str) -> None:
        self.calls += 1
        raise self.error
