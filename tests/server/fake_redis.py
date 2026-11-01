"""A minimal in-memory redis.asyncio stand-in: SET NX PX, GET, DEL and the one EVAL script."""

from __future__ import annotations

from typing import Any

from agent_adapter.server.idempotency import SWAP_SCRIPT
from tests.fakes.clock import FakeClock


class FakeRedis:
    """Values are bytes; `data[key] = (value, px_ms_or_None)`; expiry follows `clock`."""

    def __init__(self, *, broken: bool = False, vanish: int = 0) -> None:
        self.data: dict[str, tuple[bytes, int | None]] = {}
        self.expires: dict[str, float] = {}
        self.clock = FakeClock()
        self.broken = broken
        self.vanish = vanish  # GET returns None this many times after a failed SET NX
        self.closed = False

    def _check(self) -> None:
        if self.broken:
            raise ConnectionError("redis down")

    def _live(self, key: str) -> bytes | None:
        if key in self.expires and self.clock() >= self.expires[key]:
            self.data.pop(key, None)
            self.expires.pop(key, None)
        item = self.data.get(key)
        return item[0] if item is not None else None

    def _put(self, key: str, value: bytes, px: int | None) -> None:
        self.data[key] = (value, px)
        if px is None:
            self.expires.pop(key, None)
        else:
            self.expires[key] = self.clock() + px / 1000

    async def set(self, key: str, value: Any, *, nx: bool = False, px: int | None = None) -> bool:
        self._check()
        if nx and self._live(key) is not None:
            return False
        self._put(key, value.encode() if isinstance(value, str) else value, px)
        return True

    async def get(self, key: str) -> bytes | None:
        self._check()
        if self.vanish:
            self.vanish -= 1
            return None
        return self._live(key)

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        self._check()
        assert script == SWAP_SCRIPT and numkeys == 1
        key, expected, new, px = (a.decode() if isinstance(a, bytes) else str(a) for a in args)
        current = self._live(key)
        if current is None or not current.decode().startswith(expected):
            return 0
        if new == "":
            self.data.pop(key, None)
            self.expires.pop(key, None)
        else:
            self._put(key, new.encode(), int(px))
        return 1

    async def aclose(self) -> None:
        self.closed = True
