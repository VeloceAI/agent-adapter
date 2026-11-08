"""An in-memory redis.asyncio stand-in for RedisApprovalStore: HMGET and its four scripts.

Each script is mirrored in Python step by step; the Lua itself runs under fakeredis (lupa)
or a real Redis in the conformance suite when those are available.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from agent_adapter.approval import redis_scripts
from tests.fakes.clock import FakeClock


def _b(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode()


class FakeApprovalRedis:
    """Strings, hashes and sorted sets with millisecond expiry on `clock` (seconds)."""

    def __init__(self, *, broken: bool = False) -> None:
        self.strings: dict[bytes, bytes] = {}
        self.hashes: dict[bytes, dict[bytes, bytes]] = {}
        self.zsets: dict[bytes, dict[bytes, int]] = {}
        self.expires: dict[bytes, int] = {}  # key -> expiry, ms
        self.clock = FakeClock(1_000.0)
        self.broken = broken
        self.closed = False
        self.evals = 0
        self._lock = threading.Lock()  # Redis runs one command at a time

    # --- primitives -------------------------------------------------------------------------

    def _now_ms(self) -> int:
        return int(self.clock() * 1000)

    def _expire_due(self, key: bytes) -> None:
        if key in self.expires and self._now_ms() >= self.expires[key]:
            for store in (self.strings, self.hashes, self.zsets, self.expires):
                store.pop(key, None)  # type: ignore[attr-defined]

    def _check(self) -> None:
        if self.broken:
            raise ConnectionError("redis down")

    def _hmget(self, key: bytes, fields: list[bytes]) -> list[bytes | None]:
        self._expire_due(key)
        found = self.hashes.get(key, {})
        return [found.get(f) for f in fields]

    def _zset(self, key: bytes) -> dict[bytes, int]:
        self._expire_due(key)
        return self.zsets.setdefault(key, {})

    def _pexpire(self, key: bytes, ms: int) -> None:
        self.expires[key] = self._now_ms() + ms

    # --- client API -------------------------------------------------------------------------

    async def hmget(self, name: bytes, keys: list[str]) -> list[bytes | None]:
        self._check()
        with self._lock:
            return self._hmget(name, [_b(k) for k in keys])

    async def zscore(self, name: bytes, member: bytes) -> float | None:
        self._check()
        with self._lock:
            return self.zsets.get(_b(name), {}).get(_b(member))

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        self._check()
        self.evals += 1
        keys = [_b(a) for a in args[:numkeys]]
        argv = [_b(a) for a in args[numkeys:]]
        handler = {
            redis_scripts.CREATE: self._create,
            redis_scripts.FIND: self._find,
            redis_scripts.SWAP: self._swap,
            redis_scripts.PAGE: self._page,
        }[script]
        with self._lock:
            return handler(keys, argv)

    async def aclose(self) -> None:
        self.closed = True

    # --- script mirrors ---------------------------------------------------------------------

    _ROW: ClassVar[list[bytes]] = [b"record", b"status", b"consumed", b"run_id", b"call_id"]

    def _create(self, keys: list[bytes], argv: list[bytes]) -> list[Any]:
        self._expire_due(keys[0])
        open_id = self.strings.get(keys[0])
        if open_id is not None:
            cur = self._hmget(argv[0] + open_id, self._ROW)
            if cur[0] is not None and cur[2] != b"1":
                return [b"open", open_id, *cur]
        self._expire_due(keys[1])
        if keys[1] in self.hashes:
            return [b"taken"]
        ttl = int(argv[5])
        now = self._now_ms()
        if ttl > 0:
            exp = self._zset(keys[7])
            dead = sorted((s, m) for m, s in exp.items() if s <= now)[:100]
            for _, member in dead:
                for i in (2, 4, 5, 6, 7):
                    self._zset(keys[i]).pop(member, None)
        seq = int(self.strings.get(keys[8], b"0")) + 1
        self.strings[keys[8]] = str(seq).encode()
        self.hashes[keys[1]] = {
            b"record": argv[4],
            b"run_id": argv[2],
            b"call_id": argv[3],
            b"status": b"pending",
            b"consumed": b"0",
            b"seq": str(seq).encode(),
        }
        self.strings[keys[0]] = argv[1]
        self.expires.pop(keys[0], None)  # SET clears a TTL
        for i in (2, 3, 4):
            self._zset(keys[i])[argv[1]] = seq
        if ttl > 0:
            for i in (1, 0, 3):
                self._pexpire(keys[i], ttl)
            self._zset(keys[7])[argv[1]] = now + ttl
        return [b"created"]

    def _find(self, keys: list[bytes], argv: list[bytes]) -> list[Any]:
        self._expire_due(keys[0])
        open_id = self.strings.get(keys[0])
        if open_id is None:
            return []
        cur = self._hmget(argv[0] + open_id, self._ROW)
        if cur[0] is None or cur[2] == b"1":
            return []
        return [open_id, *cur]

    def _swap(self, keys: list[bytes], argv: list[bytes]) -> int:
        record, status, seq = self._hmget(keys[0], [b"record", b"status", b"seq"])
        if record != argv[0]:
            return 0
        self.hashes[keys[0]].update({b"record": argv[1], b"status": argv[2], b"consumed": argv[3]})
        if status != argv[2]:
            sets = {b"pending": keys[1], b"approved": keys[2], b"denied": keys[3]}
            if status in sets:
                self._zset(sets[status]).pop(argv[4], None)
            assert seq is not None, "ZADD with a missing score is a script error"
            self._zset(sets[argv[2]])[argv[4]] = int(seq)
        return 1

    def _page(self, keys: list[bytes], argv: list[bytes]) -> list[Any]:
        members = sorted(self._zset(keys[0]).items(), key=lambda item: item[1], reverse=True)
        start, stop = int(argv[1]), int(argv[2])
        ids = [m for m, _ in members][start : stop + 1]
        return [[member, *self._hmget(argv[0] + member, self._ROW)] for member in ids]
