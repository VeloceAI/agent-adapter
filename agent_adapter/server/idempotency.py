"""Idempotency stores: one lease per `Idempotency-Key`, so a request runs once on any worker."""

from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from agent_adapter.runtime.loop_local import LoopLocal, close_client

logger = logging.getLogger(__name__)

ClaimState = Literal["new", "in_progress", "done"]
_ATTEMPTS = 3
# KEYS[1]; ARGV: expected value prefix, new value ("" deletes), PX ms. Atomic compare-and-set.
SWAP_SCRIPT = """\
local v = redis.call('GET', KEYS[1])
if v and string.sub(v, 1, string.len(ARGV[1])) == ARGV[1] then
  if ARGV[2] == '' then redis.call('DEL', KEYS[1])
  else redis.call('SET', KEYS[1], ARGV[2], 'PX', ARGV[3]) end
  return 1
end
return 0
"""


@dataclass(frozen=True)
class Stored:
    """A response kept for replay, with the fingerprint of the request that produced it."""

    fingerprint: str
    status: int
    body: Any
    media_type: str = "application/json"
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    """`new` (run it; `token` completes or releases the lease), `in_progress` or `done`.

    `fingerprint` names the request holding the key (in_progress / done); `stored` is the
    response to replay (done).
    """

    state: ClaimState
    token: str = ""
    fingerprint: str = ""
    stored: Stored | None = None


def stored_fields(stored: Stored) -> dict[str, Any]:
    """The JSON-ready fields of `stored`, without its fingerprint."""
    return {
        "status": stored.status,
        "body": stored.body,
        "media_type": stored.media_type,
        "headers": dict(stored.headers),
    }


def stored_from(data: Any, fingerprint: str) -> Stored:
    """The `Stored` that `stored_fields` wrote into `data`.

    Raises:
        ValueError, KeyError, TypeError, AttributeError: `data` is malformed.
    """
    return Stored(
        fingerprint=fingerprint,
        status=int(data["status"]),
        body=data["body"],
        media_type=str(data["media_type"]),
        headers={str(k): str(v) for k, v in data["headers"].items()},
    )


@runtime_checkable
class IdempotencyStore(Protocol):
    """Atomic per-key leases and replay entries shared by every worker of a service."""

    async def claim(self, key: str, fingerprint: str, lease_s: float) -> Claim:
        """Take a lease of `lease_s` seconds on a free key, or report who holds it."""
        ...

    async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
        """Replace the lease `token` holds with `stored` for `ttl_s`; False if it was lost."""
        ...

    async def release(self, key: str, token: str) -> None:
        """Free the lease `token` holds (nothing is replayed)."""
        ...

    async def update(self, key: str, stored: Stored, ttl_s: float) -> bool:
        """Replace a done entry with the same fingerprint (a decided approval's outcome)."""
        ...


@dataclass
class _Entry:
    token: str  # "" once done
    fingerprint: str
    expires: float
    stored: Stored | None = None


class MemoryIdempotencyStore:
    """In-process `IdempotencyStore` (one worker); oldest replay entries go first past
    `max_entries`, leases are never evicted."""

    def __init__(
        self, *, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._max = max_entries
        self._clock = clock
        self._lock = threading.Lock()

    async def claim(self, key: str, fingerprint: str, lease_s: float) -> Claim:
        with self._lock:
            entry = self._live(key)
            if entry is not None and entry.stored is not None:
                return Claim("done", fingerprint=entry.fingerprint, stored=entry.stored)
            if entry is not None:
                return Claim("in_progress", fingerprint=entry.fingerprint)
            token = secrets.token_hex(16)
            self._entries[key] = _Entry(token, fingerprint, self._clock() + lease_s)
            self._evict()
            return Claim("new", token=token)

    async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
        with self._lock:
            entry = self._live(key)
            if entry is None or not token or entry.token != token:
                return False
            self._entries[key] = _Entry("", stored.fingerprint, self._clock() + ttl_s, stored)
            self._entries.move_to_end(key)
            return True

    async def release(self, key: str, token: str) -> None:
        with self._lock:
            entry = self._live(key)
            if entry is not None and token and entry.token == token:
                del self._entries[key]

    async def update(self, key: str, stored: Stored, ttl_s: float) -> bool:
        with self._lock:
            entry = self._live(key)
            if entry is None or entry.stored is None or entry.fingerprint != stored.fingerprint:
                return False
            self._entries[key] = _Entry("", stored.fingerprint, self._clock() + ttl_s, stored)
            return True

    def _live(self, key: str) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is not None and entry.expires <= self._clock():
            del self._entries[key]
            return None
        return entry

    def _evict(self) -> None:
        if len(self._entries) <= self._max:
            return
        for key in [k for k, e in self._entries.items() if e.stored is not None]:
            del self._entries[key]
            if len(self._entries) <= self._max:
                return
        logger.warning("idempotency leases exceed max_entries (%d): kept", self._max)


class RedisIdempotencyStore:
    """`IdempotencyStore` on Redis (`agent-adapter[redis]`), shared by every worker.

    `client_factory()` returns a `redis.asyncio.Redis` (one per event loop), e.g.
    `lambda: Redis.from_url(url)`. Leases are `SET key NX PX`; completing, releasing and
    updating are one compare-and-set script, so a lost lease never overwrites another
    worker's entry. Backend errors propagate: the service answers 503 (fails closed).
    """

    def __init__(self, client_factory: Callable[[], Any], *, prefix: str = "") -> None:
        self.prefix = prefix
        self._clients: LoopLocal[Any] = LoopLocal(client_factory, close=close_client)

    async def claim(self, key: str, fingerprint: str, lease_s: float) -> Claim:
        client = self._clients.get()
        name = self.prefix + key
        token = secrets.token_hex(16)
        for _ in range(_ATTEMPTS):
            if await client.set(name, f"i:{token}:{fingerprint}", nx=True, px=_ms(lease_s)):
                return Claim("new", token=token)
            raw = await client.get(name)
            if raw is not None:
                return _parse(raw)
        logger.warning("idempotency key kept changing under a claim: treating it as busy")
        return Claim("in_progress")

    async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
        return await self._swap(key, f"i:{token}:", _done(stored), ttl_s) if token else False

    async def release(self, key: str, token: str) -> None:
        if token:
            await self._swap(key, f"i:{token}:", "", 0)

    async def update(self, key: str, stored: Stored, ttl_s: float) -> bool:
        return await self._swap(key, f"d:{stored.fingerprint}:", _done(stored), ttl_s)

    async def aclose(self) -> None:
        """Close the running loop's client."""
        await self._clients.aclose()

    async def _swap(self, key: str, expected: str, new: str, ttl_s: float) -> bool:
        client = self._clients.get()
        swapped = await client.eval(SWAP_SCRIPT, 1, self.prefix + key, expected, new, _ms(ttl_s))
        return bool(swapped)


def _done(stored: Stored) -> str:
    text = json.dumps(
        stored_fields(stored), separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return f"d:{stored.fingerprint}:{text}"


def _parse(raw: Any) -> Claim:
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    kind, _, rest = text.partition(":")
    if kind == "i":
        return Claim("in_progress", fingerprint=rest.partition(":")[2])
    if kind == "d":
        fingerprint, _, payload = rest.partition(":")
        try:
            stored = stored_from(json.loads(payload), fingerprint)
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("unreadable idempotency entry: treating the key as busy")
            return Claim("in_progress")
        return Claim("done", fingerprint=fingerprint, stored=stored)
    logger.warning("unknown idempotency entry: treating the key as busy")
    return Claim("in_progress")


def _ms(seconds: float) -> int:
    return max(1, math.ceil(seconds * 1000))
