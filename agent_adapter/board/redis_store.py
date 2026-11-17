"""RedisBoardStore: boards in Redis, shared by every worker and host of a service."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
from collections.abc import Callable
from typing import Any

from agent_adapter.board.entries import Entry, dumps, loads
from agent_adapter.core.errors import ConfigError, IntegrationError, log_failure
from agent_adapter.runtime.loop_local import LoopLocal, close_client

logger = logging.getLogger(__name__)

# KEYS: 1 entries list (JSON, seq = position), 2 bytes used, 3 claims hash.
_EXPIRE = """\
if ARGV[9] ~= '' then
  for i = 1, 3 do redis.call('PEXPIRE', KEYS[i], ARGV[9]) end
end
"""
# ARGV: 1 entry JSON, 2 max entries, 3 max bytes, 4 entry bytes, 9 ttl ms ("" = none).
APPEND = (
    """\
local n = redis.call('LLEN', KEYS[1])
local used = tonumber(redis.call('GET', KEYS[2]) or '0')
if n >= tonumber(ARGV[2]) or used + tonumber(ARGV[4]) > tonumber(ARGV[3]) then return -1 end
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('INCRBY', KEYS[2], ARGV[4])
"""
    + _EXPIRE
    + "return n + 1\n"
)
# ARGV: 1 key, 2 holder, 9 ttl ms. Returns {1 when this call recorded it, holder}.
CLAIM = (
    """\
local made = redis.call('HSETNX', KEYS[3], ARGV[1], ARGV[2])
local holder = redis.call('HGET', KEYS[3], ARGV[1])
"""
    + _EXPIRE
    + "return {made, holder}\n"
)
_SHAS = {script: hashlib.sha1(script.encode()).hexdigest() for script in (APPEND, CLAIM)}


class RedisBoardStore:
    """A `BoardStore` on Redis (`agent-adapter[redis]`), shared by every worker and host.

    `client_factory()` returns a `redis.asyncio.Redis` (one per event loop). `append` and
    `claim` are Lua scripts (Redis 5 or later), so cap checks and first-wins claims hold
    across processes. A board's keys share a hash tag (one Cluster slot) and expire
    `ttl_s` after its last write (None: never).

    Raises:
        ConfigError: `client_factory` is not callable, a bad `prefix` or `ttl_s`.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        prefix: str = "aa:board:",
        ttl_s: float | None = 7 * 24 * 3600,
    ) -> None:
        if not callable(client_factory):
            raise ConfigError("client_factory must be a callable returning a redis client")
        if not isinstance(prefix, str) or not prefix.endswith(":"):
            raise ConfigError(f"prefix must be a str ending in ':', got {prefix!r}")
        if "{" in prefix or "}" in prefix:
            raise ConfigError("prefix must not hold a hash tag: the store adds one per board")
        if ttl_s is not None and (
            isinstance(ttl_s, bool)
            or not isinstance(ttl_s, int | float)
            or not math.isfinite(ttl_s)
            or ttl_s <= 0
        ):
            raise ConfigError(f"ttl_s must be a finite number > 0 or None, got {ttl_s!r}")
        self.prefix = prefix
        self._ttl = "" if ttl_s is None else str(max(1, math.ceil(ttl_s * 1000)))
        self._clients: LoopLocal[Any] = LoopLocal(client_factory, close=close_client)

    def keys_of(self, scope: str) -> tuple[str, str, str]:
        """The entries, bytes-used and claims keys of one board."""
        base = f"{self.prefix}{{{scope}}}:"
        return base + "e", base + "b", base + "c"

    async def append(
        self, scope: str, entry: Entry, *, max_entries: int, max_bytes: int
    ) -> Entry | None:
        """Add `entry` as the board's next one, or None when a cap would be passed."""
        args = (dumps(entry), str(max_entries), str(max_bytes), str(entry.size()))
        seq = _int(await self._script("append", APPEND, scope, args))
        return None if seq < 0 else dataclasses.replace(entry, seq=seq)

    async def read(self, scope: str, *, since: int = 0) -> list[Entry]:
        """The board's entries after `seq` `since`, oldest first."""
        since = max(since, 0)
        rows = await self._command("read", "lrange", self.keys_of(scope)[0], since, -1)
        return [_entry(row, since + n + 1) for n, row in enumerate(rows or ())]

    async def claim(self, scope: str, key: str, holder: str) -> tuple[str, bool]:
        """The key's holder: `holder` when it was free (True) or already theirs."""
        reply = await self._script("claim", CLAIM, scope, (key, holder))
        if not isinstance(reply, list | tuple) or len(reply) != 2:
            raise _bad("claim")
        return _text(reply[1]), _int(reply[0]) == 1

    async def drop(self, scope: str) -> None:
        """Forget a board's entries and claims."""
        await self._command("drop", "delete", *self.keys_of(scope))

    async def _script(self, op: str, script: str, scope: str, args: tuple[str, ...]) -> Any:
        padded = (*args, *([""] * (8 - len(args))), self._ttl)  # ARGV[9] is the ttl
        keys = self.keys_of(scope)
        client = self._clients.get()
        try:
            try:
                return await client.evalsha(_SHAS[script], len(keys), *keys, *padded)
            except Exception as exc:
                if not _no_script(exc):
                    raise
            return await client.eval(script, len(keys), *keys, *padded)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis board store %s failed", op)
            raise

    async def _command(self, op: str, name: str, *args: Any) -> Any:
        try:
            return await getattr(self._clients.get(), name)(*args)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis board store %s failed", op)
            raise


def _no_script(exc: Exception) -> bool:
    text = str(exc).upper()
    return (
        type(exc).__name__ == "NoScriptError" or "NOSCRIPT" in text or "NO MATCHING SCRIPT" in text
    )


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _int(value: Any) -> int:
    try:
        return int(_text(value))
    except ValueError as exc:
        raise _bad("script") from exc


def _entry(row: Any, seq: int) -> Entry:
    try:
        return dataclasses.replace(loads(_text(row)), seq=seq)
    except ValueError as exc:
        raise _bad("read") from exc


def _bad(op: str) -> IntegrationError:
    logger.error("redis board store %s: unexpected reply", op)
    return IntegrationError(f"unexpected reply from the board store ({op})", integration="redis")
