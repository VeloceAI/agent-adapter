"""RedisBudgetStore: budget counters in Redis, shared by every worker and host of a service."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable
from typing import Any

from agent_adapter.core.errors import ConfigError, IntegrationError, log_failure
from agent_adapter.hooks.builtin.budgets.store import (
    EPSILON,
    BudgetUsage,
    Reserved,
    check_ids,
    check_reserve,
    check_usd,
)
from agent_adapter.runtime.loop_local import LoopLocal, close_client

logger = logging.getLogger(__name__)

# KEYS: 1 counter hash (field spent), 2 holds hash (id -> usd), 3 hold expiry zset (id -> ms).
_PURGE = """\
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local dead = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for _, id in ipairs(dead) do redis.call('HDEL', KEYS[2], id) end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
local spent = tonumber(redis.call('HGET', KEYS[1], 'spent') or '0')
local held = 0
for _, v in ipairs(redis.call('HVALS', KEYS[2])) do held = held + tonumber(v) end
"""
_EXPIRE = """\
if ARGV[5] ~= '' then
  for i = 1, 3 do redis.call('PEXPIREAT', KEYS[i], ARGV[5]) end
end
"""
# ARGV: 1 id, 2 usd, 3 limit ("" = none), 4 hold ms, 5 expire-at ms ("" = never), 6 epsilon.
# Returns {accepted "1"/"0", spent, held} as read before the hold.
RESERVE = (
    _PURGE
    + """\
local amount = tonumber(ARGV[2])
if ARGV[3] ~= '' then
  local limit = tonumber(ARGV[3])
  local committed = spent + held
  if not (committed < limit and committed + amount <= limit + tonumber(ARGV[6])) then
    return {'0', string.format('%.17g', spent), string.format('%.17g', held)}
  end
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
redis.call('ZADD', KEYS[3], now + tonumber(ARGV[4]), ARGV[1])
redis.call('HSETNX', KEYS[1], 'spent', '0')
"""
    + _EXPIRE
    + """\
return {'1', string.format('%.17g', spent), string.format('%.17g', held)}
"""
)
# ARGV: 1 id, 2 cost usd, 5 expire-at ms ("" = never); 3 and 4 unused.
SETTLE = (
    """\
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call('HINCRBYFLOAT', KEYS[1], 'spent', ARGV[2])
"""
    + _EXPIRE
    + "return 1\n"
)
RELEASE = """\
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
return 1
"""
# Read-only, so it also runs on a replica: expired holds are skipped, not deleted.
USAGE = """\
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local spent = tonumber(redis.call('HGET', KEYS[1], 'spent') or '0')
local held = 0
for _, id in ipairs(redis.call('ZRANGEBYSCORE', KEYS[3], '(' .. now, '+inf')) do
  local v = redis.call('HGET', KEYS[2], id)
  if v then held = held + tonumber(v) end
end
return {'1', string.format('%.17g', spent), string.format('%.17g', held)}
"""
_SHAS = {
    script: hashlib.sha1(script.encode()).hexdigest()
    for script in (RESERVE, SETTLE, RELEASE, USAGE)
}


class RedisBudgetStore:
    """A `BudgetStore` on Redis (`agent-adapter[redis]`), shared by every worker and host.

    `client_factory()` returns a `redis.asyncio.Redis` (one per event loop). Each operation
    is one Lua script (Redis 5 or later), so the check and the hold are atomic across
    processes; hold expiry uses the Redis clock. A windowed counter expires with its window.
    A counter's three keys share a hash tag (one Cluster slot). Backend errors propagate.

    Raises:
        ConfigError: `client_factory` is not callable, or `prefix` does not end in ":".
    """

    def __init__(self, client_factory: Callable[[], Any], *, prefix: str = "aa:budget:") -> None:
        if not callable(client_factory):
            raise ConfigError("client_factory must be a callable returning a redis client")
        if not isinstance(prefix, str) or not prefix.endswith(":"):
            raise ConfigError(f"prefix must be a str ending in ':', got {prefix!r}")
        if "{" in prefix or "}" in prefix:
            raise ConfigError("prefix must not hold a hash tag: the store adds one per counter")
        self.prefix = prefix
        self._clients: LoopLocal[Any] = LoopLocal(client_factory, close=close_client)

    async def reserve(
        self,
        key: str,
        reservation_id: str,
        amount_usd: float,
        *,
        limit_usd: float | None,
        hold_s: float,
        expire_at: float | None,
    ) -> Reserved:
        """Hold `amount_usd` when it fits `limit_usd` (None: always), in one script."""
        check_reserve(key, reservation_id, amount_usd, hold_s)
        limit = "" if limit_usd is None else repr(float(limit_usd))
        args = (
            reservation_id,
            repr(float(amount_usd)),
            limit,
            str(max(1, math.ceil(hold_s * 1000))),
            _ms(expire_at),
            repr(EPSILON),
        )
        reply = await self._eval("reserve", RESERVE, key, args)
        return Reserved(reply[0], reply[1])

    async def settle(
        self, key: str, reservation_id: str, cost_usd: float, *, expire_at: float | None
    ) -> None:
        """Drop the hold and add the real cost (even when the hold already expired)."""
        check_ids(key, reservation_id)
        check_usd(cost_usd, "cost_usd")
        args = (reservation_id, repr(float(cost_usd)), "", "", _ms(expire_at))
        await self._call("settle", SETTLE, key, args)

    async def release(self, key: str, reservation_id: str) -> None:
        """Drop the hold; nothing was spent."""
        check_ids(key, reservation_id)
        await self._call("release", RELEASE, key, (reservation_id,))

    async def usage(self, key: str) -> BudgetUsage:
        """The counter's spend and live holds; zero for an unknown or ended key."""
        if not isinstance(key, str) or not key:
            raise ConfigError(f"key must be a non-empty str, got {key!r}")
        return (await self._eval("usage", USAGE, key, ()))[1]

    def keys_of(self, key: str) -> tuple[str, str, str]:
        """The counter, holds and expiry keys of one budget counter."""
        base = f"{self.prefix}{{{key}}}:"
        return base + "c", base + "h", base + "x"

    async def _eval(
        self, op: str, script: str, key: str, args: tuple[str, ...]
    ) -> tuple[bool, BudgetUsage]:
        reply = await self._call(op, script, key, args)
        if not isinstance(reply, list | tuple) or len(reply) != 3:
            logger.error("redis budget store %s: unexpected script reply", op)
            raise IntegrationError("unexpected reply from the budget script", integration="redis")
        flag, spent, held = (_text(part) for part in reply)
        if flag not in ("0", "1"):
            logger.error("redis budget store %s: unexpected script reply", op)
            raise IntegrationError("unexpected reply from the budget script", integration="redis")
        return flag == "1", BudgetUsage(_usd(spent), _usd(held))

    async def _call(self, op: str, script: str, key: str, args: tuple[str, ...]) -> Any:
        client = self._clients.get()
        keys = self.keys_of(key)
        try:
            try:
                return await client.evalsha(_SHAS[script], len(keys), *keys, *args)
            except Exception as exc:
                if not _no_script(exc):
                    raise
            return await client.eval(script, len(keys), *keys, *args)  # loads it for next time
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis budget store %s failed", op)
            raise


def _no_script(exc: Exception) -> bool:
    """The server does not have the script cached yet (redis-py's NoScriptError)."""
    text = str(exc).upper()
    return (
        type(exc).__name__ == "NoScriptError" or "NOSCRIPT" in text or "NO MATCHING SCRIPT" in text
    )


def _ms(at: float | None) -> str:
    return "" if at is None else str(math.ceil(at * 1000))


def _text(value: Any) -> str:
    # Clients made with decode_responses=True hand back str; keep both readable.
    if isinstance(value, bytes):
        return value.decode("ascii", "replace")
    return str(value)


def _usd(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        logger.error("redis budget store holds a malformed amount")
        raise IntegrationError("the budget store holds a malformed amount", integration="redis")
    return value
