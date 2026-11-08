"""RedisCheckpointStore: checkpoints in Redis, shared by every worker and host of a service."""

from __future__ import annotations

import builtins
import logging
import math
import time
from collections.abc import Callable
from typing import Any

from agent_adapter.core.errors import CheckpointStoreCorrupt, ConfigError, log_failure
from agent_adapter.exec.checkpoints._port import (
    CallRecord,
    Checkpoint,
    StoredCheckpoint,
    check_id,
    check_limit,
    check_save,
    conflict,
    decode_call,
    encode_call,
    encode_checkpoint,
    stale_segment,
    stored,
)
from agent_adapter.runtime.loop_local import LoopLocal, close_client

logger = logging.getLogger(__name__)

DEFAULT_TTL_S = 7 * 24 * 3600.0
_PAGE = 100

# A run is a hash <p>run:<hex run> (step, name, status, segment, record). <p>unfinished
# scores running run ids by save time. A run's logged calls are one hash <p>calls:<hex run>
# (field "<hex segment>:<hex key>" -> record) and a list <p>order:<hex run> of those fields in
# first-write order. Every key a script touches is passed in KEYS.

# KEYS: 1 run, 2 unfinished, 3 calls, 4 order. ARGV: 1 run id, 2 step, 3 name, 4 status,
# 5 segment, 6 hex segment, 7 saved_at, 8 record, 9 ttl ms ("0": never).
SAVE = """\
local cur = redis.call('HGET', KEYS[1], 'step')
local want = tonumber(ARGV[2])
if cur then
  local have = tonumber(cur)
  if have == nil then return {'corrupt'} end
  if have + 1 ~= want then return {'conflict', cur} end
elseif want ~= 0 then
  return {'conflict'}
end
redis.call('HSET', KEYS[1], 'step', ARGV[2], 'name', ARGV[3], 'status', ARGV[4],
  'segment', ARGV[5], 'record', ARGV[8])
if ARGV[4] == 'running' then
  redis.call('ZADD', KEYS[2], ARGV[7], ARGV[1])
else
  redis.call('ZREM', KEYS[2], ARGV[1])
end
local prefix = ARGV[6] .. ':'
local kept = {}
for _, field in ipairs(redis.call('LRANGE', KEYS[4], 0, -1)) do
  if string.sub(field, 1, #prefix) == prefix then
    kept[#kept + 1] = field
  else
    redis.call('HDEL', KEYS[3], field)
  end
end
redis.call('DEL', KEYS[4])
if #kept > 0 then redis.call('RPUSH', KEYS[4], (table.unpack or unpack)(kept)) end
local ttl = tonumber(ARGV[9])
if ARGV[4] == 'paused' then
  for i = 1, 4 do if i ~= 2 then redis.call('PERSIST', KEYS[i]) end end
elseif ttl > 0 then
  for i = 1, 4 do if i ~= 2 then redis.call('PEXPIRE', KEYS[i], ttl) end end
end
return {'ok'}
"""

# KEYS: 1 run, 2 calls, 3 order. ARGV: 1 segment, 2 field, 3 record, 4 ttl ms.
# Returns 1, or 0 when the segment is not the run's current one.
RECORD = """\
local seg = redis.call('HGET', KEYS[1], 'segment')
if not seg or seg ~= ARGV[1] then return 0 end
if redis.call('HEXISTS', KEYS[2], ARGV[2]) == 0 then redis.call('RPUSH', KEYS[3], ARGV[2]) end
redis.call('HSET', KEYS[2], ARGV[2], ARGV[3])
local ttl = tonumber(ARGV[4])
if ttl > 0 then
  for i = 1, 3 do redis.call('PEXPIRE', KEYS[i], ttl) end
end
return 1
"""

# KEYS: 1 calls, 2 order. ARGV: 1 "<hex segment>:". Returns {field, record, ...} in order.
CALLS = """\
local out = {}
for _, field in ipairs(redis.call('LRANGE', KEYS[2], 0, -1)) do
  if string.sub(field, 1, #ARGV[1]) == ARGV[1] then
    out[#out + 1] = field
    out[#out + 1] = redis.call('HGET', KEYS[1], field) or ''
  end
end
return out
"""

# KEYS: 1 run, 2 unfinished, 3 calls, 4 order. ARGV: 1 run id. Returns 1 / 0.
DELETE = """\
redis.call('DEL', KEYS[3], KEYS[4])
redis.call('ZREM', KEYS[2], ARGV[1])
return redis.call('DEL', KEYS[1])
"""


class RedisCheckpointStore:
    """CheckpointStore on Redis (`agent-adapter[redis]`), shared by every worker and host.

    `client_factory()` returns a `redis.asyncio.Redis` (one per event loop), e.g.
    `lambda: Redis.from_url(url)`. `save`, `record_call` and `delete` are Lua scripts, so a
    step's compare-and-set is atomic. A run's keys expire `ttl_s` after its last write
    (`None`: never), except while it is paused: a paused run waits for a person and never
    expires. On Redis Cluster put a hash tag in `prefix` (`"{checkpoints}:"`), since the
    scripts touch a run's keys and the shared index together.

    Raises:
        ConfigError: `prefix` is not a string ending in ":", or `ttl_s` is not a finite
            number > 0 or None.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        prefix: str = "aa:checkpoint:",
        ttl_s: float | None = DEFAULT_TTL_S,
    ) -> None:
        if not callable(client_factory):
            raise ConfigError("client_factory must be a callable returning a redis client")
        if not isinstance(prefix, str) or not prefix.endswith(":"):
            raise ConfigError(f"prefix must be a str ending in ':', got {prefix!r}")
        if ttl_s is not None and (
            isinstance(ttl_s, bool)
            or not isinstance(ttl_s, int | float)
            or not math.isfinite(ttl_s)
            or ttl_s <= 0
        ):
            raise ConfigError(f"ttl_s must be a finite number > 0 or None, got {ttl_s!r}")
        self.prefix = prefix
        self.ttl_s = ttl_s
        self._p = prefix.encode()
        self._clients: LoopLocal[Any] = LoopLocal(client_factory, close=close_client)

    async def save(self, run_id: str, step: int, checkpoint: Checkpoint) -> None:
        """Store the run's next checkpoint (compare-and-set on `step`).

        Raises:
            CheckpointConflict: `step` is not 0 for a new run or one past the stored step.
            CheckpointStoreCorrupt: the stored step is not an integer.
            ToolUsageError: a bad run id, step or checkpoint.
        """
        check_save(run_id, step, checkpoint)
        if not checkpoint.saved_at:
            checkpoint = checkpoint.model_copy(update={"saved_at": time.time()})
        keys = self._keys(run_id)
        args = (
            _key(run_id),
            str(step),
            checkpoint.name,
            checkpoint.status,
            _key(checkpoint.segment),
            _hex(checkpoint.segment),
            repr(checkpoint.saved_at),
            encode_checkpoint(checkpoint),
            self._ttl_ms(),
        )
        reply = await self._eval("save", SAVE, keys, args)
        outcome = _bytes(reply[0]) if reply else None
        if outcome == b"ok":
            return
        if outcome == b"conflict":
            have = _int(reply[1]) if len(reply) > 1 else None
            raise conflict(run_id, step, have)
        logger.error("run %r: the stored checkpoint step is unreadable", run_id[:64])
        raise CheckpointStoreCorrupt(f"the checkpoint of run {run_id[:64]!r} is unreadable")

    async def latest(self, run_id: str) -> StoredCheckpoint | None:
        """The run's latest checkpoint, or None.

        Raises:
            CheckpointStoreCorrupt: the stored hash is unreadable or disagrees with itself.
        """
        check_id(run_id, "run_id")
        row = await self._hmget(self._run_key(_hex(run_id)))
        if row is None:
            return None
        return stored(_key(run_id), row[0], row[2], row[1])

    async def list_unfinished(
        self, *, name: str | None = None, limit: int = 100
    ) -> builtins.list[StoredCheckpoint]:
        """Runs whose latest checkpoint is `running`, oldest first, optionally of one name.

        Raises:
            ConfigError: a negative `limit`.
        """
        check_limit(limit)
        found: builtins.list[StoredCheckpoint] = []
        start = 0
        client = self._clients.get()
        while len(found) < limit:
            try:
                ids = await client.zrange(self._p + b"unfinished", start, start + _PAGE - 1)
            except Exception as exc:
                log_failure(logger, logging.ERROR, exc, "redis checkpoint store list failed")
                raise
            if not ids:
                break
            start += len(ids)
            for raw in ids:
                run_id = _text(raw)
                item = await self.latest(run_id) if run_id else None
                if item is None:  # expired: drop it from the index
                    await self._forget(raw)
                    start -= 1
                    continue
                if item.checkpoint.status != "running":
                    continue  # finished since the index was read
                if name is None or item.checkpoint.name == name:
                    found.append(item)
        return found[:limit]

    async def delete(self, run_id: str) -> bool:
        """Forget the run and its call log; False when nothing was stored."""
        check_id(run_id, "run_id")
        return bool(await self._eval("delete", DELETE, self._keys(run_id), (_key(run_id),)))

    async def record_call(self, run_id: str, segment: str, call: CallRecord) -> None:
        """Keep a call record in the run's current segment (latest record per key wins).

        Raises:
            CheckpointConflict: `segment` is not the run's current one.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        check_id(call.key, "call key")
        run, _, calls, order = self._keys(run_id)
        field = _hex(segment) + b":" + _hex(call.key)
        args = (_key(segment), field, encode_call(call), self._ttl_ms())
        keys = (run, calls, order)
        if not await self._eval("record_call", RECORD, keys, args):
            raise stale_segment(run_id)

    async def calls(self, run_id: str, segment: str) -> builtins.list[CallRecord]:
        """The call records of one segment, in the order first written.

        Raises:
            CheckpointStoreCorrupt: a stored record is unreadable or has another key.
        """
        check_id(run_id, "run_id")
        check_id(segment, "segment")
        _, _, calls, order = self._keys(run_id)
        prefix = _hex(segment) + b":"
        flat = await self._eval("calls", CALLS, (calls, order), (prefix,))
        records = []
        for raw_key, raw in zip(flat[::2], flat[1::2], strict=True):
            record = decode_call(_bytes(raw) or b"", repr(run_id[:64]))
            if _bytes(raw_key) != prefix + _hex(record.key):
                logger.error("checkpoint call of run %r disagrees with its key", run_id[:64])
                raise CheckpointStoreCorrupt(f"a call of run {run_id[:64]!r} has another key")
            records.append(record)
        return records

    async def aclose(self) -> None:
        """Close the running loop's client."""
        await self._clients.aclose()

    # --- Redis side --------------------------------------------------------------------------

    def _run_key(self, run_hex: bytes) -> bytes:
        return self._p + b"run:" + run_hex

    def _keys(self, run_id: str) -> tuple[bytes, bytes, bytes, bytes]:
        """A run's hash, the unfinished index, its calls hash and its call order list."""
        run = _hex(run_id)
        return (
            self._run_key(run),
            self._p + b"unfinished",
            self._p + b"calls:" + run,
            self._p + b"order:" + run,
        )

    async def _forget(self, raw_id: Any) -> None:
        try:
            await self._clients.get().zrem(self._p + b"unfinished", raw_id)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis checkpoint store index cleanup failed")
            raise

    def _ttl_ms(self) -> int:
        return 0 if self.ttl_s is None else max(1, math.ceil(self.ttl_s * 1000))

    async def _hmget(self, name: bytes) -> builtins.list[Any] | None:
        try:
            row = await self._clients.get().hmget(name, ["step", "status", "record"])
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis checkpoint store read failed")
            raise
        if not row or all(value is None for value in row):
            return None
        return list(row)

    async def _eval(
        self, op: str, script: str, keys: tuple[bytes, ...], args: tuple[Any, ...]
    ) -> Any:
        client = self._clients.get()
        try:
            return await client.eval(script, len(keys), *keys, *args)
        except Exception as exc:
            log_failure(logger, logging.ERROR, exc, "redis checkpoint store %s failed", op)
            raise


def _key(text: str) -> bytes:
    return text.encode("utf-8", "surrogatepass")


def _hex(text: str) -> bytes:
    """An id in key form: hex, so no run id / segment pair can collide."""
    return _key(text).hex().encode()


def _bytes(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return _key(value)
    return None


def _text(value: Any) -> str:
    raw = _bytes(value)
    return "" if raw is None else raw.decode("utf-8", "surrogatepass")


def _int(value: Any) -> int | None:
    raw = _bytes(value)
    return int(raw) if raw is not None and raw.isdigit() else None
