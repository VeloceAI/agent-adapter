"""The Lua scripts that make RedisApprovalStore's writes atomic (one script = one step)."""

from __future__ import annotations

# A record is a hash at <prefix>rec:<id> with fields record (JSON), run_id, call_id, status,
# consumed ("0" / "1") and seq. <prefix>open:<run>:<call> names the call's newest record;
# it is open only while that record exists with consumed "0". Sorted sets, scored by seq,
# index ids for `list`: all, run:<run_id>, status:<status>; exp scores ids by expiry (ms).

# KEYS: 1 open, 2 rec, 3 all, 4 run, 5 status:pending, 6 status:approved, 7 status:denied,
#       8 exp, 9 seq. ARGV: 1 record key prefix, 2 id, 3 run_id, 4 call_id, 5 record JSON,
#       6 ttl ms ("0" = never expires).
# Returns {"open", id, record, status, consumed, run_id, call_id} | {"taken"} | {"created"}.
CREATE = """\
local open = redis.call('GET', KEYS[1])
if open then
  local cur = redis.call('HMGET', ARGV[1] .. open, 'record', 'status', 'consumed', 'run_id',
    'call_id')
  if cur[1] and cur[3] ~= '1' then
    return {'open', open, cur[1], cur[2], cur[3], cur[4], cur[5]}
  end
end
if redis.call('EXISTS', KEYS[2]) == 1 then return {'taken'} end
local ttl = tonumber(ARGV[6])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
if ttl > 0 then
  local dead = redis.call('ZRANGEBYSCORE', KEYS[8], '-inf', now, 'LIMIT', 0, 100)
  for _, id in ipairs(dead) do
    for i = 3, 8 do
      if i ~= 4 then redis.call('ZREM', KEYS[i], id) end
    end
  end
end
local seq = redis.call('INCR', KEYS[9])
redis.call('HSET', KEYS[2], 'record', ARGV[5], 'run_id', ARGV[3], 'call_id', ARGV[4],
  'status', 'pending', 'consumed', '0', 'seq', seq)
redis.call('SET', KEYS[1], ARGV[2])
redis.call('ZADD', KEYS[3], seq, ARGV[2])
redis.call('ZADD', KEYS[4], seq, ARGV[2])
redis.call('ZADD', KEYS[5], seq, ARGV[2])
if ttl > 0 then
  redis.call('PEXPIRE', KEYS[2], ttl)
  redis.call('PEXPIRE', KEYS[1], ttl)
  redis.call('PEXPIRE', KEYS[4], ttl)
  redis.call('ZADD', KEYS[8], now + ttl, ARGV[2])
end
return {'created'}
"""

# KEYS: 1 open. ARGV: 1 record key prefix.
# Returns {} or {id, record, status, consumed, run_id, call_id} of the call's open record.
FIND = """\
local open = redis.call('GET', KEYS[1])
if not open then return {} end
local cur = redis.call('HMGET', ARGV[1] .. open, 'record', 'status', 'consumed', 'run_id',
  'call_id')
if not cur[1] or cur[3] == '1' then return {} end
return {open, cur[1], cur[2], cur[3], cur[4], cur[5]}
"""

# Compare-and-set of one record (decide, consume). KEYS: 1 rec, 2 status:pending,
# 3 status:approved, 4 status:denied. ARGV: 1 expected record JSON, 2 new record JSON,
# 3 new status, 4 new consumed flag, 5 id. Returns 1 when set, 0 when the record changed.
SWAP = """\
local cur = redis.call('HMGET', KEYS[1], 'record', 'status', 'seq')
if cur[1] ~= ARGV[1] then return 0 end
redis.call('HSET', KEYS[1], 'record', ARGV[2], 'status', ARGV[3], 'consumed', ARGV[4])
if cur[2] ~= ARGV[3] then
  local sets = {pending = KEYS[2], approved = KEYS[3], denied = KEYS[4]}
  if sets[cur[2]] then redis.call('ZREM', sets[cur[2]], ARGV[5]) end
  redis.call('ZADD', sets[ARGV[3]], cur[3], ARGV[5])
end
return 1
"""

# KEYS: 1 an index sorted set. ARGV: 1 record key prefix, 2 start, 3 stop (newest first).
# Returns one {id, record, status, consumed, run_id, call_id} per id; expired ones read false.
PAGE = """\
local ids = redis.call('ZREVRANGE', KEYS[1], ARGV[2], ARGV[3])
local out = {}
for i, id in ipairs(ids) do
  local cur = redis.call('HMGET', ARGV[1] .. id, 'record', 'status', 'consumed', 'run_id',
    'call_id')
  out[i] = {id, cur[1], cur[2], cur[3], cur[4], cur[5]}
end
return out
"""
