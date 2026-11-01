"""`Idempotency-Key` replay for the routes: keys, fingerprints and fail-closed store calls."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from collections.abc import Mapping
from typing import Any, NoReturn

from agent_adapter.core.ports import CachePort
from agent_adapter.exec.control import is_control_flow
from agent_adapter.runtime.ids import scoped_id
from agent_adapter.server._payloads import refuse
from agent_adapter.server.idempotency import (
    Claim,
    IdempotencyStore,
    Stored,
    stored_fields,
    stored_from,
)

logger = logging.getLogger(__name__)

VERSION = 1
LEASE_MARGIN_S = 30.0  # beyond the attempts' timeouts: backoff and bookkeeping


def fingerprint(tool: str, arguments: Mapping[str, Any], run_key: str, thread_key: str) -> str:
    """What a repeat must match: tool, arguments, and the run and thread keys as sent."""
    payload = json.dumps(
        {"tool": tool, "arguments": arguments, "run": run_key, "thread": thread_key},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


class Replays:
    """The routes' view of an `IdempotencyStore`: store failures refuse with 503."""

    def __init__(
        self, store: IdempotencyStore, *, ttl: float, lease: float, namespace: str = "aa"
    ) -> None:
        self.store = store
        self.ttl = ttl
        self.lease = lease
        self._prefix = f"{namespace}:http:idem:v{VERSION}:"

    def key(self, principal_id: str | None, idempotency_key: str) -> str:
        """The store key of one principal's key (never shared with another principal)."""
        return self._prefix + scoped_id("", principal_id, idempotency_key)

    def lease_for(self, timeout: float | None, retries: int) -> float:
        """A lease outliving every attempt of a tool with a timeout (+ 30 s), never shorter
        than the default; a tool without a timeout gets the default."""
        if timeout is None:
            return self.lease
        return max(self.lease, timeout * (retries + 1) + LEASE_MARGIN_S)

    async def claim(self, key: str, sent: str, lease: float | None = None) -> Claim:
        """The store's claim. Raises a 503 refusal when the store fails (nothing runs)."""
        try:
            return await self.store.claim(key, sent, self.lease if lease is None else lease)
        except Exception as exc:
            _unavailable(exc)

    async def complete(self, key: str, token: str, stored: Stored) -> None:
        """Keep the outcome; a lost lease or a store failure is logged (the response stands)."""
        try:
            if not await self.store.complete(key, token, stored, self.ttl):
                logger.warning("idempotency lease was lost before the call finished")
        except Exception as exc:
            _logged(exc, "complete")

    async def release(self, key: str, token: str) -> None:
        try:
            await self.store.release(key, token)
        except Exception as exc:
            _logged(exc, "release")

    async def update(self, key: str, stored: Stored) -> None:
        try:
            await self.store.update(key, stored, self.ttl)
        except Exception as exc:
            _logged(exc, "update")


class CacheIdempotency:
    """Best effort over a plain `CachePort`: leases live in this process only, so two workers
    can both run one key; a cache failure is a miss."""

    def __init__(self, cache: CachePort) -> None:
        self.cache = cache
        self._busy: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    async def claim(self, key: str, fingerprint: str, lease_s: float) -> Claim:
        with self._lock:
            if key in self._busy:
                return Claim("in_progress", fingerprint=self._busy[key][1])
            token = secrets.token_hex(16)
            self._busy[key] = (token, fingerprint)
        stored = await self._get(key)
        if stored is not None:
            self._drop(key, token)
            return Claim("done", fingerprint=stored.fingerprint, stored=stored)
        return Claim("new", token=token)

    async def complete(self, key: str, token: str, stored: Stored, ttl_s: float) -> bool:
        await self._put(key, stored, ttl_s)
        return self._drop(key, token)

    async def release(self, key: str, token: str) -> None:
        self._drop(key, token)

    async def update(self, key: str, stored: Stored, ttl_s: float) -> bool:
        current = await self._get(key)
        if current is None or current.fingerprint != stored.fingerprint:
            return False
        await self._put(key, stored, ttl_s)
        return True

    def _drop(self, key: str, token: str) -> bool:
        with self._lock:
            if self._busy.get(key, ("", ""))[0] != token:
                return False
            del self._busy[key]
            return True

    async def _get(self, key: str) -> Stored | None:
        try:
            raw = await self.cache.get(key)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            logger.warning("idempotency cache read failed: %s", type(exc).__name__)
            return None
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            if data.get("v") != VERSION:
                return None
            return stored_from(data, str(data["fp"]))
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("ignoring an unreadable idempotency entry")
            return None

    async def _put(self, key: str, stored: Stored, ttl: float) -> None:
        payload = {"v": VERSION, "fp": stored.fingerprint, **stored_fields(stored)}
        try:
            raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
            await self.cache.set(key, raw, ttl=ttl)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            logger.warning("idempotency cache write failed: %s", type(exc).__name__)


def _unavailable(exc: Exception) -> NoReturn:
    if is_control_flow(exc):
        raise exc
    logger.warning("idempotency store failed: %s", type(exc).__name__)
    refuse("service-unavailable", "the idempotency store is unavailable; retry later")


def _logged(exc: Exception, operation: str) -> None:
    if is_control_flow(exc):
        raise exc
    logger.warning("idempotency store %s failed: %s", operation, type(exc).__name__)
