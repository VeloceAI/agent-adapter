"""CallableAuditStore: sign and chain each record, then hand it to a function (SIEM, queue)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from agent_adapter.audit._actions import AuditEntry
from agent_adapter.audit._chain import Chain, key_id, keyed_digest, resolve_key
from agent_adapter.core.errors import ConfigError, IntegrationError, ToolUsageError
from agent_adapter.runtime.ids import new_id

AuditRecord = dict[str, Any]


class CallableAuditStore:
    """Signs and chains entries into one stream, then calls `fn(record)` for each.

    Raises:
        ConfigError: `fn` is not callable, or no key or a short one.
    """

    def __init__(
        self, fn: Callable[[AuditRecord], None], *, key: str | bytes | None = None
    ) -> None:
        if not callable(fn):
            raise ConfigError("CallableAuditStore needs a callable")
        self._fn = fn
        self._chain = Chain(resolve_key(key))
        self.stream = new_id("aud_")
        self._events = 0
        self._opened = False
        self._closed = False
        self._lock = threading.Lock()

    def write(self, entry: AuditEntry) -> None:
        """Send one signed entry (the stream's `open` record first).

        Raises:
            ToolUsageError: the store is closed.
            IntegrationError: `fn` failed; the record's seq is used, so the gap shows.
        """
        with self._lock:
            if self._closed:
                raise ToolUsageError("the audit store is closed")
            if not self._opened:
                self._opened = True
                self._send(self._header("open"))
            self._events += 1
            self._send({**entry.body(), "stream": self.stream})

    def close(self) -> None:
        """Send the `seal` once; a store that sent nothing sends nothing."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._opened:
                self._send({**self._header("seal"), "events": self._events})

    def digest(self, value: Any) -> str:
        """A keyed digest of `value` under this store's key."""
        return keyed_digest(self._chain.key, value)

    def _header(self, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "stream": self.stream,
            "key_id": key_id(self._chain.key),
            "ts": time.time(),
        }

    def _send(self, body: dict[str, Any]) -> None:
        record, line = self._chain.sign(body)
        self._chain.advance(line)
        try:
            self._fn(record)
        except Exception as exc:
            raise IntegrationError("the audit callable failed", integration="audit") from exc
