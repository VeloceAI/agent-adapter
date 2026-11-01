"""Cache envelopes: versioned UTF-8 JSON, never pickle."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from agent_adapter.cache.keys import json_default

logger = logging.getLogger(__name__)

ENVELOPE_VERSION = 1
EnvelopeKind = Literal["tool", "llm", "gcache", "embedding"]


@dataclass(frozen=True)
class Envelope:
    """A decoded cache entry."""

    kind: EnvelopeKind
    stored_at: float  # epoch seconds
    payload: Any


def encode(kind: EnvelopeKind, payload: Any) -> bytes:
    """Wrap `payload` as `{"v", "kind", "stored_at", "payload"}` JSON bytes.

    Raises:
        TypeError: `payload` is not JSON-serialisable.
    """
    envelope = {"v": ENVELOPE_VERSION, "kind": kind, "stored_at": time.time(), "payload": payload}
    try:
        text = json.dumps(envelope, ensure_ascii=False, allow_nan=False, default=json_default)
    except ValueError as exc:  # NaN / infinity, or a circular reference
        raise TypeError(str(exc)) from exc
    return text.encode("utf-8")


def decode(data: bytes, kind: EnvelopeKind) -> Envelope | None:
    """The envelope in `data`; None for bad JSON, another version or another kind."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        logger.debug("cache entry is not valid UTF-8 JSON; treated as a miss")
        return None
    if not isinstance(raw, dict) or raw.get("v") != ENVELOPE_VERSION:
        logger.debug("cache entry has another envelope version; treated as a miss")
        return None
    if raw.get("kind") != kind:
        logger.debug("cache entry is a %r envelope, wanted %r", raw.get("kind"), kind)
        return None
    stored_at = raw.get("stored_at")
    if (
        not isinstance(stored_at, int | float)
        or isinstance(stored_at, bool)
        or "payload" not in raw
    ):
        logger.debug("cache entry is malformed; treated as a miss")
        return None
    return Envelope(kind=kind, stored_at=float(stored_at), payload=raw["payload"])
