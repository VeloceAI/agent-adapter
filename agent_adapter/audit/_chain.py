"""The signed hash chain: canonical JSON, the user's key, record MACs and chain hashes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.errors import ConfigError

KEY_ENV = "AA_AUDIT_KEY"
MIN_KEY_BYTES = 16
GENESIS = "0" * 64  # the `prev` of a chain's first record
VERSION = 1


def resolve_key(key: str | bytes | None) -> bytes:
    """The signing key: `key`, else the `AA_AUDIT_KEY` environment variable.

    Raises:
        ConfigError: no key, or one shorter than 16 bytes.
    """
    if key is None:
        key = os.environ.get(KEY_ENV, "")
    raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    if not raw:
        raise ConfigError(f"audit needs a signing key: pass key= or set {KEY_ENV}")
    if len(raw) < MIN_KEY_BYTES:
        raise ConfigError(f"the audit key must be at least {MIN_KEY_BYTES} bytes")
    return raw


def key_id(key: bytes) -> str:
    """A short public id of `key`, so a reader can tell which key signed a segment."""
    return hmac.new(key, b"agent-adapter audit key id", hashlib.sha256).hexdigest()[:16]


def canonical(body: Mapping[str, Any]) -> bytes:
    """The one byte form of a record: sorted keys, no spaces, UTF-8, no NaN."""
    text = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def mac(key: bytes, body: Mapping[str, Any]) -> str:
    """HMAC-SHA256 over a record without its `mac` field."""
    unsigned = {name: value for name, value in body.items() if name != "mac"}
    return hmac.new(key, canonical(unsigned), hashlib.sha256).hexdigest()


def keyed_digest(key: bytes, value: Any) -> str:
    """A keyed digest of a value (edited arguments, a scope): linkable, never reversible."""
    payload = canonical({"value": value})
    return hmac.new(key, b"digest\x00" + payload, hashlib.sha256).hexdigest()[:32]


def chain_hash(line: bytes) -> str:
    """The SHA-256 the next record carries as `prev`."""
    return hashlib.sha256(line).hexdigest()


class Chain:
    """Signs records in order: each carries `seq`, the previous record's hash and a MAC."""

    def __init__(self, key: bytes, *, seq: int = 0, prev: str = GENESIS) -> None:
        self.key = key
        self.seq = seq
        self.prev = prev

    def sign(self, body: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
        """The signed record and its line (no newline); the chain does not advance yet."""
        record = {**body, "v": VERSION, "seq": self.seq, "prev": self.prev}
        record["mac"] = mac(self.key, record)
        return record, canonical(record)

    def advance(self, line: bytes) -> None:
        """Move past a record once it is stored."""
        self.seq += 1
        self.prev = chain_hash(line)


def check_mac(key: bytes, record: Mapping[str, Any]) -> bool:
    """True when the record's MAC matches its content."""
    given = record.get("mac")
    return isinstance(given, str) and hmac.compare_digest(given, mac(key, record))
