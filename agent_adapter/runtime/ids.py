"""Opaque ids: a prefix (`run_`, `call_`, `apr_`, `evt_`, `opt_`) plus 32 hex characters."""

from __future__ import annotations

import hashlib
import json
import uuid


def new_id(prefix: str = "") -> str:
    """`prefix` + a random uuid4 hex."""
    return prefix + uuid.uuid4().hex


def scoped_id(prefix: str, principal: str | None, key: str) -> str:
    """The id a server derives from a client key: two principals never share one.

    `prefix` + the first 32 hex of SHA-256 over `json.dumps({"k": key, "p": principal},
    sort_keys=True, separators=(",", ":"), ensure_ascii=False)` encoded UTF-8 with
    `surrogatepass`.
    """
    payload = json.dumps(
        {"k": key, "p": principal}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return prefix + hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:32]
