"""Cache keys: canonical JSON, digests and the key formats for tools, LLM calls and Gemini."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import LLMRequest
from agent_adapter.core.types import CacheScope

logger = logging.getLogger(__name__)

KEY_VERSION = 1
_V = f"v{KEY_VERSION}"
_LLM_EXCLUDE = {"stream", "stage", "response_cache", "metadata"}


def json_default(obj: Any) -> Any:
    """`json.dumps` hook: pydantic models in JSON mode, sets as sorted lists; else TypeError."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=canonical_json)
    raise TypeError(f"{type(obj).__name__} is not JSON-serialisable")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no spaces, UTF-8).

    Raises:
        TypeError: `obj` holds a value JSON cannot represent (the caller treats it as
            uncacheable).
    """
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )
    except ValueError as exc:  # NaN / infinity, or a circular reference
        raise TypeError(str(exc)) from exc


def digest(obj: Any) -> str:
    """First 32 hex characters of the SHA-256 of `canonical_json(obj)`."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:32]


def tool_key(
    namespace: str,
    tool: str,
    args: Mapping[str, Any],
    *,
    scope: CacheScope,
    run_id: str,
    thread_id: str,
) -> str:
    """Key for a tool result; "" when the call cannot be cached (no run or thread id).

    Raises:
        TypeError: `args` are not JSON-serialisable.
        ToolUsageError: unknown `scope`.
    """
    if scope == "global":
        return f"{namespace}:tool:{_V}:global:{tool}:{digest(args)}"
    if scope not in ("run", "thread"):
        raise ToolUsageError(f"unknown cache scope {scope!r}")
    owner = run_id if scope == "run" else thread_id
    if not owner:
        logger.debug("tool %s: %s-scoped result not cached without a %s id", tool, scope, scope)
        return ""
    return f"{namespace}:tool:{_V}:{scope}={owner}:{tool}:{digest(args)}"


def llm_key(
    namespace: str, request: LLMRequest, *, provider: str, model: str, fields: Mapping[str, Any]
) -> str:
    """Key for an LLM response; `fields` is extra key material from the port and plug-ins.

    Raises:
        TypeError: `request.extra` or `fields` are not JSON-serialisable.
    """
    # Python mode, so canonical_json rejects NaN and non-JSON values instead of pydantic
    # turning NaN into null (a key collision) or raising its own serialisation error.
    payload = request.model_dump(exclude=_LLM_EXCLUDE)
    payload["fields"] = dict(fields)
    return f"{namespace}:llm:{_V}:{provider}:{model}:{digest(payload)}"


def gcache_key(namespace: str, model: str, prefix_digest: str) -> str:
    """Name under which a Gemini explicit cache for one static prompt prefix is stored."""
    return f"{namespace}:gcache:{_V}:{model}:{prefix_digest}"
