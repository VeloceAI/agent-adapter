"""What a recorder may write: no content unless asked, secrets masked, sizes bounded."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.errors import log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import to_jsonable
from agent_adapter.exec._settings import ToolPolicy
from agent_adapter.hooks.builtin.logging import is_secret_name, redact_secrets
from agent_adapter.tracing.sinks import CONTENT_KEYS

MAX_TEXT = 2_000  # characters kept of one string value
MAX_ITEMS = 200  # items kept of one list or mapping
MAX_DEPTH = 12
_VALUE_KEYS = ("default", "examples", "example", "const", "enum")
STATE_KEYS = frozenset({"updates"})
MAX_POLICY_LINES = 40

logger = logging.getLogger(__name__)  # node.finished state values (FlowRunner(trace_state=))


def scrub_data(
    data: Mapping[str, Any], *, capture_content: bool, keep_state: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Event data safe to write, and the content keys left out.

    State values stay with `capture_content` or `keep_state` (a run traced on purpose).
    """
    hidden = set() if capture_content else set(CONTENT_KEYS - STATE_KEYS)
    if not (capture_content or keep_state):
        hidden |= STATE_KEYS
    omitted = sorted(hidden & set(data))
    kept = {key: value for key, value in data.items() if key not in omitted}
    masked = redact_secrets(to_jsonable(kept))
    return _clip(masked, 0), omitted


def _clip(value: Any, depth: int) -> Any:
    if depth > MAX_DEPTH:
        return "[too deep]"
    if isinstance(value, str):
        if len(value) <= MAX_TEXT:
            return value
        return f"{value[:MAX_TEXT]}... [+{len(value) - MAX_TEXT} chars]"
    if isinstance(value, Mapping):
        items = list(value.items())
        clipped = {str(k): _clip(v, depth + 1) for k, v in items[:MAX_ITEMS]}
        if len(items) > MAX_ITEMS:
            clipped["..."] = f"+{len(items) - MAX_ITEMS} keys"
        return clipped
    if isinstance(value, list):
        kept = [_clip(item, depth + 1) for item in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            kept.append(f"... +{len(value) - MAX_ITEMS} items")
        return kept
    return value


def catalogue(registry: Registry, policy: ToolPolicy | None = None) -> list[dict[str, Any]]:
    """Each tool's name, schema and run settings (under `policy`, with its rules listed)."""
    return [_with_policy(spec, policy) for spec in registry]


def _with_policy(spec: ToolSpec, policy: ToolPolicy | None) -> dict[str, Any]:
    if policy is None:
        return _tool(spec)
    try:
        shown = _tool(policy.effective(spec))
        lines = policy.describe(spec)
    except Exception as exc:  # the catalogue must never stop a run from being recorded
        log_failure(logger, logging.WARNING, exc, "policy lines for tool %s not written", spec.name)
        return _tool(spec)
    shown["policy"] = [str(line)[:MAX_TEXT] for line in lines[:MAX_POLICY_LINES]]
    return shown


def _tool(spec: ToolSpec) -> dict[str, Any]:
    confirm = spec.confirm
    return {
        "name": spec.name,
        "description": spec.description[:MAX_TEXT],
        "kind": spec.kind,
        "schema": _schema(to_jsonable(spec.json_schema()), secret=False),
        "confirm": "always" if confirm is True else ("rule" if callable(confirm) else "never"),
        "cache_ttl": spec.cache_ttl,
        "cache_scope": spec.cache_scope,
        "effects": spec.effects,
        "idempotent": spec.idempotent,
        "timeout": spec.timeout,
        "retries": spec.retries,
        "parallel_safe": spec.parallel_safe,
        "once_per_run": spec.once_per_run,
        "tags": sorted(spec.tags),
    }


def _schema(value: Any, *, secret: bool) -> Any:
    """A JSON schema copy whose secret-named properties lose their example values."""
    if isinstance(value, list):
        return [_schema(item, secret=secret) for item in value]
    if not isinstance(value, Mapping):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if secret and key in _VALUE_KEYS:
            continue
        if key == "properties" and isinstance(item, Mapping):
            out[key] = {
                name: _schema(prop, secret=is_secret_name(name)) for name, prop in item.items()
            }
        else:
            out[key] = _schema(item, secret=secret)
    return out
