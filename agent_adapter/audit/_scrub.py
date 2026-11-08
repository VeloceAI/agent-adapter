"""What an audit record may hold: short scalars only, no content, secrets and PII masked."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_adapter.hooks.builtin.guards.pii import PII_KINDS, find_pii, redact_pii
from agent_adapter.hooks.builtin.logging import REDACTED, is_secret_name
from agent_adapter.tracing.sinks import CONTENT_KEYS

MAX_TEXT = 200  # characters kept of one string
MAX_ITEMS = 50  # items kept of one list
# Keys that name a person, team or tenant: kept only as a keyed digest.
IDENTITY_KEYS = frozenset(
    {"scope", "scope_id", "user", "user_id", "principal", "team", "tenant", "owner", "key"}
)
# Keys whose values the SDK or the developer names (ids, tools, backends): their digit runs
# are ids, not phone or card numbers, so only the secret patterns apply.
NAME_KEYS = frozenset(
    {"approval_id", "tool", "backend", "sandbox", "provider", "model", "language", "kind"}
    | {"stage", "window", "error_kind", "outcome"}
)
_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
Digest = Callable[[Any], str]


def clean_value(value: Any, *, name: bool = False) -> Any:
    """A short JSON scalar or list safe to keep, else None; a `name` is checked for secrets only."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value[:MAX_TEXT]
        plain = name and _NAME.fullmatch(text) is not None
        found = find_pii(text, ("secret",) if plain else PII_KINDS)
        return redact_pii(text, found) if found else text
    if isinstance(value, list | tuple):
        items = [clean_value(item) for item in list(value)[:MAX_ITEMS]]
        return [item for item in items if not isinstance(item, list)]
    return None


def clean_data(
    data: Mapping[str, Any], keys: Iterable[str] | None, digest: Digest
) -> dict[str, Any]:
    """`keys` of `data` (all when None): content dropped, secrets masked, identities digested."""
    wanted = data.keys() if keys is None else [key for key in keys if key in data]
    out: dict[str, Any] = {}
    for key in wanted:
        name = str(key)
        if name in CONTENT_KEYS:
            continue
        if is_secret_name(name):
            out[name] = REDACTED
        elif name in IDENTITY_KEYS and not _label(data[key]):
            out[f"{name}_digest"] = digest(data[key])
        else:
            out[name] = clean_value(data[key], name=name in NAME_KEYS)
    return out


def _label(value: Any) -> bool:
    """A short lower-case word such as a budget scope `"user"` is a label, not an identity."""
    return isinstance(value, str) and value in SCOPE_LABELS


SCOPE_LABELS = frozenset({"run", "thread", "global", "principal", "user", "team"})
