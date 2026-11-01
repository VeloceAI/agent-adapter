"""What `node.finished` says about state: changed field names, and values when traced."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from agent_adapter.core.types import to_jsonable
from agent_adapter.hooks.builtin.logging import redact_secrets

logger = logging.getLogger(__name__)

MAX_VALUE_CHARS = 2_048  # JSON characters kept of one traced value


def changed_fields(updates: Sequence[Mapping[str, Any]]) -> list[str]:
    """The state field names a node's updates set, in first-seen order."""
    seen: dict[str, None] = {}
    for update in updates:
        seen.update(dict.fromkeys(str(key) for key in update))
    return list(seen)


def traced_updates(view: Mapping[str, Any], changed: Sequence[str]) -> dict[str, Any]:
    """The changed fields' values as one node visit set them, JSON-safe, each capped.

    `view` is what `after_node` hooks see: a Map's field holds every kept item value.
    """
    return {key: _capped(key, view[key]) for key in changed if key in view}


def _capped(key: str, value: Any) -> Any:
    """`value` as JSON data with secret-named entries masked, or its truncated JSON text.

    Masked first: a truncated text is one string, so no later key-based masking can reach in.
    """
    try:
        data = redact_secrets({key: to_jsonable(value)})[key]  # the field's own name counts too
        text = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # a value with no JSON form never fails the node
        logger.warning(
            "state field %s has no JSON form (%s); traced as its type", key, type(exc).__name__
        )
        return f"<{type(value).__name__}>"
    if len(text) <= MAX_VALUE_CHARS:
        return data
    return f"{text[:MAX_VALUE_CHARS]}... [+{len(text) - MAX_VALUE_CHARS} chars]"
