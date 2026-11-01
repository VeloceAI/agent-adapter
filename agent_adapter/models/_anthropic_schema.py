"""Anthropic structured-output schemas without the SDK (mirrors `anthropic.transform_schema`).

Every object is closed (`additionalProperties: false`); constraints the API does not enforce
(`minimum`, `maxLength`, `pattern`, unknown formats, `minItems` > 1, ...) move into the
description, so the model still sees them. Local `$ref`s are inlined first.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_adapter.core.errors import ConfigError, SchemaNotPortable
from agent_adapter.schema.transforms import inline_refs

logger = logging.getLogger(__name__)

STRING_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)


def to_anthropic_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """`output_config.format.schema` for a neutral JSON schema.

    Raises:
        ConfigError: the schema is recursive, has a dangling `$ref`, or a node has no type.
    """
    try:
        inlined = inline_refs(schema)
    except SchemaNotPortable as exc:
        logger.warning("anthropic output schema rejected: %s", "; ".join(exc.problems))
        raise ConfigError(f"anthropic output schema: {'; '.join(exc.problems)}") from exc
    return _node(inlined)


def _node(schema: dict[str, Any]) -> dict[str, Any]:
    rest = dict(schema)
    out: dict[str, Any] = {}
    kind = rest.pop("type", None)
    variants = {key: rest.pop(key, None) for key in ("anyOf", "oneOf", "allOf")}
    if isinstance(variants["anyOf"], list):
        out["anyOf"] = [_node(v) for v in variants["anyOf"]]
    elif isinstance(variants["oneOf"], list):
        out["anyOf"] = [_node(v) for v in variants["oneOf"]]
    elif isinstance(variants["allOf"], list):
        out["allOf"] = [_node(v) for v in variants["allOf"]]
    elif kind is None:
        problem = "a node needs 'type', 'anyOf', 'oneOf' or 'allOf'"
        logger.warning("anthropic output schema rejected: %s", problem)
        raise ConfigError(f"anthropic output schema: {problem}")
    else:
        out["type"] = kind
    for key in ("enum", "description", "title"):
        value = rest.pop(key, None)
        if value is not None and (key != "enum" or isinstance(value, list)):
            out[key] = value
    if kind == "object":
        out["properties"] = {name: _node(sub) for name, sub in rest.pop("properties", {}).items()}
        rest.pop("additionalProperties", None)
        out["additionalProperties"] = False
        required = rest.pop("required", None)
        if required is not None:
            out["required"] = required
    elif kind == "string":
        fmt = rest.pop("format", None)
        if fmt in STRING_FORMATS:
            out["format"] = fmt
        elif fmt:
            rest["format"] = fmt  # unsupported: described below
    elif kind == "array":
        items = rest.pop("items", None)
        if items is not None:
            out["items"] = _node(items)
        min_items = rest.pop("minItems", None)
        if min_items in (0, 1):
            out["minItems"] = min_items
        elif min_items is not None:
            rest["minItems"] = min_items
    if rest:
        note = "{" + ", ".join(f"{key}: {value}" for key, value in rest.items()) + "}"
        description = out.get("description")
        out["description"] = f"{description}\n\n{note}" if description is not None else note
    return out
