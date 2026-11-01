"""Pure mapping between ToolSpec and MCP shapes: hints, schemas, results, `_meta` keys."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.tool import Effects, ToolSpec, strict_downgraded
from agent_adapter.core.types import to_jsonable
from agent_adapter.schema.transforms import to_anthropic_input_schema

# `_meta` keys this adapter reads on requests and writes on results.
_META_PREFIX = "agent-adapter/"
RUN_ID_META = f"{_META_PREFIX}run_id"
CALL_ID_META = f"{_META_PREFIX}call_id"
APPROVAL_ID_META = f"{_META_PREFIX}approval_id"
ERROR_KIND_META = f"{_META_PREFIX}error_kind"

RESULT_KEY = "result"  # MCP outputSchema must be an object: other results are wrapped here
_DEFS_KEYS = ("$defs", "definitions")
_WRAPPER_KEYS = frozenset({"type", "properties", "required", "title", "description", *_DEFS_KEYS})


def tool_annotations(spec: ToolSpec) -> dict[str, bool]:
    """MCP `ToolAnnotations` fields (snake_case) from `effects`, `confirm` and `idempotent`."""
    hints: dict[str, bool] = {}
    if spec.effects is not None:
        hints["read_only_hint"] = spec.effects == "read_only"
        hints["destructive_hint"] = spec.effects == "destructive"
    elif spec.confirm is not False:
        hints["destructive_hint"] = True
    if spec.idempotent is not None:
        hints["idempotent_hint"] = spec.idempotent
    return hints


def input_schema_for(spec: ToolSpec) -> dict[str, Any]:
    """`inputSchema`: the Anthropic-plain schema; a non-portable imported schema as given."""
    schema = spec.json_schema()
    return schema if strict_downgraded(spec) else to_anthropic_input_schema(schema)


def output_schema_for(spec: ToolSpec) -> dict[str, Any] | None:
    """`outputSchema`: the result schema, wrapped under `result` unless it is an object."""
    schema = spec.result_schema()
    if schema is None or schema.get("type") == "object":
        return schema
    # `$ref`s point at the root, so definitions move up to the wrapper.
    inner = {k: v for k, v in schema.items() if k not in _DEFS_KEYS}
    wrapped: dict[str, Any] = {
        "type": "object",
        "properties": {RESULT_KEY: inner},
        "required": [RESULT_KEY],
    }
    wrapped.update({k: schema[k] for k in _DEFS_KEYS if k in schema})
    return wrapped


def structured_content(spec: ToolSpec, value: Any) -> Any:
    """`structuredContent` for `value`, wrapped exactly as `output_schema_for` wraps the schema."""
    data = to_jsonable(value)
    schema = spec.result_schema()
    if schema is not None and schema.get("type") == "object":
        return data
    return {RESULT_KEY: data}


def text_content(value: Any) -> str:
    """The text block for a result: strings as they are, anything else as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(to_jsonable(value), ensure_ascii=False)


def is_wrapper_schema(schema: Mapping[str, Any] | None) -> bool:
    """True for an object schema whose only (required) property is `result`."""
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return False
    properties = schema.get("properties")
    return (
        isinstance(properties, Mapping)
        and set(properties) == {RESULT_KEY}
        and list(schema.get("required") or []) == [RESULT_KEY]
        and set(schema) <= _WRAPPER_KEYS
    )


def unwrap_output_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The inner `result` schema of a wrapper, with the wrapper's definitions put back."""
    inner = dict(schema["properties"][RESULT_KEY])
    inner.update({k: schema[k] for k in _DEFS_KEYS if k in schema})
    return inner


def unwrap_structured(structured: Any, output_schema: Mapping[str, Any] | None) -> Any:
    """The tool's value from `structuredContent`: unwrapped when the result was wrapped."""
    if output_schema is not None:
        wrapped = is_wrapper_schema(output_schema)
    else:
        wrapped = isinstance(structured, dict) and set(structured) == {RESULT_KEY}
    if wrapped and isinstance(structured, dict) and RESULT_KEY in structured:
        return structured[RESULT_KEY]
    return structured


def effects_from_hints(read_only: bool | None, destructive: bool | None) -> Effects | None:
    """`ToolSpec.effects` from MCP hints; None when the hints do not say."""
    if read_only:
        return "read_only"
    if destructive is True:
        return "destructive"
    if destructive is False:
        return "additive"
    return None


def annotation_options(annotations: Any) -> dict[str, Any]:
    """`effects` / `idempotent` policy options from MCP annotations (a model or a mapping)."""
    if annotations is None:
        return {}

    def hint(name: str) -> bool | None:
        if isinstance(annotations, Mapping):
            value = annotations.get(name)
        else:
            value = getattr(annotations, name, None)
        return value if isinstance(value, bool) else None

    options: dict[str, Any] = {}
    effects = effects_from_hints(hint("read_only_hint"), hint("destructive_hint"))
    if effects is not None:
        options["effects"] = effects
    idempotent = hint("idempotent_hint")
    if idempotent is not None:
        options["idempotent"] = idempotent
    return options


__all__ = [
    "APPROVAL_ID_META",
    "CALL_ID_META",
    "ERROR_KIND_META",
    "RUN_ID_META",
    "output_schema_for",
    "tool_annotations",
]
