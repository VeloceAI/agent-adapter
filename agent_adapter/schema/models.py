"""Build a loose pydantic model from a raw JSON schema (for tools imported from elsewhere)."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.schema._names import unique

_PRIMITIVES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}
_CONTAINERS: dict[str, Any] = {"array": list[Any], "object": dict[str, Any]}
_UNSAFE = re.compile(r"[^0-9A-Za-z_]")


def model_from_json_schema(name: str, schema: Mapping[str, Any]) -> type[BaseModel]:
    """A model that validates the top-level properties of `schema` loosely.

    Only the root is typed (primitives, `Literal` enums, nullable primitives, lists, dicts);
    anything richer becomes `Any`, since the tool's owner validates it again.

    Raises:
        ToolDefinitionError: the root is not an object schema.
    """
    if not is_object_schema(schema):
        raise ToolDefinitionError(f"{name}: the schema root must be an object")
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    names = _field_names([str(prop) for prop in properties])
    fields: dict[str, Any] = {
        names[str(prop)]: _field(str(prop), names[str(prop)], sub, prop in required)
        for prop, sub in properties.items()
    }
    closed = schema.get("additionalProperties", False) is False
    config = ConfigDict(
        extra="forbid" if closed else "allow", validate_by_name=True, validate_by_alias=True
    )
    model: type[BaseModel] = create_model(name, __config__=config, **fields)
    return model


def field_aliases(schema: Mapping[str, Any]) -> dict[str, str]:
    """Field name -> property name for each property `model_from_json_schema` renames."""
    props = [str(prop) for prop in schema.get("properties") or {}]
    return {field: prop for prop, field in _field_names(props).items() if field != prop}


def is_object_schema(schema: Any) -> bool:
    """True for `type: "object"`, or no `type` but `properties`."""
    if not isinstance(schema, Mapping):
        return False
    if "type" in schema:
        return bool(schema["type"] == "object")
    return isinstance(schema.get("properties"), Mapping)


def _field(prop: str, field_name: str, sub: Any, required: bool) -> tuple[Any, Any]:
    sub = sub if isinstance(sub, Mapping) else {}
    annotation = _annotation(sub)
    options: dict[str, Any] = {}
    if isinstance(sub.get("description"), str):
        options["description"] = sub["description"]
    if field_name != prop:
        options["alias"] = prop
    if required:
        return annotation, Field(**options)
    if annotation is not Any:
        annotation = annotation | None  # an optional property may arrive as null
    return annotation, Field(copy.deepcopy(sub.get("default")), **options)


def _annotation(sub: Mapping[str, Any]) -> Any:
    kind = sub.get("type")
    if isinstance(kind, str):
        return _single(kind, sub)
    if isinstance(kind, list):
        others = [k for k in kind if k != "null"]
        if len(kind) == 2 and len(others) == 1 and others[0] in _PRIMITIVES:
            return _single(others[0], sub) | None
        return Any
    branches = sub.get("anyOf")
    if kind is None and isinstance(branches, list) and len(branches) == 2:
        rest = [b for b in branches if not (isinstance(b, Mapping) and b.get("type") == "null")]
        if len(rest) == 1 and isinstance(rest[0], Mapping) and rest[0].get("type") in _PRIMITIVES:
            return _single(rest[0]["type"], rest[0]) | None
    return Any


def _single(kind: str, sub: Mapping[str, Any]) -> Any:
    enum = sub.get("enum")
    if (
        kind == "string"
        and isinstance(enum, list)
        and enum
        and all(isinstance(v, str) for v in enum)
    ):
        return Literal[tuple(enum)]
    return _PRIMITIVES.get(kind) or _CONTAINERS.get(kind, Any)


def _field_names(props: list[str]) -> dict[str, str]:
    """Property -> field name; safe names are kept, the rest sanitised and made unique."""
    names = {prop: prop for prop in props if _safe(prop)}
    taken = set(names)
    counters: dict[str, int] = {}
    for prop in props:
        if prop in names:
            continue
        base = _UNSAFE.sub("_", prop)
        base = base if _safe(base) else "p_" + base
        names[prop] = candidate = unique(base, taken, counters)
        taken.add(candidate)
    return names


def _safe(name: str) -> bool:
    return (
        name.isidentifier()
        and not name.startswith(("_", "model_"))
        and not hasattr(BaseModel, name)
    )
