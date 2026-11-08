"""A small JSON-schema checker for hook configs (type, enum, bounds, properties, items)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
}


def schema_problems(value: Any, schema: Mapping[str, Any], at: str = "$") -> list[str]:
    """Every way `value` breaks `schema`; keywords outside the supported set are ignored."""
    problems = _type_problems(value, schema, at)
    if problems:
        return problems
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{at}: {value!r} is not one of {list(schema['enum'])}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{at}: must be {schema['const']!r}")
    problems += _bound_problems(value, schema, at)
    if isinstance(value, dict):
        problems += _object_problems(value, schema, at)
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            problems += schema_problems(item, schema["items"], f"{at}[{index}]")
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if options and not any(not schema_problems(value, option, at) for option in options):
            problems.append(f"{at}: matches none of the allowed shapes")
    return problems


def _type_problems(value: Any, schema: Mapping[str, Any], at: str) -> list[str]:
    wanted = schema.get("type")
    if wanted is None:
        return []
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    if any(_is_type(value, name) for name in names):
        return []
    return [f"{at}: expected {' or '.join(names)}, got {_json_type(value)}"]


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, _TYPES.get(name, ()))


def _json_type(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_type(value, name):
            return name
    return type(value).__name__


def _bound_problems(value: Any, schema: Mapping[str, Any], at: str) -> list[str]:
    problems: list[str] = []
    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{at}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{at}: {value} is above the maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            problems.append(f"{at}: {value} must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            problems.append(f"{at}: {value} must be below {schema['exclusiveMaximum']}")
    if isinstance(value, str | list):
        unit = "chars" if isinstance(value, str) else "items"
        low = schema.get("minLength" if isinstance(value, str) else "minItems")
        high = schema.get("maxLength" if isinstance(value, str) else "maxItems")
        if low is not None and len(value) < low:
            problems.append(f"{at}: {len(value)} {unit}, at least {low} needed")
        if high is not None and len(value) > high:
            problems.append(f"{at}: {len(value)} {unit}, at most {high} allowed")
    return problems


def _object_problems(value: dict[str, Any], schema: Mapping[str, Any], at: str) -> list[str]:
    problems = [
        f"{at}: missing key {key!r}" for key in schema.get("required", ()) if key not in value
    ]
    properties: Mapping[str, Any] = schema.get("properties", {})
    extra = schema.get("additionalProperties", True)
    for key in value:
        if key not in properties and extra is False:
            problems.append(f"{at}: unexpected key {key!r}")
    for key, item in value.items():
        if key in properties:
            problems += schema_problems(item, properties[key], f"{at}.{key}")
        elif isinstance(extra, Mapping):
            problems += schema_problems(item, extra, f"{at}.{key}")
    return problems
