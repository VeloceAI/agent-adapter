"""Turn strict-mode nulls back into declared defaults before validation."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping
from typing import Annotated, Any, TypeGuard, Union

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def drop_strict_nulls(model: type[BaseModel], data: dict[str, Any]) -> None:
    """Delete `null` for every optional field whose default is not None, at any depth.

    OpenAI strict mode makes every field required, so the model sends `null` to mean
    "use the default". Fields whose default is None keep the null; it equals the default.
    """
    for name, info in model.model_fields.items():
        for key in _input_keys(model, name, info):
            if key not in data:
                continue
            value = data[key]
            if value is None:
                if not info.is_required() and info.default is not None:
                    del data[key]
                continue
            _descend(info.annotation, value)


def _input_keys(model: type[BaseModel], name: str, info: FieldInfo) -> list[str]:
    alias = info.validation_alias if isinstance(info.validation_alias, str) else info.alias
    if alias is None or alias == name:
        return [name]
    config = model.model_config
    by_name = config.get("populate_by_name") or config.get("validate_by_name")
    return [alias, name] if by_name else [alias]


def _descend(annotation: Any, value: Any) -> None:
    inner = _unwrap(annotation)
    if isinstance(value, dict) and _is_model(inner):
        drop_strict_nulls(inner, value)
    elif isinstance(value, list) and typing.get_origin(inner) is list:
        item_args = typing.get_args(inner)
        item = _unwrap(item_args[0]) if item_args else None
        if _is_model(item):
            for entry in value:
                if isinstance(entry, dict):
                    drop_strict_nulls(item, entry)


def _unwrap(annotation: Any) -> Any:
    """Strip `Annotated[...]` and a single `| None`."""
    if typing.get_origin(annotation) is Annotated:
        return _unwrap(typing.get_args(annotation)[0])
    if typing.get_origin(annotation) in (Union, types.UnionType):
        rest = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(rest) == 1:
            return _unwrap(rest[0])
    return annotation


def _is_model(annotation: Any) -> TypeGuard[type[BaseModel]]:
    return inspect.isclass(annotation) and issubclass(annotation, BaseModel)


def imported_kwargs(validated: BaseModel, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Arguments for an imported tool: set values by property name, no nulls it rejects.

    Unset optional arguments are not sent, so the remote side applies its own defaults.
    """
    properties = schema.get("properties") or {}
    extra = schema.get("additionalProperties", True)
    data = validated.model_dump(exclude_unset=True, by_alias=True)
    return {
        key: value
        for key, value in data.items()
        if value is not None or schema_allows_null(properties.get(key, extra))
    }


def schema_allows_null(schema: Any) -> bool:
    """True when a JSON schema accepts `null` (untyped and boolean `true` schemas do)."""
    if not isinstance(schema, Mapping):
        return schema is not False
    if schema.get("nullable") is True:
        return True
    if "enum" in schema:
        return None in schema["enum"]
    if "const" in schema:
        return schema["const"] is None
    kind = schema.get("type")
    if kind is not None:
        return kind == "null" or (isinstance(kind, list) and "null" in kind)
    branches = [b for key in ("anyOf", "oneOf") for b in schema.get(key) or ()]
    if branches:
        return any(schema_allows_null(branch) for branch in branches)
    return "$ref" not in schema
