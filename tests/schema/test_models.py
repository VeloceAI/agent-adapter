from typing import Any, Literal

import pytest
from pydantic import ValidationError

from agent_adapter.core.errors import ToolDefinitionError
from agent_adapter.schema.models import is_object_schema, model_from_json_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search text."},
        "unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
        "limit": {"type": "integer"},
        "ratio": {"type": "number"},
        "exact": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "object"},
        "region": {"type": ["string", "null"]},
        "page": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "mixed": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        "ref": {"$ref": "#/$defs/X"},
        "untyped": {},
    },
    "required": ["query", "limit"],
}


def annotations(model) -> dict[str, Any]:
    return {name: info.annotation for name, info in model.model_fields.items()}


def test_field_types_follow_the_mapping_table():
    model = model_from_json_schema("SearchArgs", SCHEMA)
    assert model.__name__ == "SearchArgs"
    assert annotations(model) == {
        "query": str,
        "unit": Literal["c", "f"] | None,
        "limit": int,
        "ratio": float | None,
        "exact": bool | None,
        "tags": list[Any] | None,
        "filters": dict[str, Any] | None,
        "region": str | None,
        "page": int | None,
        "mixed": Any,
        "ref": Any,
        "untyped": Any,
    }


def test_required_defaults_and_descriptions():
    model = model_from_json_schema("A", SCHEMA)
    fields = model.model_fields
    assert fields["query"].is_required() and fields["limit"].is_required()
    assert fields["unit"].default == "c" and fields["ratio"].default is None
    assert fields["query"].description == "Search text."
    value = model.model_validate({"query": "x", "limit": 2})
    assert value.unit == "c" and value.tags is None
    assert model.model_validate({"query": "x", "limit": 2, "region": None}).region is None


def test_validation_is_loose_but_typed_at_the_root():
    model = model_from_json_schema("A", SCHEMA)
    model.model_validate({"query": "x", "limit": 1, "tags": [1, {"a": 2}], "mixed": [1]})
    with pytest.raises(ValidationError):
        model.model_validate({"query": "x"})
    with pytest.raises(ValidationError):
        model.model_validate({"query": "x", "limit": 1, "unit": "k"})


def test_extra_properties_are_forbidden_unless_the_schema_opens_them():
    closed = model_from_json_schema("A", {"type": "object", "properties": {}})
    with pytest.raises(ValidationError):
        closed.model_validate({"other": 1})
    explicit = model_from_json_schema("A", {"type": "object", "additionalProperties": False})
    assert explicit.model_config["extra"] == "forbid"
    for extra in (True, {"type": "string"}):
        model = model_from_json_schema("A", {"type": "object", "additionalProperties": extra})
        assert model.model_validate({"other": 1}).model_extra == {"other": 1}


def test_unsafe_property_names_are_sanitised_with_aliases():
    props = ["first-name", "1st", "_hidden", "schema", "model_id", "a b", "first_name", ""]
    schema = {"type": "object", "properties": {p: {"type": "string"} for p in props}}
    model = model_from_json_schema("A", schema)
    fields = model.model_fields
    by_alias = {name if info.alias is None else info.alias: name for name, info in fields.items()}
    assert by_alias == {
        "first_name": "first_name",
        "first-name": "first_name_2",
        "1st": "p_1st",
        "_hidden": "p__hidden",
        "schema": "p_schema",
        "model_id": "p_model_id",
        "a b": "a_b",
        "": "p_",
    }
    value = model.model_validate({"first-name": "Ann", "first_name": "Bo", "1st": "x"})
    assert (value.first_name_2, value.first_name, value.p_1st) == ("Ann", "Bo", "x")


def test_root_without_type_but_with_properties_is_an_object():
    model = model_from_json_schema("A", {"properties": {"x": {"type": "integer"}}})
    assert annotations(model) == {"x": int | None}


@pytest.mark.parametrize(
    "schema",
    [{"type": "string"}, {"type": ["object", "null"]}, {}, {"properties": []}, "object"],
)
def test_non_object_roots_are_rejected(schema):
    assert not is_object_schema(schema)
    with pytest.raises(ToolDefinitionError, match="root must be an object"):
        model_from_json_schema("A", schema)


def test_odd_property_schemas_fall_back_to_any():
    schema = {
        "type": "object",
        "properties": {
            "nulls": {"type": ["null", "null"]},
            "triple": {"type": ["string", "integer", "null"]},
            "list_or_null": {"type": ["array", "null"]},
            "int_enum": {"type": "string", "enum": [1, 2]},
            "not_a_schema": True,
            "three": {"anyOf": [{"type": "string"}, {"type": "null"}, {"type": "integer"}]},
        },
    }
    model = model_from_json_schema("A", schema)
    assert annotations(model) == {
        "nulls": Any,
        "triple": Any,
        "list_or_null": Any,
        "int_enum": str | None,
        "not_a_schema": Any,
        "three": Any,
    }
