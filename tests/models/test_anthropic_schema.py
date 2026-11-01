import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.models._anthropic_schema import to_anthropic_output_schema

NESTED = {
    "type": "object",
    "title": "Report",
    "description": "A report.",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Score."},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 5}, "minItems": 1},
        "many": {"type": "array", "items": {"type": "number"}, "minItems": 3},
        "when": {"type": "string", "format": "date-time"},
        "slug": {"type": "string", "format": "slug", "pattern": "^[a-z]+$"},
        "owner": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": True,
        },
        "either": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "one": {"oneOf": [{"type": "integer"}, {"type": "boolean"}]},
        "both": {
            "allOf": [
                {"type": "object", "properties": {}},
                {"type": "object", "properties": {"x": {"type": "string"}}},
            ]
        },
        "kind": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["score"],
}


def test_objects_closed_and_constraints_moved_to_descriptions():
    out = to_anthropic_output_schema(NESTED)
    assert out["additionalProperties"] is False and out["title"] == "Report"
    props = out["properties"]
    assert props["score"] == {
        "type": "integer",
        "description": "Score.\n\n{minimum: 1, maximum: 10}",
    }
    assert props["tags"] == {
        "type": "array",
        "items": {"type": "string", "description": "{maxLength: 5}"},
        "minItems": 1,
    }
    assert props["many"] == {
        "type": "array",
        "items": {"type": "number"},
        "description": "{minItems: 3}",
    }
    assert props["when"] == {"type": "string", "format": "date-time"}
    assert props["slug"] == {"type": "string", "description": "{pattern: ^[a-z]+$, format: slug}"}
    assert props["owner"]["additionalProperties"] is False
    assert props["owner"]["required"] == ["name"]
    assert props["either"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert props["one"] == {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}
    assert props["both"]["allOf"][0]["additionalProperties"] is False
    assert props["kind"] == {"type": "string", "enum": ["a", "b"]}


def test_refs_are_inlined():
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }
    out = to_anthropic_output_schema(schema)
    assert out["properties"]["item"]["additionalProperties"] is False
    assert "$defs" not in out


def test_recursive_or_broken_schemas_are_rejected():
    recursive = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    }
    with pytest.raises(ConfigError, match="recursive"):
        to_anthropic_output_schema(recursive)
    with pytest.raises(ConfigError, match="type"):
        to_anthropic_output_schema({"properties": {}})


def test_list_types_pass_through():
    assert to_anthropic_output_schema({"type": ["string", "null"]}) == {"type": ["string", "null"]}


def test_type_less_nodes_are_logged_before_rejecting(caplog):
    import logging

    with caplog.at_level(logging.WARNING), pytest.raises(ConfigError):
        to_anthropic_output_schema({"type": "object", "properties": {"x": {"description": "?"}}})
    assert "anthropic output schema rejected" in caplog.text
