import pytest

from agent_adapter.optimiser._schema_check import schema_problems


@pytest.mark.parametrize(
    ("value", "schema", "problems"),
    [
        (1, {"type": "integer"}, []),
        (True, {"type": "integer"}, ["$: expected integer, got boolean"]),
        (1.5, {"type": ["integer", "null"]}, ["$: expected integer or null, got number"]),
        (None, {"type": ["integer", "null"]}, []),
        ("a", {"type": "string", "minLength": 2}, ["$: 1 chars, at least 2 needed"]),
        ("abc", {"maxLength": 2}, ["$: 3 chars, at most 2 allowed"]),
        ([1], {"type": "array", "minItems": 2}, ["$: 1 items, at least 2 needed"]),
        ([1, 2, 3], {"maxItems": 2}, ["$: 3 items, at most 2 allowed"]),
        (["x", 2], {"items": {"type": "string"}}, ["$[1]: expected string, got integer"]),
        (5, {"maximum": 4}, ["$: 5 is above the maximum 4"]),
        (4, {"exclusiveMaximum": 4}, ["$: 4 must be below 4"]),
        (0, {"exclusiveMinimum": 0}, ["$: 0 must be above 0"]),
        ("b", {"enum": ["a"]}, ["$: 'b' is not one of ['a']"]),
        (2, {"const": 1}, ["$: must be 1"]),
        ({}, {"required": ["a"]}, ["$: missing key 'a'"]),
        (
            {"x": "s"},
            {"additionalProperties": {"type": "number"}},
            ["$.x: expected number, got string"],
        ),
        ({"x": 1}, {"properties": {}, "additionalProperties": True}, []),
        (
            3,
            {"anyOf": [{"type": "string"}, {"type": "boolean"}]},
            ["$: matches none of the allowed shapes"],
        ),
        ("s", {"oneOf": [{"type": "string"}]}, []),
        (object(), {"type": "object"}, ["$: expected object, got object"]),
        ({"a": 1}, {"unknownKeyword": 1}, []),
    ],
)
def test_schema_problems(value, schema, problems):
    assert schema_problems(value, schema) == problems
