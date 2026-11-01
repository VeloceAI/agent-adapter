import pytest

from agent_adapter.core._nulls import schema_allows_null


@pytest.mark.parametrize(
    ("schema", "allowed"),
    [
        ({}, True),
        (True, True),
        (False, False),
        (None, True),
        ({"type": "null"}, True),
        ({"type": ["integer", "null"]}, True),
        ({"type": "string"}, False),
        ({"type": "string", "nullable": True}, True),
        ({"enum": ["a", None]}, True),
        ({"enum": ["a"]}, False),
        ({"const": None}, True),
        ({"const": "a"}, False),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, True),
        ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, False),
        ({"$ref": "#/$defs/X"}, False),
        ({"description": "anything"}, True),
    ],
)
def test_schema_allows_null(schema, allowed):
    assert schema_allows_null(schema) is allowed
