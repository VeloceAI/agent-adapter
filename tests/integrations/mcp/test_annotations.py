"""ToolSpec <-> MCP shape mapping: annotations, input / output schemas, structured results."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_adapter.core.imported import spec_from_schema
from agent_adapter.core.tool import build_spec
from agent_adapter.integrations.mcp.annotations import (
    annotation_options,
    effects_from_hints,
    input_schema_for,
    is_wrapper_schema,
    output_schema_for,
    structured_content,
    text_content,
    tool_annotations,
    unwrap_output_schema,
    unwrap_structured,
)
from agent_adapter.schema.transforms import to_anthropic_input_schema

pytestmark = pytest.mark.contract


async def remove(item: str) -> str:
    """Remove an item."""
    return item


class Point(BaseModel):
    x: int
    y: int


class Shape(BaseModel):
    name: str
    corners: list[Point]


async def make_shape(name: str, corners: list[Point]) -> Shape:
    """Build a shape."""
    return Shape(name=name, corners=corners)


async def count(n: int) -> int:
    """Return n."""
    return n


async def maybe_point(x: int) -> Point | None:
    """A point or nothing."""
    return Point(x=x, y=x)


async def untyped(x: int):  # type: ignore[no-untyped-def]
    """No return annotation."""
    return x


@pytest.mark.parametrize(
    ("policy", "hints"),
    [
        ({"effects": "read_only"}, {"read_only_hint": True, "destructive_hint": False}),
        ({"effects": "additive"}, {"read_only_hint": False, "destructive_hint": False}),
        ({"effects": "destructive"}, {"read_only_hint": False, "destructive_hint": True}),
        (
            {"effects": "read_only", "confirm": True},
            {"read_only_hint": True, "destructive_hint": False},
        ),
        ({"confirm": True}, {"destructive_hint": True}),
        ({"confirm": lambda call, ctx: False}, {"destructive_hint": True}),
        ({}, {}),
        ({"idempotent": True}, {"idempotent_hint": True}),
        (
            {"idempotent": False, "effects": "additive"},
            {
                "read_only_hint": False,
                "destructive_hint": False,
                "idempotent_hint": False,
            },
        ),
    ],
)
def test_tool_annotations_follow_the_contract_table(policy: dict[str, Any], hints: dict[str, Any]):
    assert tool_annotations(build_spec(remove, **policy)) == hints


def test_input_schema_is_the_plain_transform_of_json_schema():
    spec = build_spec(make_shape)
    schema = input_schema_for(spec)
    assert schema == to_anthropic_input_schema(spec.json_schema())
    assert "$defs" not in schema and "title" not in schema
    assert schema["properties"]["corners"]["items"]["properties"]["x"] == {"type": "integer"}


def test_input_schema_of_a_non_portable_imported_tool_is_verbatim():
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "title": "Recursive",
    }
    spec = spec_from_schema("walk", "Walk a list.", schema, lambda **kw: kw)
    assert not spec.portable
    assert input_schema_for(spec) == schema


def test_output_schema_object_results_pass_through():
    spec = build_spec(make_shape)
    assert output_schema_for(spec) == spec.result_schema()


def test_output_schema_wraps_scalars_under_result():
    assert output_schema_for(build_spec(count)) == {
        "type": "object",
        "properties": {"result": {"type": "integer"}},
        "required": ["result"],
    }


def test_output_schema_hoists_defs_so_refs_still_resolve():
    rs = build_spec(maybe_point).result_schema()
    assert rs is not None and "$defs" in rs
    wrapped = output_schema_for(build_spec(maybe_point))
    assert wrapped is not None
    assert wrapped["$defs"] == rs["$defs"]
    assert "$defs" not in wrapped["properties"]["result"]
    assert wrapped["properties"]["result"]["anyOf"][0] == {"$ref": "#/$defs/Point"}


def test_output_schema_is_none_without_a_result_schema():
    assert output_schema_for(build_spec(untyped)) is None


def test_structured_content_follows_the_same_wrapping():
    shape = Shape(name="tri", corners=[Point(x=0, y=1)])
    assert structured_content(build_spec(make_shape), shape) == {
        "name": "tri",
        "corners": [{"x": 0, "y": 1}],
    }
    assert structured_content(build_spec(count), 3) == {"result": 3}
    assert structured_content(build_spec(untyped), {"a": 1}) == {"result": {"a": 1}}


def test_text_content_keeps_strings_and_dumps_everything_else():
    assert text_content("héllo") == "héllo"
    assert text_content({"a": "é"}) == '{"a": "é"}'
    assert text_content(Point(x=1, y=2)) == '{"x": 1, "y": 2}'
    assert text_content(None) == "null"


def test_wrapper_schema_detection_accepts_titles_and_defs_only():
    assert is_wrapper_schema(output_schema_for(build_spec(count)))
    assert is_wrapper_schema(
        {"type": "object", "title": "Out", "properties": {"result": {}}, "required": ["result"]}
    )
    assert not is_wrapper_schema(None)
    assert not is_wrapper_schema({"type": "object", "properties": {"result": {}}})
    assert not is_wrapper_schema(
        {"type": "object", "properties": {"result": {}, "x": {}}, "required": ["result"]}
    )
    assert not is_wrapper_schema(
        {
            "type": "object",
            "properties": {"result": {}},
            "required": ["result"],
            "additionalProperties": False,
        }
    )


def test_unwrap_output_schema_reattaches_defs():
    wrapped = output_schema_for(build_spec(maybe_point))
    assert wrapped is not None
    assert unwrap_output_schema(wrapped) == build_spec(maybe_point).result_schema()


@pytest.mark.parametrize(
    ("structured", "schema", "expected"),
    [
        (
            {"result": 5},
            {"type": "object", "properties": {"result": {}}, "required": ["result"]},
            5,
        ),
        ({"result": 5}, None, 5),
        ({"result": 5, "x": 1}, None, {"result": 5, "x": 1}),
        ({"a": 1}, {"type": "object", "properties": {"a": {}}}, {"a": 1}),
        ({"result": 5}, {"type": "object", "properties": {"result": {}}}, {"result": 5}),
        ([1, 2], None, [1, 2]),
    ],
)
def test_unwrap_structured(structured: Any, schema: Any, expected: Any):
    assert unwrap_structured(structured, schema) == expected


@pytest.mark.parametrize(
    ("read_only", "destructive", "effects"),
    [
        (True, None, "read_only"),
        (True, True, "read_only"),
        (False, True, "destructive"),
        (None, True, "destructive"),
        (False, False, "additive"),
        (None, False, "additive"),
        (False, None, None),
        (None, None, None),
    ],
)
def test_effects_from_hints(read_only: Any, destructive: Any, effects: Any):
    assert effects_from_hints(read_only, destructive) == effects


def test_annotation_options_round_trip_the_table():
    for policy in (
        {"effects": "read_only", "idempotent": True},
        {"effects": "additive", "idempotent": False},
        {"effects": "destructive"},
    ):
        hints = tool_annotations(build_spec(remove, **policy))
        assert annotation_options(hints) == policy
    assert annotation_options({}) == {}
    assert annotation_options(None) == {}
