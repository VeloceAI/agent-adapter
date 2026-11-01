import json
import warnings
from enum import StrEnum
from typing import Annotated, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.warnings import UnsupportedFieldAttributeWarning

from agent_adapter.core.errors import SchemaNotPortable
from agent_adapter.schema import transforms
from agent_adapter.schema.transforms import (
    CHILD_KEYS,
    inline_refs,
    to_anthropic_input_schema,
    to_gemini_parameters,
    to_openai_parameters,
    to_plain,
)


class Court(StrEnum):
    SC = "sc"
    HC = "hc"


class Filter(BaseModel):
    court: Court = Court.SC
    year: int | None = None


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="The search text.")
    top_k: int = 8
    flt: Filter | None = None


SCHEMA = SearchArgs.model_json_schema()


def test_inline_refs_removes_defs_and_keeps_siblings():
    out = inline_refs(SCHEMA)
    assert "$defs" not in json.dumps(out) and "$ref" not in json.dumps(out)
    court = out["properties"]["flt"]["anyOf"][0]["properties"]["court"]
    assert court == {"enum": ["sc", "hc"], "title": "Court", "type": "string", "default": "sc"}


def test_inline_refs_flattens_single_all_of():
    schema = {
        "type": "object",
        "properties": {"a": {"allOf": [{"$ref": "#/$defs/X"}], "description": "d"}},
        "$defs": {"X": {"type": "string"}},
    }
    assert inline_refs(schema)["properties"]["a"] == {"type": "string", "description": "d"}


def test_inline_refs_rejects_recursion():
    schema = {
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/N"}},
        "$defs": {"N": {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}}},
    }
    with pytest.raises(SchemaNotPortable, match="recursive"):
        inline_refs(schema)


def test_to_plain_drops_title_keyword_but_not_a_property_named_title():
    schema = {
        "type": "object",
        "title": "T",
        "properties": {"title": {"const": "x", "type": "string", "title": "Title"}},
    }
    assert to_plain(schema) == {
        "type": "object",
        "properties": {"title": {"enum": ["x"], "type": "string"}},
    }


def test_openai_strict_requires_all_and_makes_optional_nullable():
    assert to_openai_parameters(SCHEMA) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "top_k", "flt"],
        "properties": {
            "query": {"description": "The search text.", "type": "string"},
            "top_k": {
                "description": "Default: 8.",
                "anyOf": [{"type": "integer"}, {"type": "null"}],
            },
            "flt": {
                "description": "Default: null.",
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["court", "year"],
                        "properties": {
                            "court": {
                                "description": 'Default: "sc".',
                                "anyOf": [
                                    {"enum": ["sc", "hc"], "type": "string"},
                                    {"type": "null"},
                                ],
                            },
                            "year": {
                                "description": "Default: null.",
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                            },
                        },
                    },
                    {"type": "null"},
                ],
            },
        },
    }


def test_openai_strict_moves_description_to_nullable_wrapper():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "description": "How many."}},
    }
    assert to_openai_parameters(schema)["properties"]["n"] == {
        "description": "How many.",
        "anyOf": [{"type": "integer"}, {"type": "null"}],
    }


def test_openai_strict_appends_dropped_default_to_existing_description():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "How many.", "default": 3},
            "f": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "default": {"a": "x"},
            },
        },
    }
    props = to_openai_parameters(schema)["properties"]
    assert props["n"]["description"] == "How many. Default: 3."
    assert props["f"]["description"] == 'Default: {"a": "x"}.'
    assert "default" not in json.dumps(props)


def _post_order(node: dict, fn):
    """Children before parent: the opposite of the production traversal."""
    out = dict(node)
    for key, shape in CHILD_KEYS:
        child = out.get(key)
        if shape == "map" and isinstance(child, dict):
            out[key] = {k: _post_order(v, fn) for k, v in child.items()}
        elif shape == "one" and isinstance(child, dict):
            out[key] = _post_order(child, fn)
        elif shape == "list" and isinstance(child, list):
            out[key] = [_post_order(v, fn) for v in child]
    return fn(out)


def test_only_property_defaults_are_described():
    with warnings.catch_warnings():
        # pydantic warns that `default` inside list items / Optional branches has no effect.
        warnings.simplefilter("ignore", UnsupportedFieldAttributeWarning)
        model = create_model(
            "M",
            xs=(list[Annotated[int, Field(default=3)]], ...),
            y=(Optional[Annotated[int, Field(default=3)]], None),  # noqa: UP045
        )
    schema = model.model_json_schema()
    assert schema["properties"]["xs"]["items"]["default"] == 3  # the case under test exists
    for render in (to_openai_parameters, to_gemini_parameters):
        out = render(schema)
        assert "Default: 3." not in json.dumps(out)
        assert "default" not in json.dumps(out)
        assert out["properties"]["y"]["description"] == "Default: null."


NESTED = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "default": [],
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer", "default": 1}},
            },
        },
        "flt": {
            "anyOf": [
                {"type": "object", "properties": {"c": {"type": "string", "default": "sc"}}},
                {"type": "null"},
            ],
            "default": None,
        },
    },
}


@pytest.mark.parametrize("render", [to_openai_parameters, to_gemini_parameters])
def test_default_descriptions_do_not_depend_on_traversal_order(render, monkeypatch):
    expected = render(NESTED)
    assert "default" not in json.dumps(expected)
    assert json.dumps(expected).count("Default: ") == 4
    monkeypatch.setattr(transforms, "_map_nodes", _post_order)
    assert render(NESTED) == expected


def test_openai_non_strict_is_plain():
    assert to_openai_parameters(SCHEMA, strict=False) == to_plain(SCHEMA)


def test_gemini_keeps_only_supported_keywords():
    assert to_gemini_parameters(SCHEMA) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"description": "The search text.", "type": "string"},
            "top_k": {"type": "integer", "description": "Default: 8."},
            "flt": {
                "description": "Default: null.",
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "court": {
                                "enum": ["sc", "hc"],
                                "type": "string",
                                "description": 'Default: "sc".',
                            },
                            "year": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "description": "Default: null.",
                            },
                        },
                    },
                    {"type": "null"},
                ],
            },
        },
    }


def test_anthropic_keeps_defaults():
    out = to_anthropic_input_schema(SCHEMA)
    assert out == to_plain(SCHEMA)
    assert out["properties"]["top_k"] == {"default": 8, "type": "integer"}
