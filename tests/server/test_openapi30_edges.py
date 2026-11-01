"""3.0 edge cases: dangling refs in gateway mode, bounds, nested and mixed unions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from agent_adapter.core.registry import Registry
from agent_adapter.server._openapi30 import _Converter
from agent_adapter.server.openapi import openapi_document

REF = "#/components/schemas/"
URL = "https://tools.example.com"


class Cat(BaseModel):
    kind: Literal["cat"]
    meows: int


class Dog(BaseModel):
    kind: Literal["dog"]
    barks: int


class Req(BaseModel):
    pet: Annotated[Cat | Dog, Field(discriminator="kind")]


def pets() -> Registry:
    reg = Registry()

    @reg.tool(portable=False)
    def adopt(req: Req) -> str:
        """Adopt a pet."""
        return "ok"

    return reg


def strings(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [s for value in node.values() for s in strings(value)]
    if isinstance(node, list):
        return [s for item in node for s in strings(item)]
    return [node] if isinstance(node, str) else []


def nodes(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        return [node, *(n for value in node.values() for n in nodes(value))]
    if isinstance(node, list):
        return [n for item in node for n in nodes(item)]
    return []


def convert(node: Any, **schemas: Any) -> dict[str, Any]:
    return _Converter(schemas).convert(node)


def test_gateway_mode_has_no_dangling_component_refs() -> None:
    doc = openapi_document(pets(), title="t", version="1", server_url=URL, dialect="3.0-gateway")
    names = set(doc["components"]["schemas"])
    targets = [s.removeprefix(REF) for s in strings(doc) if s.startswith(REF)]
    assert targets and all(t in names for t in targets), (targets, names)


def test_a_3_0_document_has_no_discriminator_without_a_union() -> None:
    for dialect in ("3.0", "3.0-gateway"):
        doc = openapi_document(pets(), title="t", version="1", server_url=URL, dialect=dialect)
        assert not [n for n in nodes(doc) if "discriminator" in n], dialect


def test_the_stricter_lower_bound_wins() -> None:
    assert convert({"type": "integer", "minimum": 10, "exclusiveMinimum": 5}) == {
        "type": "integer",
        "minimum": 10,
    }
    assert convert({"exclusiveMinimum": 5, "minimum": 10}) == {"minimum": 10}
    assert convert({"minimum": 5, "exclusiveMinimum": 5}) == {
        "minimum": 5,
        "exclusiveMinimum": True,
    }
    assert convert({"exclusiveMaximum": 5, "maximum": 3}) == {"maximum": 3}
    assert convert({"maximum": 9, "exclusiveMaximum": 5}) == {
        "maximum": 5,
        "exclusiveMaximum": True,
    }


def test_a_nested_union_inside_the_one_non_null_branch_is_kept() -> None:
    inner = {"anyOf": [{"const": "a"}, {"const": "b"}]}
    out = convert({"anyOf": [inner, {"type": "null"}]})
    assert out == {"type": "string", "enum": ["a", "b"], "nullable": True}


def test_any_of_with_one_of_widens_to_their_union() -> None:
    # 3.0 has no intersection; the union of both lists accepts every valid value.
    out = convert({"anyOf": [{"const": "a"}, {"const": "b"}], "oneOf": [{"const": "b"}]})
    assert out == {"type": "string", "enum": ["a", "b"]}


def test_mixed_and_null_only_unions_accept_any_value() -> None:
    mixed = convert({"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]})
    assert mixed == {"nullable": True}
    assert convert({"oneOf": [{"type": "null"}, {"const": None}]}) == {
        "nullable": True,
        "enum": [None],
    }
