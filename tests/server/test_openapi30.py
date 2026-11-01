"""The 3.0.3 renderings of the OpenAPI document: no unions, nullable flags, flattened unions."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.server.openapi import openapi_document

COMPOSITION = ("oneOf", "anyOf", "allOf")
URL = "https://tools.example.com"


class Colour(StrEnum):
    RED = "red"
    BLUE = "blue"


class Card(BaseModel):
    kind: Literal["card"]
    last4: str


class Bank(BaseModel):
    kind: Literal["bank"]
    iban: str


class Node(BaseModel):
    label: str
    child: Node | None = None


class Address(BaseModel):
    city: str


class Order(BaseModel):
    amount: Annotated[float, Field(gt=0)]
    note: str | None = None
    home: Address | None = None
    method: Annotated[Card | Bank, Field(discriminator="kind")]
    code: int | str
    pair: tuple[int, str]
    colour: Colour = Colour.RED
    tree: Node | None = None


def registry() -> Registry:
    reg = Registry()

    @reg.tool(portable=False)  # every awkward shape on purpose
    def place(order: Order, priority: Literal["low", "high"] = "low") -> Order:
        """Place an order."""
        return order

    @reg.tool
    def ping() -> None:
        """Ping."""

    return reg


def walk(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        found = [(path, node)]
        return found + [x for k, v in node.items() for x in walk(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in walk(v, f"{path}[{i}]")]
    return []


def schemas_of(doc: dict[str, Any]) -> dict[str, Any]:
    return dict(doc["components"]["schemas"])


@pytest.mark.parametrize("dialect", ["3.0", "3.0-gateway"])
def test_a_3_0_document_has_no_composition_and_no_3_1_keywords(dialect: Any) -> None:
    doc = openapi_document(registry(), title="t", version="1", server_url=URL, dialect=dialect)
    assert doc["openapi"] == "3.0.3"
    for path, node in walk(doc):
        assert not set(COMPOSITION) & set(node), path
        assert "const" not in node and "prefixItems" not in node and "$defs" not in node, path
        assert not isinstance(node.get("type"), list), path
        assert node.get("type") != "null", path
    json.dumps(doc, allow_nan=False)


def test_the_default_stays_3_1_with_its_unions() -> None:
    doc = openapi_document(registry(), title="t", version="1")
    assert doc["openapi"] == "3.1.0"
    assert any(set(COMPOSITION) & set(node) for _, node in walk(doc))


def test_optional_fields_become_nullable() -> None:
    order = schemas_of(openapi_document(registry(), title="t", version="1", dialect="3.0"))[
        "place__Order"
    ]["properties"]
    assert order["note"]["type"] == "string" and order["note"]["nullable"] is True
    assert order["home"]["nullable"] is True
    assert order["home"]["properties"]["city"] == {"title": "City", "type": "string"}
    assert order["amount"]["minimum"] == 0 and order["amount"]["exclusiveMinimum"] is True


def test_a_tagged_union_flattens_to_one_object_with_its_tag_as_an_enum() -> None:
    order = schemas_of(openapi_document(registry(), title="t", version="1", dialect="3.0"))[
        "place__Order"
    ]["properties"]
    method = order["method"]
    assert method["type"] == "object"
    assert method["properties"]["kind"]["enum"] == ["card", "bank"]
    assert set(method["properties"]) == {"kind", "last4", "iban"}
    assert method["required"] == ["kind"]
    assert "discriminator" not in method


def test_the_3_1_discriminator_mapping_points_into_components() -> None:
    doc = openapi_document(registry(), title="t", version="1")
    mapping = schemas_of(doc)["place__Order"]["properties"]["method"]["discriminator"]["mapping"]
    assert mapping == {
        "card": "#/components/schemas/place__Card",
        "bank": "#/components/schemas/place__Bank",
    }


def test_mixed_unions_accept_any_value_and_tuples_become_arrays() -> None:
    order = schemas_of(openapi_document(registry(), title="t", version="1", dialect="3.0"))[
        "place__Order"
    ]["properties"]
    assert "type" not in order["code"]
    assert order["pair"]["type"] == "array" and order["pair"]["items"] == {}
    assert order["pair"]["minItems"] == order["pair"]["maxItems"] == 2


def test_a_recursive_optional_model_keeps_a_ref_and_terminates() -> None:
    schemas = schemas_of(openapi_document(registry(), title="t", version="1", dialect="3.0"))
    child = schemas["place__Node"]["properties"]["child"]
    assert child == {"$ref": "#/components/schemas/place__Node"}


def test_const_becomes_a_typed_enum_and_server_unions_flatten() -> None:
    schemas = schemas_of(openapi_document(registry(), title="t", version="1", dialect="3.0"))
    assert schemas["InvokeOk"]["properties"]["status"] == {"enum": ["ok"], "type": "string"}
    decision = schemas["Approval"]["properties"]["decision"]
    assert decision["type"] == "object" and decision["nullable"] is True
    assert decision["properties"]["edited_args"] == {"type": "object", "nullable": True}
    decide = openapi_document(registry(), title="t", version="1", dialect="3.0")["paths"][
        "/approvals/{approval_id}/decide"
    ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert decide["properties"]["status"]["enum"] == ["ok", "denied"]


def test_the_gateway_dialect_keeps_only_invoke_operations_and_their_schemas() -> None:
    doc = openapi_document(
        registry(), title="t", version="1", server_url=URL, dialect="3.0-gateway"
    )
    assert set(doc["paths"]) == {"/tools/place/invoke", "/tools/ping/invoke"}
    assert "security" not in doc and "securitySchemes" not in doc["components"]
    assert "responses" not in doc["components"]
    assert doc["servers"] == [{"url": URL}]
    responses = doc["paths"]["/tools/place/invoke"]["post"]["responses"]
    assert set(responses["400"]) == {"description"}
    assert "Approval" not in doc["components"]["schemas"]
    assert {"InvokePending", "place__Order", "place__Node"} <= set(doc["components"]["schemas"])
    for _, node in walk(doc):
        if "content" in node:
            assert set(node["content"]) == {"application/json"}


def test_the_3_0_dialect_keeps_every_route_and_its_bearer_scheme() -> None:
    doc = openapi_document(registry(), title="t", version="1", dialect="3.0")
    full = openapi_document(registry(), title="t", version="1")
    assert set(doc["paths"]) == set(full["paths"])
    assert "/readyz" in doc["paths"]
    ready = doc["paths"]["/readyz"]["get"]["responses"]["200"]["content"]["application/json"]
    assert doc["components"]["schemas"]["Ready"]["properties"]["status"] == {
        "enum": ["ready"],
        "type": "string",
    }
    assert ready["schema"] == {"$ref": "#/components/schemas/Ready"}
    assert doc["security"] == full["security"]


def test_the_downgrade_leaves_the_3_1_document_untouched() -> None:
    reg = registry()
    before = openapi_document(reg, title="t", version="1")
    openapi_document(reg, title="t", version="1", dialect="3.0")
    assert openapi_document(reg, title="t", version="1") == before


def test_a_bad_dialect_or_a_gateway_without_url_is_refused() -> None:
    with pytest.raises(ConfigError, match="dialect"):
        openapi_document(registry(), title="t", version="1", dialect="2.0")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="server_url"):
        openapi_document(registry(), title="t", version="1", dialect="3.0-gateway")


@pytest.mark.parametrize("dialect", ["3.1", "3.0", "3.0-gateway"])
def test_the_document_passes_openapi_spec_validator(dialect: Any) -> None:
    validator = pytest.importorskip("openapi_spec_validator")
    doc = openapi_document(registry(), title="t", version="1", server_url=URL, dialect=dialect)
    validator.validate(doc)


def convert(node: Any, **schemas: Any) -> dict[str, Any]:
    from agent_adapter.server._openapi30 import _Converter

    return _Converter(schemas).convert(node)


REF = "#/components/schemas/"


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (True, {}),
        (False, {"not": {}}),
        ("junk", {}),
        ({"type": "null"}, {"nullable": True, "enum": [None]}),
        ({"type": ["string", "integer"]}, {}),
        ({"anyOf": [{"type": "null"}]}, {"nullable": True, "enum": [None]}),
        ({"enum": ["a", None]}, {"enum": ["a"], "nullable": True, "type": "string"}),
        ({"enum": [1, "a"]}, {"enum": [1, "a"]}),
        ({"examples": ["x"], "type": "string"}, {"example": "x", "type": "string"}),
        (
            {"type": "array", "prefixItems": [{"type": "integer"}], "items": False},
            {"type": "array", "items": {"type": "integer"}},
        ),
        ({"type": "array"}, {"type": "array", "items": {}}),
        (
            {"type": "object", "required": [], "$comment": "x", "x-tag": 1},
            {"type": "object", "x-tag": 1},
        ),
        ({"exclusiveMaximum": 5}, {"maximum": 5, "exclusiveMaximum": True}),
        (
            {"additionalProperties": {"type": ["integer", "null"]}},
            {"additionalProperties": {"type": "integer", "nullable": True}},
        ),
        ({"not": {"const": 1}}, {"not": {"enum": [1], "type": "integer"}}),
    ],
)
def test_single_keywords_convert(node: Any, expected: Any) -> None:
    assert convert(node) == expected


def test_unions_of_one_kind_merge_their_enums_and_items() -> None:
    assert convert({"anyOf": [{"const": "a"}, {"const": "b"}]}) == {
        "type": "string",
        "enum": ["a", "b"],
    }
    same = {"type": "array", "items": {"type": "string"}}
    assert convert({"oneOf": [same, same]}) == same
    mixed = convert({"oneOf": [same, {"type": "array", "items": {"type": "integer"}}]})
    assert mixed == {"type": "array", "items": {}}
    assert convert({"anyOf": [{"type": "string", "format": "date"}, {"type": "string"}]}) == {
        "type": "string"
    }


def test_object_unions_merge_clashing_properties() -> None:
    a = {
        "type": "object",
        "properties": {"k": {"type": "string", "enum": ["x"]}, "n": {"type": "integer"}},
    }
    b = {
        "type": "object",
        "properties": {"k": {"type": "string", "enum": ["y"]}, "n": {"type": "string"}},
    }
    out = convert({"anyOf": [a, b]})
    assert out["properties"] == {"k": {"type": "string", "enum": ["x", "y"]}, "n": {}}
    assert "discriminator" not in out  # the tag is not required in every variant


def test_all_of_merges_properties_and_required() -> None:
    base = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    node = {
        "allOf": [{"$ref": REF + "Base"}, {"properties": {"b": {"type": "integer"}}}, "junk"],
        "required": ["b"],
        "description": "both",
    }
    out = convert(node, Base=base)
    assert out == {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["b", "a"],
        "description": "both",
    }


def test_refs_with_siblings_inline_and_unknown_refs_stay() -> None:
    target = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert convert({"$ref": REF + "T", "description": "d"}, T=target) == {
        **target,
        "description": "d",
    }
    assert convert({"$ref": REF + "Missing", "description": "d"}) == {"$ref": REF + "Missing"}
    assert convert({"$ref": "https://x/y", "nullable": True}) == {"$ref": "https://x/y"}
    unresolved = convert({"anyOf": [{"$ref": REF + "Missing"}, {"type": "string"}]})
    assert unresolved == {}
