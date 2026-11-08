from typing import Any

from pydantic import BaseModel

from agent_adapter.core.registry import Registry
from agent_adapter.server.openapi import openapi_document


class Address(BaseModel):
    city: str


class Person(BaseModel):
    name: str
    home: Address


def registry() -> Registry:
    reg = Registry()

    @reg.tool(tags=["people"])
    def register(person: Person) -> Person:
        """Register a person.

        Stores the record.
        """
        return person

    @reg.tool
    def ping() -> None:
        """Ping."""

    @reg.tool(targets=["mcp"])
    def hidden() -> str:
        """Not exported over HTTP."""
        return "x"

    return reg


def refs(node: Any) -> list[str]:
    if isinstance(node, dict):
        found = [node["$ref"]] if isinstance(node.get("$ref"), str) else []
        return found + [r for value in node.values() for r in refs(value)]
    if isinstance(node, list):
        return [r for item in node for r in refs(item)]
    return []


def resolve(document: dict[str, Any], ref: str) -> Any:
    node: Any = document
    for part in ref.removeprefix("#/").split("/"):
        node = node[part]
    return node


def test_one_invoke_operation_per_http_tool():
    doc = openapi_document(registry(), title="t", version="1.2.3")
    assert doc["openapi"] == "3.1.0"
    assert doc["info"] == {"title": "t", "version": "1.2.3"}
    invokes = [path for path in doc["paths"] if path.endswith("/invoke")]
    assert invokes == ["/tools/register/invoke", "/tools/ping/invoke"]
    op = doc["paths"]["/tools/register/invoke"]["post"]
    assert op["operationId"] == "invoke_register"
    assert op["summary"] == "Register a person."
    assert op["tags"] == ["people"]
    assert {p["name"] for p in op["parameters"]} == {"Idempotency-Key", "X-Run-Id", "X-Thread-Id"}
    assert set(op["responses"]) >= {"200", "202", "422", "429", "504"}
    assert "servers" not in doc


def test_nested_schemas_move_to_components_and_every_ref_resolves():
    doc = openapi_document(registry(), title="t", version="1", server_url="https://x.test")
    assert doc["servers"] == [{"url": "https://x.test"}]
    schemas = doc["components"]["schemas"]
    assert {"register__Person", "register__Address", "register__result__Address"} <= set(schemas)
    body = doc["paths"]["/tools/register/invoke"]["post"]["requestBody"]
    args = body["content"]["application/json"]["schema"]["properties"]["arguments"]
    assert "$defs" not in args
    for ref in refs(doc):
        assert ref.startswith("#/components/")
        assert resolve(doc, ref) is not None


def test_result_schema_and_the_fixed_routes():
    doc = openapi_document(registry(), title="t", version="1")
    ping = doc["paths"]["/tools/ping/invoke"]["post"]["responses"]["200"]
    ok = ping["content"]["application/json"]["schema"]
    assert ok["properties"]["status"] == {"const": "ok"}
    assert {
        "/healthz",
        "/readyz",
        "/tools",
        "/tools/{name}",
        "/approvals",
        "/approvals/{approval_id}",
        "/approvals/{approval_id}/decide",
        "/runs/{run_id}/events",
        "/runs/{run_id}/cost",
        "/budget",
        "/budget/{scope}",
    } <= set(doc["paths"])
    scoped = doc["paths"]["/budget/{scope}"]["get"]
    assert "400" in scoped["responses"] and "400" in doc["paths"]["/budget"]["get"]["responses"]
    assert scoped["parameters"][0]["schema"]["enum"] == ["principal", "user", "team", "global"]
    assert doc["paths"]["/healthz"]["get"]["security"] == []
    assert doc["paths"]["/readyz"]["get"]["security"] == []
    ready = doc["paths"]["/readyz"]["get"]["responses"]
    assert set(ready) == {"200", "503"}
    assert doc["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"
    assert doc["components"]["schemas"]["DecideBody"]["required"] == ["approved"]
    decide = doc["paths"]["/approvals/{approval_id}/decide"]["post"]["responses"]
    assert "note" in doc["components"]["schemas"]["ApprovedElsewhere"]["properties"]
    assert set(decide) >= {"200", "202"}
