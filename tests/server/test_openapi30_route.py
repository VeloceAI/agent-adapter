"""GET /openapi-3.0.json: the caller's tools as OpenAPI 3.0.3, behind the same auth."""

from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.server.auth import Principal
from tests.server.helpers import call, make_service


def unions(node: Any) -> int:
    if isinstance(node, dict):
        here = sum(key in node for key in ("oneOf", "anyOf", "allOf"))
        return here + sum(unions(value) for value in node.values())
    if isinstance(node, list):
        return sum(unions(item) for item in node)
    return 0


async def test_the_3_0_route_serves_a_union_free_document() -> None:
    reply = await call(make_service(), "GET", "/openapi-3.0.json")
    assert reply.status == 200 and reply.body["openapi"] == "3.0.3"
    assert unions(reply.body) == 0
    assert {"/tools/add/invoke", "/readyz", "/openapi.json", "/openapi-3.0.json"} <= set(
        reply.body["paths"]
    )
    full = await call(make_service(), "GET", "/openapi.json")
    assert full.body["openapi"] == "3.1.0" and unions(full.body) > 0


async def test_the_3_0_route_needs_a_principal_and_hides_forbidden_tools() -> None:
    async def only_add(principal: Principal, spec: ToolSpec, args: Any) -> bool:
        return spec.name == "add"

    service = make_service(authorize=only_add)
    assert (await call(service, "GET", "/openapi-3.0.json", token="wrong")).status == 401
    assert (await call(service, "POST", "/openapi-3.0.json")).status == 405
    doc = await call(service, "GET", "/openapi-3.0.json")
    assert [p for p in doc.body["paths"] if p.endswith("/invoke")] == ["/tools/add/invoke"]


def test_service_openapi_takes_a_dialect() -> None:
    doc = make_service().openapi(server_url="https://x.example", dialect="3.0-gateway")
    assert doc["openapi"] == "3.0.3" and doc["servers"] == [{"url": "https://x.example"}]
    assert all(path.endswith("/invoke") for path in doc["paths"])
