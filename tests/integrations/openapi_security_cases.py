"""OpenAPI importer security and robustness cases (review fix round 1).

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_adapter.core.errors import ConfigError, ToolArgsInvalid, ToolError, ToolFailed
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations.openapi import HeaderAuth, QueryAuth, _refs, import_openapi
from agent_adapter.integrations.openapi._http import OperationCall
from agent_adapter.integrations.openapi._schema import Operation
from agent_adapter.runtime.loop_local import LoopLocal
from tests.integrations.openapi_support import PETS, Wire, pets
from tests.integrations.openapi_work import run_counted


def one(path: str, op: dict[str, Any] | None = None, method: str = "get") -> dict[str, Any]:
    """A document with one operation at `path`."""
    return {**PETS, "paths": {path: {method: {"operationId": "op", **(op or {})}}}}


def param(name: str, where: str = "path", **schema: Any) -> dict[str, Any]:
    return {
        "name": name,
        "in": where,
        "required": where == "path",
        "schema": schema or {"type": "string"},
    }


async def no_sleep(_: float) -> None:
    return None


# --- C1: paths may not leave the server ----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "http://attacker.example/steal",
        "https://attacker.example/steal",
        "//attacker.example/steal",
        "attacker.example/steal",
        "",
        "/a?x=1",
        "/a#frag",
        "/a/../admin",
        "/./a",
        "/a/.",
        "/a\\b",
    ],
)
def test_paths_that_could_leave_the_server_are_rejected_at_import(path: str):
    wire = Wire()
    with pytest.raises(ConfigError, match="path"):
        import_openapi(
            Registry(), one(path), client_factory=wire.factory, auth=HeaderAuth("K", "v")
        )
    assert wire.requests == []


async def test_the_call_checks_the_origin_before_sending_credentials():
    wire = Wire()
    clients: LoopLocal[Any] = LoopLocal(lambda: wire.factory(base_url="https://api.example.com/v1"))
    sent: list[dict[str, str]] = []

    class Spy(HeaderAuth):
        async def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
            await super().apply(headers, params)
            sent.append(dict(headers))

    op = Operation(
        key="evil",
        name="evil",
        method="get",
        path="https://attacker.example/steal",
        description="x",
        input_schema={"type": "object", "properties": {}},
        output_schema=None,
        bindings=(),
        body="none",
        tags=(),
    )
    call = OperationCall(op, clients, Spy("X-Key", "s3cr3t"), base_url="https://api.example.com/v1")
    with pytest.raises(ConfigError, match=r"attacker\.example"):
        await call()
    assert wire.requests == [] and sent == []  # auth is never even applied


@pytest.mark.parametrize("path", ["/a/{missing}", "/a/{id}/{id}"])
def test_path_placeholders_must_match_path_parameters(path: str):
    with pytest.raises(ConfigError, match=r"placeholder|path parameter"):
        import_openapi(
            Registry(), one(path, {"parameters": [param("id")]}), client_factory=Wire().factory
        )


def test_a_path_parameter_without_a_placeholder_is_rejected():
    with pytest.raises(ConfigError, match="no placeholder"):
        import_openapi(
            Registry(), one("/a", {"parameters": [param("id")]}), client_factory=Wire().factory
        )


# --- C2: path values cannot traverse ---------------------------------------------------------


@pytest.mark.parametrize("value", [".", ".."])
async def test_dot_segments_in_path_values_are_invalid_arguments(value: str):
    registry, wire = pets(document=one("/users/{id}", {"parameters": [param("id")]}))
    with pytest.raises(ToolArgsInvalid, match="id"):
        await registry.get("op").invoke({"id": value})
    assert wire.requests == []


async def test_dots_in_path_values_are_percent_encoded():
    doc = one("/a/{x}/b/{y}", {"parameters": [param("x"), param("y")]})
    registry, wire = pets(document=doc)
    await registry.get("op").invoke({"x": "...", "y": "v1.2"})
    assert wire.last.url.raw_path == b"/v1/a/%2E%2E%2E/b/v1%2E2"
    with pytest.raises(ToolArgsInvalid):
        await registry.get("op").invoke({"x": "..", "y": ".."})
    assert len(wire.requests) == 1


async def test_a_null_path_value_is_an_invalid_argument():
    doc = one("/users/{id}", {"parameters": [param("id", type=["string", "null"])]})
    registry, wire = pets(document=doc)
    with pytest.raises(ToolArgsInvalid, match="id"):
        await registry.get("op").invoke({"id": None})
    assert wire.requests == []


# --- header and cookie values ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "value"), [("header", "a\r\nX-Evil: 1"), ("header", "café"), ("cookie", "a\nb")]
)
async def test_header_and_cookie_values_must_be_printable_ascii(where: str, value: str):
    registry, wire = pets(document=one("/a", {"parameters": [param("v", where)]}), retries=2)
    with pytest.raises(ToolArgsInvalid, match="v"):
        await Executor(registry, sleep=no_sleep).run("op", {"v": value})
    assert wire.requests == []


# --- I1: expansion bombs fail fast ----------------------------------------------------------


def ref_chain(depth: int, fanout: int) -> dict[str, Any]:
    schemas = {f"S{depth}": {"type": "string"}}
    for level in range(depth - 1, -1, -1):
        child = {"$ref": f"#/components/schemas/S{level + 1}"}
        schemas[f"S{level}"] = {
            "type": "object",
            "properties": {f"p{i}": child for i in range(fanout)},
        }
    body = {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/S0"}}}}
    return {**one("/a", {"requestBody": body}, "post"), "components": {"schemas": schemas}}


# Lines of agent_adapter run (deterministic, unlike seconds): ~5 000 to skip a 4**10 ref
# chain, ~600 to refuse the alias bomb; expanding the chain runs over 50 M.
BOMB_LINES = 100_000


def test_an_exponential_ref_chain_skips_its_operation_in_bounded_work(tmp_path: Path):
    result = run_counted(ref_chain(10, 4), tmp_path, cap=BOMB_LINES)
    assert result["names"] == [], result["outcome"]
    skipped = "1 OpenAPI operations skipped: op: an OpenAPI schema is too large"
    assert any(warning.startswith(skipped) for warning in result["warnings"]), result


def test_a_shared_ref_within_the_cap_still_imports():
    [spec] = import_openapi(Registry(), ref_chain(4, 3), client_factory=Wire().factory)
    assert list(spec.json_schema()["properties"]) == ["p0", "p1", "p2"]


YAML_BOMB = """openapi: 3.0.0
info: {title: bomb, version: '1'}
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
e: &e [*d,*d,*d,*d,*d,*d,*d,*d,*d]
f: &f [*e,*e,*e,*e,*e,*e,*e,*e,*e]
g: &g [*f,*f,*f,*f,*f,*f,*f,*f,*f]
h: &h [*g,*g,*g,*g,*g,*g,*g,*g,*g]
i: &i [*h,*h,*h,*h,*h,*h,*h,*h,*h]
paths: {}
"""


def test_a_yaml_alias_bomb_is_a_config_error_in_bounded_work(tmp_path: Path):
    result = run_counted(YAML_BOMB, tmp_path, cap=BOMB_LINES)
    assert result["outcome"].startswith("ConfigError") and "too large" in result["outcome"]


def test_aliases_count_towards_the_document_value_cap(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_refs, "MAX_DOCUMENT_NODES", 100)  # far below the expanded bomb
    with pytest.raises(ConfigError, match="too large"):
        import_openapi(Registry(), YAML_BOMB, client_factory=Wire().factory)


def test_a_too_deep_document_is_a_config_error():
    node: dict[str, Any] = {"type": "string"}
    for _ in range(3000):
        node = {"type": "object", "properties": {"x": node}}
    doc = one("/a", {"requestBody": {"content": {"application/json": {"schema": node}}}}, "post")
    with pytest.raises(ConfigError, match="too deep"):
        import_openapi(Registry(), doc, client_factory=Wire().factory)


@pytest.mark.parametrize(
    "paths",
    [
        {"/a": ["get"]},
        {"/a": {"get": {"parameters": "id"}}},
        {"/a": {"get": {"parameters": [{"name": "q", "in": "query", "schema": "string"}]}}},
        {"/a": {"post": {"requestBody": {"content": ["application/json"]}}}},
        {"/a": {"post": {"requestBody": "json"}}},
        {"/a": {"get": {"responses": ["200"]}}},
        {"/a": {"get": {"tags": "x", "parameters": [{"name": 3, "in": None}]}}},
    ],
)
def test_malformed_shapes_are_config_errors(paths: dict[str, Any]):
    with pytest.raises(ConfigError):
        import_openapi(Registry(), {**PETS, "paths": paths}, client_factory=Wire().factory)


def test_an_oversized_response_schema_is_dropped_with_a_warning(caplog: pytest.LogCaptureFixture):
    big = ref_chain(10, 4)
    schema = {"$ref": "#/components/schemas/S0"}
    big["paths"]["/a"] = {
        "get": {
            "operationId": "op",
            "responses": {
                "200": {"description": "x", "content": {"application/json": {"schema": schema}}}
            },
        }
    }
    with caplog.at_level(logging.WARNING):
        [spec] = import_openapi(Registry(), big, client_factory=Wire().factory)
    assert spec.output_schema is None and "response schema" in caplog.text


# --- I2: QueryAuth secrets never reach any log ------------------------------------------------


@pytest.mark.parametrize("status", [200, 302, 404, 429, 500])
async def test_query_auth_secrets_are_redacted_from_every_log(
    status: int, caplog: pytest.LogCaptureFixture
):
    wire = Wire(httpx.Response(status, text="body"))
    registry, _ = pets(wire=wire, auth=QueryAuth("api_key", "SECRET-123"))
    # the outcome is not under test, only the logs
    with caplog.at_level(logging.DEBUG), contextlib.suppress(ToolError, ToolFailed):
        await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 1})
    assert wire.last.url.params["api_key"] == "SECRET-123"  # it was sent
    assert any("HTTP Request" in r.getMessage() for r in caplog.records)
    for record in caplog.records:
        assert "SECRET-123" not in record.getMessage()
        assert "SECRET-123" not in caplog.text


# --- retries and idempotency ----------------------------------------------------------------


async def test_5xx_on_a_non_idempotent_method_is_not_retried():
    wire = Wire(httpx.Response(503), httpx.Response(201, json={"id": 1}))
    registry, _ = pets(wire=wire, retries=3)
    with pytest.raises(ToolError, match="not retried"):
        await Executor(registry, sleep=no_sleep).run("createPet", {"body_name": "rex"})
    assert len(wire.requests) == 1


async def test_429_and_connect_errors_on_a_non_idempotent_method_are_retried():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    wire = Wire(httpx.Response(429), refused, httpx.Response(201, json={"id": 1}))
    registry, _ = pets(wire=wire, retries=3)
    assert await Executor(registry, sleep=no_sleep).run("createPet", {"body_name": "rex"}) == {
        "id": 1
    }
    assert len(wire.requests) == 3


async def test_idempotent_true_in_policy_opts_a_post_into_5xx_retries():
    wire = Wire(httpx.Response(503), httpx.Response(201, json={"id": 1}))
    registry, _ = pets(wire=wire, retries=1, idempotent=True, operations=["createPet"])
    assert await Executor(registry, sleep=no_sleep).run("createPet", {"body_name": "rex"}) == {
        "id": 1
    }


async def test_a_read_timeout_on_a_post_is_not_retried():
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    registry, wire = pets(wire=Wire(slow), retries=2)
    with pytest.raises(ToolError, match="ReadTimeout"):
        await Executor(registry, sleep=no_sleep).run("createPet", {"body_name": "rex"})
    assert len(wire.requests) == 1


# --- response size and total time -------------------------------------------------------------


async def test_a_response_larger_than_the_cap_is_a_tool_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("agent_adapter.integrations.openapi._decode.MAX_RESPONSE_BYTES", 1000)
    registry, _ = pets(wire=Wire(httpx.Response(200, content=b"x" * 5000)))
    with pytest.raises(ToolError, match="larger than 1000 bytes"):
        await registry.get("getPet").invoke({"petId": 1})


async def test_the_timeout_bounds_the_whole_request():
    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[override]
            for _ in range(100):
                await asyncio.sleep(0.02)
                yield b"x"

    registry, _ = pets(wire=Wire(httpx.Response(200, stream=Drip())), timeout=0.1)
    with pytest.raises(httpx.TimeoutException, match=r"0.1s"):
        await registry.get("getPet").invoke({"petId": 1})


async def test_a_network_path_reference_stays_on_the_server():
    """`//host/x` built by hand still resolves under base_url, never to `host`."""
    wire = Wire()
    clients: LoopLocal[Any] = LoopLocal(lambda: wire.factory(base_url="https://api.example.com/v1"))
    op = Operation(
        key="odd",
        name="odd",
        method="get",
        path="//attacker.example/steal",
        description="x",
        input_schema={"type": "object", "properties": {}},
        output_schema=None,
        bindings=(),
        body="none",
        tags=(),
    )
    await OperationCall(op, clients, None, base_url="https://api.example.com/v1")()
    assert wire.last.url.host == "api.example.com"


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("openapi: 3.0.0\npaths: {}\nx: &x [*x]\n", "contains itself"),
        ('{"openapi": "3.0.0", "x": ' + "[" * 100_000, "nested too deeply"),
    ],
)
def test_self_containing_and_unparsably_deep_documents_are_config_errors(source: str, match: str):
    with pytest.raises(ConfigError, match=match):
        import_openapi(Registry(), source, client_factory=Wire().factory)


def test_path_item_refs_and_odd_parameter_content_are_handled():
    doc = {
        **PETS,
        "components": {"pathItems": {"P": {"get": {"operationId": "viaRef"}}}},
        "paths": {"/p": {"$ref": "#/components/pathItems/P"}},
    }
    [spec] = import_openapi(Registry(), doc, client_factory=Wire().factory)
    assert spec.name == "viaRef"
    odd = one("/a", {"parameters": [{"name": "q", "in": "query", "content": ["x"]}]})
    with pytest.raises(ConfigError, match="malformed OpenAPI operation op"):
        import_openapi(Registry(), odd, client_factory=Wire().factory)
