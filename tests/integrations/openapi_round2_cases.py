"""OpenAPI importer cases from review round 2: cost bounds, redirects, decompression, URLs.

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import gzip
import tracemalloc
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_adapter.core.errors import ConfigError, ToolError, ToolFailed
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.integrations.openapi import HeaderAuth, QueryAuth, _decode, _refs, import_openapi
from agent_adapter.integrations.openapi import _http as http_module
from tests.integrations.openapi_support import PETS, Wire, pets, raw
from tests.integrations.openapi_work import run_counted


async def no_sleep(_: float) -> None:
    return None


def nested(levels: int) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "string", "nullable": True}
    for _ in range(levels):
        node = {
            "type": "object",
            "nullable": True,
            "properties": {"x": node, "y": {"type": "integer"}},
        }
    return node


def post_with(schema: dict[str, Any], **extra: Any) -> dict[str, Any]:
    body = {"content": {"application/json": {"schema": schema}}}
    return {**PETS, **extra, "paths": {"/a": {"post": {"operationId": "op", "requestBody": body}}}}


# --- 1: no exponential work ----------------------------------------------------------------


# Lines of agent_adapter run (deterministic, unlike seconds): ~20 000 for 40 nested levels
# and ~15 000 for 20 operations sharing a cyclic schema. Walking each properties map twice
# per level is exponential: over 50 M.
NESTED_LINES = 100_000


def test_deeply_nested_properties_import_in_linear_work(tmp_path: Path):
    result = run_counted(post_with(nested(40)), tmp_path, cap=NESTED_LINES, schemas=True)
    body = result["schemas"][0]["properties"]["body"]  # a nullable object body is one field
    assert body["type"] == ["object", "null"] and body["properties"]["x"]["type"] == [
        "object",
        "null",
    ]


def test_many_operations_sharing_a_cyclic_schema_import_in_linear_work(tmp_path: Path):
    node = {
        "type": "object",
        "properties": {"child": {"$ref": "#/components/schemas/Node"}, "n": {"type": "integer"}},
    }
    components = {"schemas": {"Node": node}}
    ref = {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Node"}}}}
    paths = {f"/p{i}": {"post": {"operationId": f"op{i}", "requestBody": ref}} for i in range(20)}
    document = {**PETS, "components": components, "paths": paths}
    result = run_counted(document, tmp_path, cap=NESTED_LINES)
    assert len(result["names"]) == 20, result["outcome"]


def test_a_document_wide_inlining_budget_bounds_the_values_built(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_refs, "budgets", lambda values: (5_000, 2_500))
    big = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(400)}}
    schemas = {f"Big{i}": big for i in range(10)}  # distinct refs: each is built anew

    def body(i: int) -> dict[str, Any]:
        return {
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/Big{i}"}}}
        }

    paths = {
        f"/p{i}": {"post": {"operationId": f"op{i}", "requestBody": body(i)}} for i in range(10)
    }
    doc = {**PETS, "components": {"schemas": schemas}, "paths": paths}
    with pytest.raises(ConfigError, match="whole document"):
        import_openapi(Registry(), doc, client_factory=Wire().factory)
    first = {**doc, "paths": dict(list(paths.items())[:3])}
    assert len(import_openapi(Registry(), first, client_factory=Wire().factory)) == 3


def test_a_document_wide_held_budget_bounds_repeated_refs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_refs, "held_budget", lambda values: 5_000)
    big = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(400)}}
    ref = {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Big"}}}}
    paths = {f"/p{i}": {"post": {"operationId": f"op{i}", "requestBody": ref}} for i in range(10)}
    doc = {**PETS, "components": {"schemas": {"Big": big}}, "paths": paths}
    with pytest.raises(ConfigError, match="would hold more than their budget of 5000"):
        import_openapi(Registry(), doc, client_factory=Wire().factory)
    first = {**doc, "paths": dict(list(paths.items())[:3])}
    assert len(import_openapi(Registry(), first, client_factory=Wire().factory)) == 3


# --- 2: redirects are never followed, whatever the client -----------------------------------


def following_factory(wire: Wire) -> Any:
    def factory(**options: Any) -> httpx.AsyncClient:
        return wire.factory(**options, follow_redirects=True)

    return factory


@pytest.mark.parametrize(
    ("auth", "args"),
    [
        (HeaderAuth("X-Key", "s3cr3t"), {"petId": 1}),
        (QueryAuth("api_key", "s3cr3t"), {"petId": 1}),
        (None, {"petId": 1, "session": "s3cr3t"}),
    ],
)
async def test_a_redirecting_client_does_not_follow_to_another_origin(
    auth: Any, args: dict[str, Any]
):
    wire = Wire(
        lambda r: (
            httpx.Response(302, headers={"location": "https://evil.example/x"})
            if r.url.host == "api.example.com"
            else httpx.Response(200, json={"stolen": True})
        )
    )
    registry = Registry()
    import_openapi(registry, PETS, auth=auth, client_factory=following_factory(wire))
    with pytest.raises(ToolError, match="HTTP 302"):
        await registry.get("getPet").invoke(args)
    assert [r.url.host for r in wire.requests] == ["api.example.com"]


# --- 3: decompression is bounded -----------------------------------------------------------


def bomb(kind: str, megabytes: int = 64) -> bytes:
    wbits = {"gzip": 16 + zlib.MAX_WBITS, "deflate": zlib.MAX_WBITS, "raw": -zlib.MAX_WBITS}[kind]
    packer = zlib.compressobj(9, zlib.DEFLATED, wbits)
    chunk = b"\0" * (1 << 20)
    return b"".join(packer.compress(chunk) for _ in range(megabytes)) + packer.flush()


@pytest.mark.parametrize(
    ("encoding", "kind"),
    [("gzip", "gzip"), ("x-gzip", "gzip"), ("deflate", "deflate"), ("deflate", "raw")],
)
async def test_decompression_bombs_stop_at_the_cap_with_bounded_memory(encoding: str, kind: str):
    payload, headers = bomb(kind), {"content-encoding": encoding}
    # A response built per request, as a real server stream is (a prebuilt one is pre-read).
    registry, _ = pets(wire=Wire(lambda _: raw(200, payload, **headers)))
    tracemalloc.start()
    try:
        with pytest.raises(ToolError, match="larger than"):
            await registry.get("getPet").invoke({"petId": 1})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 * _decode.MAX_RESPONSE_BYTES


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "identity", ""])
async def test_supported_encodings_decode(encoding: str):
    body = b'{"name": "rex"}'
    data = {"gzip": gzip.compress(body), "deflate": zlib.compress(body)}.get(encoding, body)
    headers = {
        "content-type": "application/json",
        **({"content-encoding": encoding} if encoding else {}),
    }
    registry, wire = pets(wire=Wire(raw(200, data, **headers)))
    assert await registry.get("getPet").invoke({"petId": 1}) == {"name": "rex"}
    assert wire.last.headers["accept-encoding"] == "gzip, deflate, identity"


@pytest.mark.parametrize("encoding", ["br", "zstd", "compress", "gzip, gzip"])
async def test_other_content_encodings_are_refused(encoding: str):
    registry, _ = pets(wire=Wire(raw(200, b"xx", content_encoding=encoding)))
    with pytest.raises(ToolError, match="content encoding"):
        await registry.get("getPet").invoke({"petId": 1})


async def test_a_corrupt_compressed_body_is_a_tool_error():
    registry, _ = pets(wire=Wire(raw(200, b"not gzip", content_encoding="gzip")))
    with pytest.raises(ToolError, match="could not be decoded"):
        await registry.get("getPet").invoke({"petId": 1})


# --- 5: credentials in URLs ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["https://user:pw@api.example.com/v1", "https://user@api.example.com"]
)
def test_userinfo_in_the_base_url_is_refused(url: str):
    with pytest.raises(ConfigError, match="auth="):
        import_openapi(Registry(), PETS, base_url=url, client_factory=Wire().factory)


async def test_status_errors_carry_no_query_secret():
    registry, _ = pets(
        wire=Wire(httpx.Response(503)), auth=QueryAuth("api_key", "SECRET-9"), retries=0
    )
    with pytest.raises(ToolFailed) as info:
        await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 1})
    cause = info.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError)
    for text in (repr(cause.request), repr(cause.response.request), str(cause.request.url)):
        assert "SECRET-9" not in text


async def test_transport_errors_carry_no_query_secret():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    registry, _ = pets(wire=Wire(refused), auth=QueryAuth("api_key", "SECRET-8"))
    with pytest.raises(httpx.ConnectError) as info:
        await registry.get("getPet").invoke({"petId": 1})
    assert "SECRET-8" not in repr(info.value.request)


# --- 7: URL hardening ----------------------------------------------------------------------


@pytest.mark.parametrize("value", ["evil.example/x", "a@evil.example", "x?y", "x#y", "a\\b"])
def test_server_variable_values_cannot_change_the_host(value: str):
    doc = {
        **PETS,
        "servers": [{"url": "https://{host}/v1", "variables": {"host": {"default": value}}}],
    }
    with pytest.raises(ConfigError, match="server variable"):
        import_openapi(Registry(), doc, client_factory=Wire().factory)


@pytest.mark.parametrize(
    "host", ["0x7f.0.0.1", "0177.0.0.1", "2130706433", "127.1", "0x7f000001", "127.000.0.1"]
)
def test_ambiguous_numeric_hosts_are_refused(host: str):
    with pytest.raises(ConfigError, match="numeric host"):
        import_openapi(
            Registry(), PETS, base_url=f"https://{host}/v1", client_factory=Wire().factory
        )


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1:8443/v1", "http://[::1]:8080", "HTTPS://API.example.com/v1"]
)
async def test_plain_ip_hosts_and_upper_case_schemes_are_accepted(url: str):
    registry, wire = pets(base_url=url)
    await registry.get("deletePet").invoke({"petId": 1})
    assert wire.last.url.scheme in ("http", "https")


async def test_retry_after_is_honoured_before_a_retry_and_capped(monkeypatch: pytest.MonkeyPatch):
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(http_module, "_sleep", record)
    wire = Wire(
        httpx.Response(429, headers={"retry-after": "2"}),
        httpx.Response(503, headers={"retry-after": "3600"}),
        httpx.Response(429, headers={"retry-after": "Thu, 01 Jan 1970 00:00:00 GMT"}),
        httpx.Response(200, json={"ok": 1}),
    )
    registry, _ = pets(wire=wire, retries=3)
    assert await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 1}) == {"ok": 1}
    assert waits == [2.0, http_module.MAX_RETRY_AFTER, 0.0]


async def test_retry_after_is_not_waited_for_on_the_last_attempt(monkeypatch: pytest.MonkeyPatch):
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(http_module, "_sleep", record)
    registry, _ = pets(wire=Wire(httpx.Response(429, headers={"retry-after": "5"})), retries=1)
    with pytest.raises(ToolFailed):
        await Executor(registry, sleep=no_sleep).run("getPet", {"petId": 1})
    assert waits == [5.0]  # before the one retry, not after the last attempt
    with pytest.raises(httpx.HTTPStatusError):
        await registry.get("getPet").invoke({"petId": 1})  # no executor: no retry, no wait
    assert waits == [5.0]


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(None, None), ("", None), ("soon", None), ("7", 7.0), ("Thu, 01 Jan 1970 00:00:10 GMT", 4.0)],
)
def test_retry_after_parsing(value: str | None, seconds: float | None):
    assert http_module.retry_after(value, now=6.0) == seconds


def test_errors_without_a_request_are_left_alone():
    error = httpx.ConnectError("no request")
    http_module._redact_request(error)
    with pytest.raises(RuntimeError):
        _ = error.request


async def test_a_large_already_read_response_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_decode, "MAX_RESPONSE_BYTES", 10)
    registry, _ = pets(wire=Wire(lambda _: httpx.Response(200, json={"long": "x" * 50})))
    with pytest.raises(ToolError, match="larger than 10 bytes"):
        await registry.get("getPet").invoke({"petId": 1})
