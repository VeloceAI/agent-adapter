"""OpenAPI importer cases from review round 4: shared response refs, gzip NUL padding.

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_adapter.core.errors import ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.openapi import _refs, import_openapi
from tests.integrations.openapi_round3_cases import gzip_member
from tests.integrations.openapi_support import PETS, Wire, pets, raw
from tests.integrations.openapi_work import run_counted

SCHEMA_LOGGER = "agent_adapter.integrations.openapi._schema"
JSON = "application/json"
METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


def listing_spec(operations: int, fields: int = 100) -> dict[str, Any]:
    """`operations` list endpoints that all return one shared, realistic page schema."""
    props = {f"f{i}": {"type": "string", "description": f"F {i}."} for i in range(fields)}
    page = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/components/schemas/Item"}},
            "next": {"type": "string", "nullable": True},
        },
    }
    page_ref = {"$ref": "#/components/schemas/Page"}
    components = {
        "schemas": {"Item": {"type": "object", "properties": props}, "Page": page},
        "parameters": {"Limit": {"name": "limit", "in": "query", "schema": {"type": "integer"}}},
        "responses": {"Page": {"description": "A page.", "content": {JSON: {"schema": page_ref}}}},
    }
    paths = {
        f"/r{n}": {
            "get": {
                "operationId": f"list{n}",
                "summary": "List.",
                "parameters": [{"$ref": "#/components/parameters/Limit"}],
                "responses": {"200": {"$ref": "#/components/responses/Page"}},
            }
        }
        for n in range(operations)
    }
    return {**PETS, "components": components, "paths": paths}


def shared_big_schema_spec(where: str, paths: int = 300, size: int = 4000) -> dict[str, Any]:
    """Many operations (8 per shared path item) all sending or returning one big schema."""
    big = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(size // 2)}}
    ref = {"$ref": "#/components/schemas/Big"}
    op: dict[str, Any]
    if where == "out":
        op = {"responses": {"200": {"description": "", "content": {JSON: {"schema": ref}}}}}
    else:
        op = {"parameters": [{"name": "q", "in": "query", "schema": ref}], "responses": {}}
    return {
        **PETS,
        "components": {"schemas": {"Big": big}, "pathItems": {"X": dict.fromkeys(METHODS, op)}},
        "paths": {f"/p{n}": {"$ref": "#/components/pathItems/X"} for n in range(paths)},
    }


def fan_out(depth: int) -> dict[str, Any]:
    """T_i refers to T_{i+1} twice: about 2**(depth + 1) values once expanded."""
    schemas: dict[str, Any] = {
        f"T{i}": {
            "type": "object",
            "properties": {
                "a": {"$ref": f"#/components/schemas/T{i + 1}"},
                "b": {"$ref": f"#/components/schemas/T{i + 1}"},
            },
        }
        for i in range(depth)
    }
    schemas[f"T{depth}"] = {"type": "string"}
    return schemas


# --- 1: shared response refs are built once; drops are summarised --------------------------


def test_a_large_spec_with_shared_response_refs_keeps_every_output_schema(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=SCHEMA_LOGGER):
        specs = import_openapi(Registry(), listing_spec(2000), client_factory=Wire().factory)
    assert len(specs) == 2000
    schemas = [spec.output_schema for spec in specs]
    assert all(schema is not None for schema in schemas)
    last = schemas[-1]
    assert last is not None and last["properties"]["next"]["type"] == ["string", "null"]
    assert [r for r in caplog.records if r.name == SCHEMA_LOGGER] == []


def test_dropped_response_schemas_are_summarised_in_one_warning(
    caplog: pytest.LogCaptureFixture,
):
    response = {
        "description": "",
        "content": {JSON: {"schema": {"$ref": "#/components/schemas/T0"}}},
    }
    paths = {
        f"/d{n}": {"get": {"operationId": f"deep{n}", "responses": {"200": response}}}
        for n in range(7)
    }
    doc = {**PETS, "components": {"schemas": fan_out(13)}, "paths": paths}
    with caplog.at_level(logging.WARNING, logger=SCHEMA_LOGGER):
        specs = import_openapi(Registry(), doc, client_factory=Wire().factory)
    assert len(specs) == 7 and all(spec.output_schema is None for spec in specs)
    messages = [r.getMessage() for r in caplog.records if r.name == SCHEMA_LOGGER]
    assert messages == [
        "7 OpenAPI response schemas dropped (too large): deep0, deep1, deep2, deep3, deep4, ..."
    ]


# Lines of agent_adapter run (deterministic, unlike seconds): ~0.4 M to import 400 tools
# returning one big schema, ~0.25 M to refuse 2400 sending it. Copying the schema into every
# tool runs 1.8 M and 3.6 M.
SHARED_BIG_LINES = 1_000_000


def test_many_tools_returning_one_big_shared_schema_hold_a_bounded_total(tmp_path: Path):
    doc = shared_big_schema_spec("out", paths=50)  # 400 tools; each copies its schemas
    result = run_counted(doc, tmp_path, cap=SHARED_BIG_LINES)
    assert len(result["names"]) == 400, result["outcome"]
    kept = sum(result["outputs"])
    assert kept > 0 and kept * 4001 <= _refs.held_budget(_refs.check_document(doc))
    assert result["warnings"][-1].startswith(f"{400 - kept} OpenAPI response schemas")


def test_many_tools_sending_one_big_shared_schema_fail_in_bounded_work(tmp_path: Path):
    result = run_counted(shared_big_schema_spec("in"), tmp_path, cap=SHARED_BIG_LINES)
    assert result["outcome"].startswith("ConfigError"), result["outcome"]
    assert "would hold more than their budget" in result["outcome"], result["outcome"]


# --- 4: NUL padding after the last gzip member --------------------------------------------


class Chunks(httpx.AsyncByteStream):
    """A body delivered in the given chunks."""

    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:  # type: ignore[override]
        for chunk in self.chunks:
            yield chunk


def chunked(*chunks: bytes) -> httpx.Response:
    headers = {"content-encoding": "gzip", "content-type": JSON}
    return httpx.Response(200, headers=headers, stream=Chunks(*chunks))


@pytest.mark.parametrize(
    "reply",
    [
        lambda: chunked(gzip_member(b'{"a": 1}') + b"\0" * 512),
        lambda: chunked(gzip_member(b'{"a": ') + gzip_member(b"1}") + b"\0"),
        lambda: chunked(gzip_member(b'{"a": 1}'), b"\0" * 3, b"\0" * 5),
        lambda: chunked(gzip_member(b'{"a": 1}') + b"\0", b"\0"),
    ],
)
async def test_nul_padding_after_the_last_gzip_member_is_ignored(reply: Any):
    registry, _ = pets(wire=Wire(reply()))
    assert await registry.get("getPet").invoke({"petId": 1}) == {"a": 1}


@pytest.mark.parametrize(
    "reply",
    [
        lambda: raw(200, gzip_member(b"1") + b"\0\0x", content_encoding="gzip", content_type=JSON),
        lambda: raw(
            200,
            gzip_member(b"1") + b"\0" + gzip_member(b"2"),
            content_encoding="gzip",
            content_type=JSON,
        ),
        lambda: chunked(gzip_member(b"1"), b"\0\0", b"\0x"),
        lambda: raw(200, zlib.compress(b"1") + b"\0", content_encoding="deflate"),
    ],
)
async def test_other_data_after_the_padding_is_still_a_tool_error(reply: Any):
    registry, _ = pets(wire=Wire(reply()))
    with pytest.raises(ToolError, match="could not be decoded"):
        await registry.get("getPet").invoke({"petId": 1})
