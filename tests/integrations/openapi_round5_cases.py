"""OpenAPI importer cases from review round 5: linear name de-duplication, import budgets.

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import itertools
import logging
import string
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.registry import Registry
from agent_adapter.integrations.openapi import _refs, import_openapi
from agent_adapter.integrations.openapi._schema import build_operations
from tests.integrations.openapi_support import PETS, Wire
from tests.integrations.openapi_work import run_counted

SCHEMA_LOGGER = "agent_adapter.integrations.openapi._schema"
JSON = "application/json"
PUNCTUATION = "-.!@#$%^&*()~"


# Lines of agent_adapter run (deterministic, unlike seconds). Per-stem counters and an
# incrementally kept set name in amortised O(1); re-probing every numbered variant is
# quadratic and runs 2.5 to over 10 times these caps on the documents below.
LINES_PER_OPERATION = 600  # ~250 to build an operation
LINES_PER_TOOL = 1_200  # ~450 to build and import one as a tool
LINES_PER_FIELD = 400  # ~140 to import a field of one body
LINES_PER_SHARED_FIELD = 120  # ~40 a field of a body shared by many operations, most skipped


def clashing_body(size: int, plain: bool = False) -> dict[str, Any]:
    """An object of `size` properties that all sanitise to `a` (`f0`, `f1`, ... if plain)."""
    marks = (
        "".join(p) for n in itertools.count(1) for p in itertools.product(PUNCTUATION, repeat=n)
    )
    names = (f"f{i}" if plain else f"{next(marks)}a" for i in range(size))
    return {"type": "object", "properties": {name: {} for name in names}}


def posting(schema: dict[str, Any], operations: int) -> dict[str, Any]:
    """`operations` POST operations that all send `schema` (shared through one `$ref`)."""
    body = {"content": {JSON: {"schema": {"$ref": "#/components/schemas/Body"}}}}
    paths = {f"/p{n}": {"post": {"requestBody": body, "responses": {}}} for n in range(operations)}
    return {**PETS, "components": {"schemas": {"Body": schema}}, "paths": paths}


# --- 1: names are de-duplicated in amortised O(1); outputs are unchanged ------------------


def get(operation_id: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"responses": {}, **extra, **({"operationId": operation_id} if operation_id else {})}


def param(name: str, where: str) -> dict[str, Any]:
    return {"name": name, "in": where, "required": where == "path", "schema": {"type": "string"}}


CLASH_BODY = {
    "type": "object",
    "properties": {
        k: {"type": "string"} for k in ("id", "i-d", "i.d", "i d", "body_id", "query_id", "x")
    },
}
NAMING = {
    "openapi": "3.0.0",
    "info": {"title": "names", "version": "1"},
    "paths": {
        "/z": {"get": get("dup_3")},
        "/a": {"get": get("dup")},
        "/b": {"get": get("dup")},
        "/c": {"post": get("dup", requestBody={"content": {"multipart/form-data": {}}})},
        "/d": {"get": get("dup")},
        "/e": {"get": get("dup_2")},
        "/f": {"get": get("L" * 70), "put": get("L" * 70), "post": get("L" * 70)},
        "/x-y": {"get": get()},
        "/x_y": {"get": get()},
        "/w/{id}": {
            "post": get(
                "clash",
                parameters=[param("id", where) for where in ("path", "query", "header", "cookie")]
                + [param("query_id", "query")],
                requestBody={"content": {JSON: {"schema": CLASH_BODY}}},
            )
        },
        "/v": {
            "post": get(
                "whole",
                parameters=[param("body", "query")],
                requestBody={"content": {JSON: {"schema": {"type": "array"}}}},
            )
        },
    },
}


@pytest.mark.parametrize(
    ("prefix", "long"),
    [("", "L" * 64), ("api.", "L" * 60)],  # tool names: 64 characters
)
def test_tool_names_are_unchanged_for_clashing_ids(prefix: str, long: str):
    names = [op.name for op in build_operations(NAMING, prefix=prefix, only=None)]
    numbered = [f"{long[:-2]}_2", f"{long[:-2]}_3"]
    expected = ["dup_3", "dup", "dup_2", "dup_4", "dup_2_2", long, *numbered]
    expected += ["get_x_y", "get_x_y_2", "clash", "whole"]
    assert names == [("api_" if prefix else "") + name for name in expected]


def test_property_names_are_unchanged_for_clashing_parameters_and_fields():
    found = {op.key: op for op in build_operations(NAMING, prefix="", only=None)}
    assert [(b.prop, b.wire, b.location) for b in found["clash"].bindings] == [
        ("id", "id", "path"),
        ("query_id", "id", "query"),
        ("header_id", "id", "header"),
        ("cookie_id", "id", "cookie"),
        ("query_query_id", "query_id", "query"),
        ("body_id", "id", "body"),
        ("i_d", "i-d", "body"),
        ("body_i_d", "i.d", "body"),
        ("body_i_d_2", "i d", "body"),
        ("body_body_id", "body_id", "body"),
        ("body_query_id", "query_id", "body"),
        ("x", "x", "body"),
    ]
    assert [b.prop for b in found["whole"].bindings] == ["body", "body_body"]


def test_duplicated_operation_ids_are_imported_with_one_summary_warning(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=SCHEMA_LOGGER):
        found = build_operations(NAMING, prefix="", only=None)
    assert len(found) == 12
    messages = [r.getMessage() for r in caplog.records if r.name == SCHEMA_LOGGER]
    assert messages == [
        "2 OpenAPI operationIds are used by more than one operation (their tools get "
        f"numbered names): dup, {'L' * 70}",
        "1 OpenAPI operations skipped: dup: request body is not JSON (multipart/form-data)",
    ]


def test_ten_thousand_operations_sharing_one_operation_id_build_in_linear_work(tmp_path: Path):
    paths = {f"/p{n}": {"get": get("same")} for n in range(10_000)}
    document = {**PETS, "paths": paths}
    built = run_counted(document, tmp_path, cap=10_000 * LINES_PER_OPERATION, stage="build")
    assert built["names"][:3] == ["same", "same_2", "same_3"]
    assert built["names"][-1] == "same_10000" and len(set(built["names"])) == 10_000
    assert built["warnings"] == [
        "1 OpenAPI operationIds are used by more than one operation (their tools get "
        "numbered names): same"
    ]
    imported = run_counted(document, tmp_path, cap=10_000 * LINES_PER_TOOL)
    assert imported["names"] == built["names"]


def test_long_ids_that_share_their_numbered_names_build_in_linear_work(tmp_path: Path):
    """Distinct 64-character ids that agree on 62 characters all number into one stem."""
    chars = string.ascii_letters + string.digits
    ids = ["P" * 62 + a + b for a, b in itertools.product(chars, repeat=2)][:3000]
    paths = {f"/a{n}": {"get": get(i), "put": get(i)} for n, i in enumerate(ids)}
    cap = 6000 * LINES_PER_OPERATION
    built = run_counted({**PETS, "paths": paths}, tmp_path, cap=cap, stage="build")
    assert len(set(built["names"])) == 6000 and built["names"][:2] == [ids[0], f"{'P' * 62}_2"]


def test_one_body_of_3000_clashing_names_imports_in_linear_work(tmp_path: Path):
    document = posting(clashing_body(3000), 1)
    result = run_counted(document, tmp_path, cap=3000 * LINES_PER_FIELD)
    assert result["names"] == ["post_p0"], result["outcome"]


def test_a_clashing_body_shared_by_40_operations_builds_in_linear_work(tmp_path: Path):
    document = posting(clashing_body(4000), 40)
    cap = 40 * 4000 * LINES_PER_SHARED_FIELD
    run_counted(document, tmp_path, cap=cap, stage="build")
    imported = run_counted(document, tmp_path, cap=cap)
    kept = len(imported["names"])
    assert 0 < kept < 40 and kept * 4000 <= _refs.fields_budget(_refs.check_document(document))
    assert imported["warnings"][-1].startswith(f"{40 - kept} OpenAPI operations skipped")


# --- 2: a document-wide budget on input fields --------------------------------------------


@pytest.mark.parametrize(
    ("values", "fields"), [(1, 5_000), (625, 5_000), (1_000, 8_000), (10**6, 100_000)]
)
def test_the_input_fields_budget_is_relative_to_the_document_size(values: int, fields: int):
    assert _refs.fields_budget(values) == fields


def test_sixty_operations_sharing_a_3000_field_body_stop_at_the_fields_budget(tmp_path: Path):
    document = posting(clashing_body(3000, plain=True), 60)
    result = run_counted(document, tmp_path, cap=60 * 3000 * LINES_PER_SHARED_FIELD)
    budget = _refs.fields_budget(_refs.check_document(document))
    kept = budget // 3000
    assert result["names"] == [f"post_p{n}" for n in range(kept)]
    [warning] = result["warnings"]
    assert warning.startswith(
        f"{60 - kept} OpenAPI operations skipped: post_/p{kept}: 3000 input fields over the budget"
    )
    assert warning.endswith(
        f"(the input-fields budget for all tools is {budget}: 8 per document value, 5000 to 100000)"
    )


def test_operations_after_the_fields_budget_are_skipped_but_smaller_ones_still_fit(
    caplog: pytest.LogCaptureFixture,
):
    document = posting(clashing_body(3000, plain=True), 10)
    document["paths"]["/small"] = {"get": get("small", parameters=[param("q", "query")])}
    budget = _refs.fields_budget(_refs.check_document(document))
    with caplog.at_level(logging.WARNING, logger=SCHEMA_LOGGER):
        found = build_operations(document, prefix="", only=None)
    kept = budget // 3000
    assert [op.name for op in found] == [f"post_p{n}" for n in range(kept)] + ["small"]
    [message] = [r.getMessage() for r in caplog.records if r.name == SCHEMA_LOGGER]
    assert message.startswith(f"{10 - kept} OpenAPI operations skipped: post_/p{kept}: 3000 ")


USER = {
    "type": "object",
    "properties": {
        "login": {"type": "string"},
        "id": {"type": "integer"},
        "avatar_url": {"type": "string", "format": "uri"},
        "site_admin": {"type": "boolean"},
    },
}


def github_like(operations: int) -> dict[str, Any]:
    """A large, realistic API: shared parameters, 40 resource schemas, CRUD per resource."""
    schemas = {
        f"R{r}": {
            "type": "object",
            "required": ["id"],
            "properties": {
                **{
                    f"field_{i}": {"type": "string", "description": f"Field {i}."}
                    for i in range(10 + r % 20)
                },
                "id": {"type": "integer", "readOnly": True},
                "owner": {"$ref": "#/components/schemas/User"},
                "updated_at": {"type": "string", "format": "date-time", "nullable": True},
            },
        }
        for r in range(40)
    }
    shared = ("per_page", "page", "direction", "sort")
    components = {
        "schemas": {**schemas, "User": USER},
        "parameters": {
            name: {"name": name, "in": "query", "schema": {"type": "string"}} for name in shared
        },
    }
    query = [{"$ref": f"#/components/parameters/{name}"} for name in shared]
    paths: dict[str, Any] = {}
    for n in range(operations // 4):
        ref = {"$ref": f"#/components/schemas/R{n % 40}"}
        ok = {"200": {"description": "OK", "content": {JSON: {"schema": ref}}}}
        body = {"required": True, "content": {JSON: {"schema": ref}}}
        item_id = [
            {"name": "item_id", "in": "path", "required": True, "schema": {"type": "integer"}}
        ]
        paths[f"/repos/r{n}/items"] = {
            "get": {"operationId": f"list{n}", "parameters": query, "responses": ok},
            "post": {"operationId": f"create{n}", "requestBody": body, "responses": ok},
        }
        paths[f"/repos/r{n}/items/{{item_id}}"] = {
            "parameters": item_id,
            "patch": {"operationId": f"update{n}", "requestBody": body, "responses": ok},
            "delete": {"operationId": f"delete{n}", "responses": {"204": {"description": ""}}},
        }
    return {**PETS, "components": components, "paths": paths}


@pytest.mark.parametrize("operations", [2000, 3000])
def test_github_like_specs_lose_nothing(operations: int, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger=SCHEMA_LOGGER):
        found = build_operations(github_like(operations), prefix="", only=None)
    assert len(found) == operations
    assert sum(op.output_schema is not None for op in found) == operations * 3 // 4
    assert [r for r in caplog.records if r.name == SCHEMA_LOGGER] == []


# --- 3: one oversized required result skips its operation ----------------------------------


def fan_out_body(depth: int) -> dict[str, Any]:
    """S_i holds S_{i+1} twice: 2**(depth + 1) values once expanded."""
    schemas: dict[str, Any] = {
        f"S{i}": {
            "type": "object",
            "properties": {k: {"$ref": f"#/components/schemas/S{i + 1}"} for k in "ab"},
        }
        for i in range(depth)
    }
    schemas[f"S{depth}"] = {"type": "string"}
    return schemas


def chain(length: int) -> dict[str, Any]:
    """S_i holds S_{i+1}: shallow in the document, deeper than 128 levels once inlined."""
    schemas: dict[str, Any] = {
        f"S{i}": {"type": "object", "properties": {"x": {"$ref": f"#/components/schemas/S{i + 1}"}}}
        for i in range(length)
    }
    schemas[f"S{length}"] = {"type": "string"}
    return schemas


@pytest.mark.parametrize(
    ("schemas", "reason"),
    [
        (
            fan_out_body(14),
            "an OpenAPI schema is too large after inlining $refs (over 10000 values)",
        ),
        (chain(80), "an OpenAPI schema is too deep (over 128 levels)"),
    ],
)
def test_an_oversized_required_body_skips_only_its_operation(
    schemas: dict[str, Any], reason: str, caplog: pytest.LogCaptureFixture
):
    big = {"content": {JSON: {"schema": {"$ref": "#/components/schemas/S0"}}}, "required": True}
    paths = {
        "/a": {"get": get("before")},
        "/crd": {"post": get("createCrd", requestBody=big)},
        "/b": {"get": get("after", parameters=[param("q", "query")])},
    }
    document = {**PETS, "components": {"schemas": schemas}, "paths": paths}
    with caplog.at_level(logging.WARNING, logger=SCHEMA_LOGGER):
        specs = import_openapi(Registry(), document, client_factory=Wire().factory)
    assert [spec.name for spec in specs] == ["before", "after"]
    messages = [r.getMessage() for r in caplog.records if r.name == SCHEMA_LOGGER]
    assert messages == [f"1 OpenAPI operations skipped: createCrd: {reason}"]


def test_an_oversized_required_parameter_skips_only_its_operation():
    big = {"name": "filter", "in": "query", "schema": {"$ref": "#/components/schemas/S0"}}
    paths = {"/a": {"get": get("ok")}, "/b": {"get": get("huge", parameters=[big])}}
    document = {**PETS, "components": {"schemas": fan_out_body(14)}, "paths": paths}
    specs = import_openapi(Registry(), document, client_factory=Wire().factory)
    assert [spec.name for spec in specs] == ["ok"]
