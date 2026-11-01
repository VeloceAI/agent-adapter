"""OpenAPI importer cases: documents, `$ref`s, server URLs and the flat input schema.

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.integrations.openapi._document import load_document, server_url
from agent_adapter.integrations.openapi._refs import resolve
from agent_adapter.integrations.openapi._schema import build_operations
from tests.integrations.openapi_support import PETS, pets


def ops(document: dict[str, Any], **options: Any) -> dict[str, Any]:
    return {op.name: op for op in build_operations(document, prefix="", only=None, **options)}


def doc(paths: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"openapi": "3.1.0", "info": {}, "paths": paths, **extra}


ID = {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}


def get(op: dict[str, Any]) -> dict[str, Any]:
    return {"get": {"responses": {}, **op}}


# --- loading -----------------------------------------------------------------------------------


def test_a_mapping_is_copied_not_mutated():
    original = copy.deepcopy(PETS)
    loaded = load_document(original)
    loaded["paths"].clear()
    assert original == PETS


def test_json_and_yaml_text_load():
    assert load_document(json.dumps(PETS)) == PETS
    assert load_document(yaml.safe_dump(PETS)) == PETS


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml", ".txt"])
def test_files_load_by_path_or_string(tmp_path: Path, suffix: str):
    path = tmp_path / f"pets{suffix}"
    text = json.dumps(PETS) if suffix == ".json" else yaml.safe_dump(PETS)
    path.write_text(text, encoding="utf-8")
    assert load_document(path) == PETS
    assert load_document(str(path)) == PETS


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("https://example.com/openapi.json", "does not fetch URLs"),
        ("no/such/file.yaml", "cannot read"),
        ("{not json", "not valid JSON"),
        ("a: [unclosed\nb: 1", "not valid YAML"),
        ("- just\n- a list\n", "mapping at the root"),
        (json.dumps({"swagger": "2.0", "paths": {}}), "version: 2.0"),
        (json.dumps({"info": {}}), "version: none"),
        (json.dumps({"openapi": "3.0.0", "paths": []}), "'paths' must be a mapping"),
        (42, "mapping, text or a path"),
    ],
)
def test_bad_documents_are_config_errors(source: Any, match: str):
    with pytest.raises(ConfigError, match=match):
        load_document(source)


def test_yaml_needs_pyyaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(MissingDependency, match="pyyaml"):
        load_document("openapi: 3.0.0\npaths: {}\n")


# --- $ref --------------------------------------------------------------------------------------


def test_refs_resolve_with_escapes_list_indexes_and_siblings():
    document = {
        "components": {"schemas": {"a/b": {"type": "string"}, "pair": [{"type": "integer"}]}},
        "x": {"$ref": "#/components/schemas/a~1b", "description": "kept"},
        "y": {"$ref": "#/components/schemas/pair/0"},
    }
    assert resolve(document["x"], document) == {"type": "string", "description": "kept"}
    assert resolve(document["y"], document) == {"type": "integer"}


def test_a_recursive_ref_becomes_an_open_schema(caplog: pytest.LogCaptureFixture):
    node = {"type": "object", "properties": {"child": {"$ref": "#/components/schemas/Node"}}}
    document = {"components": {"schemas": {"Node": node}}}
    with caplog.at_level(logging.WARNING):
        resolved = resolve({"$ref": "#/components/schemas/Node"}, document)
    assert resolved["properties"]["child"] == {}  # cut where Node recurs
    assert "recursive $ref" in caplog.text


@pytest.mark.parametrize(
    ("ref", "match"),
    [
        ("other.yaml#/Pet", "only local"),
        ("#/components/schemas/Nope", "unresolvable"),
        ("#/a/9", "unresolvable"),
    ],
)
def test_bad_refs_are_config_errors(ref: str, match: str):
    with pytest.raises(ConfigError, match=match):
        resolve({"$ref": ref}, {"components": {"schemas": {}}, "a": [1]})


# --- server URL --------------------------------------------------------------------------------


def test_server_url_fills_variables_and_base_url_wins():
    assert server_url(PETS, None) == "https://api.example.com/v1"
    assert server_url(PETS, "http://localhost:8080") == "http://localhost:8080"


@pytest.mark.parametrize(
    ("servers", "match"),
    [
        ([{"url": "/v1"}], "no absolute server URL"),
        ([], "no absolute server URL"),
        ([{"url": "https://{h}/v1", "variables": {"h": {}}}], "has no default"),
        (["not-a-mapping"], "no absolute server URL"),
    ],
)
def test_server_url_problems_ask_for_base_url(servers: Any, match: str):
    with pytest.raises(ConfigError, match=match):
        server_url({"servers": servers}, None)


# --- names, descriptions, filters --------------------------------------------------------------


def test_names_come_from_operation_ids_or_method_and_path():
    document = doc(
        {
            "/a/{id}": get({"operationId": "1st.op", "parameters": [ID]}),
            "/b/{id}/c": get({"parameters": [ID]}),
            "/c": {"get": {"operationId": "dup"}, "post": {"operationId": "dup"}},
            "/d": get({"operationId": "x" * 80}),
        }
    )
    assert list(ops(document)) == ["op_1st_op", "get_b_id_c", "dup", "dup_2", "x" * 64]
    named = build_operations(document, prefix="svc.", only=["get_/b/{id}/c"])
    assert [op.name for op in named] == ["svc_get_b_id_c"]


def test_descriptions_fall_back_to_summary_then_method_and_path():
    found = ops(doc({"/a": get({"summary": "Sum."}), "/b": get({})}))
    assert found["get_a"].description == "Sum." and found["get_b"].description == "GET /b"
    assert ops(PETS)["getPet"].description == "Fetch one pet by id."


def test_operations_filter_rejects_unknown_ids_and_a_single_string():
    with pytest.raises(ConfigError, match="unknown OpenAPI operations: nope"):
        build_operations(PETS, prefix="", only=["getPet", "nope"])
    with pytest.raises(ConfigError, match="iterable"):
        build_operations(PETS, prefix="", only="getPet")


def test_non_json_bodies_skip_the_operation_with_a_warning(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        found = ops(PETS)
    assert "uploadPhoto" not in found
    assert "uploadPhoto: request body is not JSON (multipart/form-data)" in caplog.text


# --- the input schema --------------------------------------------------------------------------


def test_parameters_are_sanitised_merged_and_described():
    schema = ops(PETS)["getPet"].input_schema
    assert list(schema["properties"]) == [
        "petId",
        "verbose",
        "include",
        "fields",
        "X_Trace",
        "session",
    ]
    assert schema["required"] == ["petId"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["X_Trace"] == {"type": "string", "description": "Trace id."}


def test_operation_parameters_override_path_item_parameters_and_clashes_get_a_prefix():
    found = ops(
        doc(
            {
                "/a/{id}": {
                    "parameters": [{"name": "id", "in": "path", "schema": {"type": "string"}}],
                    "get": {
                        "parameters": [
                            {"name": "id", "in": "path", "schema": {"type": "integer"}},
                            {"name": "id", "in": "query", "schema": {"type": "string"}},
                            {
                                "name": "page[size]",
                                "in": "query",
                                "content": {"application/json": {"schema": {"type": "integer"}}},
                            },
                            {"name": "where", "in": "matrix"},
                        ],
                    },
                }
            }
        )
    )
    properties = found["get_a_id"].input_schema["properties"]
    assert properties == {
        "id": {"type": "integer"},
        "query_id": {"type": "string"},
        "page_size": {"type": "integer"},
    }


def test_json_bodies_flatten_drop_read_only_and_keep_optionality():
    schema = ops(PETS)["createPet"].input_schema
    assert schema["required"] == ["body_name"]
    assert schema["properties"]["nickname"] == {"type": ["string", "null"]}
    assert "id" not in schema["properties"]
    optional = ops(
        doc(
            {
                "/p": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["a"],
                                        "properties": {"a": {"type": "string"}},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
    )
    assert "required" not in optional["post_p"].input_schema


@pytest.mark.parametrize(
    "body_schema",
    [
        {"type": "array", "items": {"type": "integer"}},
        {"allOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]},
        {"type": "object", "additionalProperties": {"type": "string"}},
    ],
)
def test_non_flat_bodies_are_one_body_field_named_around_clashes(body_schema: dict[str, Any]):
    found = ops(
        doc(
            {
                "/p": {
                    "post": {
                        "parameters": [
                            {"name": "body", "in": "query", "schema": {"type": "string"}}
                        ],
                        "requestBody": {
                            "required": True,
                            "content": {"application/json": {"schema": body_schema}},
                        },
                    }
                }
            }
        )
    )
    schema = found["post_p"].input_schema
    assert schema["properties"]["body_body"] == body_schema and schema["required"] == ["body_body"]


def test_nullable_without_a_type_becomes_any_of_null():
    found = ops(
        doc(
            {
                "/a": get(
                    {
                        "parameters": [
                            {
                                "name": "q",
                                "in": "query",
                                "schema": {"nullable": True, "enum": ["x"]},
                            }
                        ]
                    }
                )
            }
        )
    )
    assert found["get_a"].input_schema["properties"]["q"] == {
        "anyOf": [{"enum": ["x"]}, {"type": "null"}]
    }


def test_the_output_schema_is_the_lowest_2xx_json_response():
    assert ops(PETS)["getPet"].output_schema == {
        **{k: v for k, v in PETS["components"]["schemas"]["Pet"].items() if k != "properties"},
        "properties": {
            "id": {"type": "integer", "readOnly": True},
            "name": {"type": "string", "description": "Pet name."},
            "age": {"type": "integer"},
            "nickname": {"type": ["string", "null"]},
        },
    }
    assert ops(PETS)["deletePet"].output_schema is None
    found = ops(doc({"/a": get({"responses": {"default": {"description": "x"}}})}))
    assert found["get_a"].output_schema is None


def test_imported_specs_keep_raw_schemas_and_validate_loosely():
    registry, _ = pets()
    spec = registry.get("getPet")
    assert spec.input_schema == ops(PETS)["getPet"].input_schema
    assert spec.result_schema()["required"] == ["name"]
    assert spec.portable is False  # booleans in a query array, untyped arrays
