"""OpenAPI importer cases from the MB merge: lazy models, requested skips, fields first.

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from agent_adapter.core import imported
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.openapi import _refs, import_openapi
from agent_adapter.integrations.openapi._schema import build_operations
from tests.integrations.openapi_support import PETS

JSON = "application/json"


def many_gets(count: int) -> dict[str, Any]:
    paths = {
        f"/items{n}": {
            "get": {
                "operationId": f"list{n}",
                "parameters": [
                    {"name": "page-size", "in": "query", "schema": {"type": "integer"}},
                    {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {},
            }
        }
        for n in range(count)
    }
    return {**copy.deepcopy(PETS), "paths": paths}


def test_a_60_tool_import_builds_no_argument_model_until_used(monkeypatch):
    built: list[str] = []
    real = imported.model_from_json_schema

    def counting(name: str, schema: Any) -> Any:
        built.append(name)
        return real(name, schema)

    monkeypatch.setattr(imported, "model_from_json_schema", counting)
    specs = import_openapi(Registry(), many_gets(60), client_factory=lambda **k: None)
    assert len(specs) == 60 and built == []
    first = specs[0]
    assert first.json_schema()["required"] == ["q"]
    validated = first.validate_args({"q": "x", "page_size": 3})
    assert built == ["List0Args"] and validated.model_dump() == {"page_size": 3, "q": "x"}
    eager = real("List0Args", first.input_schema)
    assert first.args_model.model_json_schema() == eager.model_json_schema()
    assert built == ["List0Args"]  # cached


def shared_body(fields: int, operations: int, *, response: bool) -> dict[str, Any]:
    """`operations` POSTs sending (and answering with) one `fields`-property body by `$ref`."""
    ref = {"$ref": "#/components/schemas/Body"}
    paths = {}
    for n in range(operations):
        op: dict[str, Any] = {"requestBody": {"content": {JSON: {"schema": ref}}}, "responses": {}}
        if response:
            op["responses"] = {"200": {"description": "", "content": {JSON: {"schema": ref}}}}
        paths[f"/p{n}"] = {"post": op}
    body = {"type": "object", "properties": {f"f{i}": {} for i in range(fields)}}
    return {**copy.deepcopy(PETS), "components": {"schemas": {"Body": body}}, "paths": paths}


@pytest.mark.parametrize("response", [False, True])
def test_skips_for_the_fields_budget_hold_nothing_so_never_refuse_the_document(response: bool):
    document = shared_body(9000, 200, response=response)
    budget = _refs.fields_budget(_refs.check_document(document))
    found = build_operations(document, prefix="", only=None)
    assert 0 < len(found) == budget // 9000 < 200
    assert all(len(op.input_schema["properties"]) == 9000 for op in found)
    assert all((op.output_schema is not None) == response for op in found)


def test_a_requested_operation_that_a_bound_skips_is_a_config_error(caplog):
    document = shared_body(9000, 30, response=False)
    kept = _refs.fields_budget(_refs.check_document(document)) // 9000
    wanted = [f"post_/p{n}" for n in range(30)]
    with pytest.raises(ConfigError, match="requested OpenAPI operations were skipped: ") as info:
        build_operations(document, prefix="", only=wanted)
    named = str(info.value).split(": ", 1)[1].split(", ")
    assert sorted(named) == sorted(wanted[kept:])
    assert "OpenAPI operations skipped" in caplog.text  # the summary is logged first
    found = build_operations(document, prefix="", only=wanted[:kept])
    assert [op.key for op in found] == wanted[:kept]  # what fits imports


def test_a_requested_operation_skipped_for_its_body_type_is_a_config_error():
    document = copy.deepcopy(PETS)
    form = {"requestBody": {"content": {"multipart/form-data": {}}}, "responses": {}}
    document["paths"] = {"/upload": {"post": {"operationId": "upload", **form}}}
    with pytest.raises(ConfigError, match="were skipped: upload"):
        import_openapi(Registry(), document, operations=["upload"], client_factory=lambda **k: None)
    assert import_openapi(Registry(), document, client_factory=lambda **k: None) == []
