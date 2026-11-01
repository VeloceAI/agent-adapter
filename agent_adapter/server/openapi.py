"""The OpenAPI document (3.1, or 3.0.3 without unions): one `invoke_<tool>` per tool."""

from __future__ import annotations

import copy
from typing import Any, Literal, get_args

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.cost.summary import LedgerSummary
from agent_adapter.server._openapi30 import downgrade
from agent_adapter.server.problems import PROBLEM_MEDIA_TYPE

Dialect = Literal["3.1", "3.0", "3.0-gateway"]

TARGET = "http"
_REF = "#/components/schemas/"
_JSON = "application/json"
_STRING: dict[str, Any] = {"type": "string"}
_ERRORS = {
    "invoke": ("400", "401", "403", "404", "409", "422", "429", "502", "504"),
    "decide": ("400", "401", "403", "404", "409", "422", "502", "504"),
    "read": ("401", "403", "404"),
}


def openapi_document(
    registry: Registry,
    *,
    title: str,
    version: str,
    server_url: str | None = None,
    dialect: Dialect = "3.1",
) -> dict[str, Any]:
    """OpenAPI for the tools of `registry` exported to `http` and every server route.

    Args:
        dialect: `"3.1"`; `"3.0"` is OpenAPI 3.0.3 with no `oneOf`/`anyOf`/`allOf` (Go and
            Rust generators); `"3.0-gateway"` is that, with only the invoke operations and no
            auth blocks, for gateways that import tools (AgentCore); it needs `server_url`.

    Raises:
        ConfigError: an unknown `dialect`, or `"3.0-gateway"` without `server_url`.
    """
    if dialect not in get_args(Dialect):
        raise ConfigError(f"dialect must be one of {get_args(Dialect)}, got {dialect!r}")
    if dialect == "3.0-gateway" and not server_url:
        raise ConfigError("dialect='3.0-gateway' needs server_url (gateways call that URL)")
    schemas = _fixed_schemas()
    paths: dict[str, Any] = _fixed_paths()
    for spec in registry.for_target(TARGET):
        paths[f"/tools/{spec.name}/invoke"] = {"post": _invoke_operation(spec, schemas)}
    document: dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {"title": title, "version": version},
        "paths": paths,
        "components": {
            "schemas": schemas,
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
            "responses": {
                "Problem": {
                    "description": "An RFC 9457 problem",
                    "content": {PROBLEM_MEDIA_TYPE: {"schema": {"$ref": _REF + "Problem"}}},
                }
            },
        },
        "security": [{"bearerAuth": []}],
    }
    if server_url:
        document["servers"] = [{"url": server_url}]
    if dialect == "3.1":
        return document
    return downgrade(document, gateway=dialect == "3.0-gateway")


def _invoke_operation(spec: ToolSpec, schemas: dict[str, Any]) -> dict[str, Any]:
    arguments = _relocated(spec.json_schema(), f"{spec.name}__", schemas)
    result_schema = spec.result_schema()
    result = _relocated(result_schema, f"{spec.name}__result__", schemas) if result_schema else {}
    body = {
        "type": "object",
        "properties": {"arguments": arguments, "run_id": _STRING, "thread_id": _STRING},
        "required": ["arguments"],
        "additionalProperties": False,
    }
    ok = copy.deepcopy(schemas["InvokeOk"])
    ok["properties"]["result"] = result
    headers = [
        _header("Idempotency-Key", "Derives the call id; a repeat replays the first response"),
        _header("X-Run-Id", "The run to invoke into (a body run_id wins)"),
        _header("X-Thread-Id", "The thread (a body thread_id wins)"),
    ]
    return {
        "operationId": f"invoke_{spec.name}",
        "summary": spec.description.splitlines()[0],
        "description": spec.description,
        "tags": sorted(spec.tags),
        "parameters": headers,
        "requestBody": {"required": True, "content": {_JSON: {"schema": body}}},
        "responses": {
            "200": _json("The tool ran", ok),
            "202": _json("Waiting for approval", {"$ref": _REF + "InvokePending"}),
            **_problems("invoke"),
        },
    }


def _relocated(schema: dict[str, Any], prefix: str, schemas: dict[str, Any]) -> dict[str, Any]:
    """`schema` with its `$defs` moved into components and its refs pointed there."""
    defs = schema.pop("$defs", {})
    for name, definition in defs.items():
        schemas[prefix + name] = _rewrite(definition, prefix)
    rewritten: dict[str, Any] = _rewrite(schema, prefix)
    return rewritten


def _rewrite(node: Any, prefix: str) -> Any:
    if isinstance(node, dict):
        out = {key: _rewrite(value, prefix) for key, value in node.items()}
        ref = out.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            out["$ref"] = _REF + prefix + ref.removeprefix("#/$defs/")
        found = out.get("discriminator")
        if isinstance(found, dict) and isinstance(found.get("mapping"), dict):
            # pydantic points tagged-union variants into $defs too
            out["discriminator"] = {**found, "mapping": _rewrite_map(found["mapping"], prefix)}
        return out
    if isinstance(node, list):
        return [_rewrite(item, prefix) for item in node]
    return node


def _rewrite_map(mapping: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        tag: _REF + prefix + ref.removeprefix("#/$defs/")
        if isinstance(ref, str) and ref.startswith("#/$defs/")
        else ref
        for tag, ref in mapping.items()
    }


def _fixed_paths() -> dict[str, Any]:
    approval_id = _path_param("approval_id")
    run_id = _path_param("run_id")
    events = {
        "description": "Server-sent events",
        "content": {"text/event-stream": {"schema": _STRING}},
    }
    decided = {"oneOf": [{"$ref": _REF + "InvokeOk"}, {"$ref": _REF + "Denied"}]}
    decide_body = {"required": True, "content": {_JSON: {"schema": {"$ref": _REF + "DecideBody"}}}}
    return {
        "/healthz": {
            "get": {
                "operationId": "healthz",
                "security": [],
                "responses": {"200": _json("Alive", {"$ref": _REF + "Health"})},
            }
        },
        "/readyz": {
            "get": {
                "operationId": "readyz",
                "description": "503 while the app starts or drains (create_app only)",
                "security": [],
                "responses": {
                    "200": _json("Ready", {"$ref": _REF + "Ready"}),
                    "503": {"$ref": "#/components/responses/Problem"},
                },
            }
        },
        "/tools": _read("listTools", "Catalogue", [_query(q) for q in ("shape", "tags", "kinds")]),
        "/tools/{name}": _read("getTool", "ToolDescriptor", [_path_param("name"), _query("shape")]),
        "/approvals": _read("listApprovals", "ApprovalList", [_query("run_id"), _query("status")]),
        "/approvals/{approval_id}": _read("getApproval", "Approval", [approval_id]),
        "/approvals/{approval_id}/decide": {
            "post": {
                "operationId": "decideApproval",
                "parameters": [approval_id],
                "requestBody": decide_body,
                "responses": {
                    "200": _json("Approved and executed, or denied", decided),
                    "202": _json(
                        "Approved, not run: parked elsewhere (MCP or a nested call)",
                        {"$ref": _REF + "ApprovedElsewhere"},
                    ),
                    **_problems("decide"),
                },
            }
        },
        "/runs/{run_id}/events": {
            "get": {
                "operationId": "runEvents",
                "parameters": [
                    run_id,
                    _query("close_on"),
                    _query("after"),
                    _header("Last-Event-ID", "Replay the events after this id"),
                ],
                "responses": {"200": events, **_problems("read")},
            }
        },
        "/runs/{run_id}/cost": _read("runCost", "LedgerSummary", [run_id]),
        "/openapi.json": _document("openapiDocument", "This document (OpenAPI 3.1)"),
        "/openapi-3.0.json": _document("openapiDocument30", "This document as OpenAPI 3.0.3"),
    }


def _document(operation_id: str, description: str) -> dict[str, Any]:
    return {
        "get": {
            "operationId": operation_id,
            "responses": {
                "200": _json(description, {"type": "object"}),
                **_problems("read"),
            },
        }
    }


def _read(operation_id: str, schema: str, parameters: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "get": {
            "operationId": operation_id,
            "parameters": parameters,
            "responses": {"200": _json("OK", {"$ref": _REF + schema}), **_problems("read")},
        }
    }


def _fixed_schemas() -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    schemas["LedgerSummary"] = _relocated(
        LedgerSummary.model_json_schema(), "LedgerSummary__", schemas
    )
    decision = _object(
        approved={"type": "boolean"},
        edited_args={"type": ["object", "null"]},
        reason=_STRING,
        decided_by=_STRING,
        decided_at={"type": "number"},
    )
    ids = {"run_id": _STRING, "call_id": _STRING}
    schemas.update(
        Problem={
            **_object(type=_STRING, title=_STRING, status={"type": "integer"}, detail=_STRING),
            "required": ["type", "title", "status"],
            "additionalProperties": True,
        },
        Health=_object(status=_STRING, version=_STRING),
        Ready=_object(status={"const": "ready"}, version=_STRING),
        InvokeOk=_object(
            status={"const": "ok"},
            **ids,
            result={},
            cached={"type": "boolean"},
            attempts={"type": "integer"},
        ),
        InvokePending=_object(status={"const": "pending_approval"}, **ids, approval_id=_STRING),
        Denied=_object(status={"const": "denied"}, **ids, approval_id=_STRING, reason=_STRING),
        ApprovedElsewhere=_object(
            status={"const": "approved"}, **ids, approval_id=_STRING, note=_STRING
        ),
        DecideBody={
            **_object(
                approved={"type": "boolean"}, edited_arguments={"type": "object"}, reason=_STRING
            ),
            "required": ["approved"],
        },
        Approval=_object(
            approval_id=_STRING,
            status={"enum": ["pending", "approved", "denied"]},
            tool=_STRING,
            arguments={"type": "object"},
            **ids,
            hint=_STRING,
            created_at={"type": "number"},
            decision={"oneOf": [decision, {"type": "null"}]},
        ),
        ApprovalList=_object(approvals={"type": "array", "items": {"$ref": _REF + "Approval"}}),
        ToolDescriptor=_object(
            name=_STRING,
            description=_STRING,
            input_schema={"type": "object"},
            output_schema={"type": ["object", "null"]},
            kind=_STRING,
            tags={"type": "array", "items": _STRING},
            approval={"enum": ["never", "always", "conditional"]},
            requires_approval={"type": "boolean"},
            parallel_safe={"type": "boolean"},
            provider={"type": "object"},
        ),
        Catalogue=_object(
            version={"const": "1"},
            tools={"type": "array", "items": {"$ref": _REF + "ToolDescriptor"}},
        ),
    )
    return schemas


def _object(**properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties}


def _json(description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"description": description, "content": {_JSON: {"schema": schema}}}


def _problems(group: str) -> dict[str, Any]:
    return {status: {"$ref": "#/components/responses/Problem"} for status in _ERRORS[group]}


def _path_param(name: str) -> dict[str, Any]:
    return {"name": name, "in": "path", "required": True, "schema": _STRING}


def _query(name: str) -> dict[str, Any]:
    return {"name": name, "in": "query", "required": False, "schema": _STRING}


def _header(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "in": "header",
        "required": False,
        "description": description,
        "schema": _STRING,
    }
