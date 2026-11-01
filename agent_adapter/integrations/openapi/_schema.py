"""One flat tool input schema per OpenAPI operation, and how each property is sent."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.openapi._names import sanitise, tool_name, unique
from agent_adapter.integrations.openapi._refs import (
    Resolver,
    SchemaTooLarge,
    check_document,
    fields_budget,
    pointer,
)
from agent_adapter.integrations.openapi._report import ImportReport

logger = logging.getLogger(__name__)

METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
Location = Literal["path", "query", "header", "cookie", "body", "body_whole"]
_PARAM_LOCATIONS = ("path", "query", "header", "cookie")
_IGNORED_HEADERS = frozenset({"accept", "content-type", "authorization"})  # OpenAPI §4.8.12
_JSON_MEDIA = re.compile(r"^application/(?:[\w.+-]+\+)?json(?:\s*;.*)?$", re.IGNORECASE)
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")  # RFC 9110 token: header and cookie names


@dataclass(frozen=True)
class Binding:
    """Where one tool property goes on the wire."""

    prop: str  # property name the model sees
    wire: str  # parameter or body field name ("" for the whole body)
    location: Location
    explode: bool = True  # arrays in query: repeated keys (True) or comma-joined (False)


@dataclass(frozen=True)
class Operation:
    """Everything needed to build and call one tool."""

    key: str  # operationId, else `{method}_{path}` (what `operations=` matches)
    name: str
    method: str
    path: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    bindings: tuple[Binding, ...]
    body: Literal["none", "optional", "required"]
    tags: tuple[str, ...]


def build_operations(
    document: Mapping[str, Any], *, prefix: str, only: Iterable[str] | None, values: int = 0
) -> list[Operation]:
    """Operations in document order, filtered by `only` (operationIds or generated keys).

    An operation over a per-operation bound is skipped (one summary WARNING per import).

    Raises:
        ConfigError: unknown `only` names, an unsafe path, a malformed operation or `$ref`,
            or a document-wide budget exceeded.
    """
    wanted = None if only is None else _wanted(only)
    size = values or check_document(document)  # the document size the budgets scale with
    resolver = Resolver(document, values=size)
    report = ImportReport(fields_budget(size))
    found: list[Operation] = []
    keys: set[str] = set()
    names: set[str] = set()
    counters: dict[str, int] = {}
    for method, path, item, op in _raw_operations(document, resolver):
        operation_id = op.get("operationId")
        key = str(operation_id) if operation_id else f"{method}_{path}"
        keys.add(key)
        if wanted is not None and key not in wanted:
            continue
        if operation_id:
            report.ids[key] += 1
        base = key if operation_id else f"{method}_{re.sub(r'[^0-9A-Za-z]+', '_', path).strip('_')}"
        name = unique(tool_name(prefix, base), names, counters)
        build = partial(_operation, resolver, method, path, item, op, key, name, report)
        operation = _checked_operation(key, report, build)
        if operation is not None:
            names.add(name)
            found.append(operation)
    report.log(logger)
    missing = sorted((wanted or set()) - keys)
    if missing:
        raise ConfigError(f"unknown OpenAPI operations: {', '.join(missing)}")
    if wanted is not None and (skipped := report.skipped_keys() & wanted):
        raise ConfigError(
            f"requested OpenAPI operations were skipped: {', '.join(sorted(skipped))}"
        )
    return found


def _wanted(only: Iterable[str]) -> set[str]:
    if isinstance(only, str):
        raise ConfigError("operations= takes an iterable of operation ids, not one string")
    return set(only)


def _raw_operations(
    document: Mapping[str, Any], resolver: Resolver
) -> Iterator[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]]:
    for path, raw_item in document.get("paths", {}).items():
        check_path(str(path))
        item = _resolved_ref(raw_item, resolver)
        if not isinstance(item, Mapping):
            raise ConfigError(f"OpenAPI path item {path!r} must be a mapping")
        for method in METHODS:
            op = item.get(method)
            if op is None:
                continue
            if not isinstance(op, Mapping):
                raise ConfigError(f"OpenAPI operation {method.upper()} {path} must be a mapping")
            yield method, str(path), item, op


def _resolved_ref(node: Any, resolver: Resolver) -> Any:
    """`node`, or the target of a `$ref` node (one level; parts are resolved when used)."""
    if isinstance(node, Mapping) and isinstance(node.get("$ref"), str):
        return pointer(resolver.document, node["$ref"])
    return node


def check_path(path: str) -> None:
    """Refuse a path key that could leave the server: it is joined to `base_url` as is.

    Raises:
        ConfigError: not one leading `/`, a `?`, `#` or backslash, or a `.` / `..` segment.
    """
    segments = path.split("/")[1:]
    if (
        not path.startswith("/")
        or path.startswith("//")
        or any(char in path for char in "?#\\")
        or any(segment in (".", "..") for segment in segments)
    ):
        raise ConfigError(f"unsafe OpenAPI path {path!r}: it must be a plain absolute path")


def _checked_operation(
    key: str, report: ImportReport, build: Callable[[], Operation | None]
) -> Operation | None:
    """`build()`; None when skipped, malformed shapes as ConfigError naming the operation."""
    try:
        return build()
    except SchemaTooLarge as exc:
        report.skip(key, str(exc))
        return None
    except ConfigError:
        raise
    except (AttributeError, TypeError, KeyError, ValueError, RecursionError) as exc:
        logger.warning("malformed OpenAPI operation %s: %s", key, type(exc).__name__)
        raise ConfigError(f"malformed OpenAPI operation {key}: {type(exc).__name__}") from exc


def _operation(
    resolver: Resolver,
    method: str,
    path: str,
    item: Mapping[str, Any],
    op: Mapping[str, Any],
    key: str,
    name: str,
    report: ImportReport,
) -> Operation | None:
    builder = _InputBuilder(resolver.derived)
    request_sizes: list[int] = []  # held only when the operation is kept
    parameters = _parameters(resolver, item, op, request_sizes)
    _check_placeholders(path, parameters, key)
    for param in parameters:
        builder.add_parameter(param)
    body = "none"
    raw_body = op.get("requestBody")
    if raw_body is not None:
        resolved = resolver.resolve(raw_body, deferred=request_sizes)
        request_body = _mapping(resolved, f"{key}: requestBody")
        content = _mapping(request_body.get("content", {}), f"{key}: requestBody.content")
        media = _json_media(content)
        if media is None:
            kinds = ", ".join(str(k) for k in content) or "none"
            report.skip(key, f"request body is not JSON ({kinds})")
            return None
        body = "required" if request_body.get("required") else "optional"
        schema = _mapping(media.get("schema", {}), f"{key}: request body schema")
        builder.add_body(normalise(schema, resolver.derived), body == "required")
    tags = op.get("tags", [])
    if not isinstance(tags, list):
        raise ConfigError(f"malformed OpenAPI operation {key}: tags must be a list")
    input_schema = builder.schema()
    # The fields budget first: a skipped operation holds nothing, so skips never add up to
    # the document-wide held budget.
    if not report.fits(key, len(input_schema["properties"])):
        return None
    resolver.hold(request_sizes)
    return Operation(
        key=key,
        name=name,
        method=method,
        path=path,
        description=str(op.get("description") or op.get("summary") or f"{method.upper()} {path}"),
        input_schema=input_schema,
        output_schema=_output_schema(resolver, op, key, report.dropped),
        bindings=tuple(builder.bindings),
        body=body,  # type: ignore[arg-type]
        tags=tuple(str(tag) for tag in tags),
    )


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"malformed OpenAPI document: {what} must be a mapping")
    return value


def _parameters(
    resolver: Resolver, item: Mapping[str, Any], op: Mapping[str, Any], sizes: list[int]
) -> list[dict[str, Any]]:
    """Path-item parameters overridden by operation parameters with the same name and `in`."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for owner in (item, op):
        raw_list = owner.get("parameters", [])
        if not isinstance(raw_list, list):
            raise ConfigError("malformed OpenAPI document: parameters must be a list")
        for raw in raw_list:
            param = dict(_mapping(resolver.resolve(raw, deferred=sizes), "a parameter"))
            if not isinstance(param.get("name"), str) or not isinstance(param.get("in"), str):
                raise ConfigError("malformed OpenAPI parameter: name and in must be strings")
            if "schema" in param:
                _mapping(param["schema"], f"parameter {param['name']}: schema")
            merged[(param["name"], param["in"])] = param
    return list(merged.values())


def _check_placeholders(path: str, parameters: list[dict[str, Any]], key: str) -> None:
    placeholders = re.findall(r"\{([^{}]*)\}", path)
    names = {str(p["name"]) for p in parameters if p["in"] == "path"}
    if len(placeholders) != len(set(placeholders)):
        raise ConfigError(f"{key}: path {path!r} repeats a placeholder")
    if unknown := sorted(set(placeholders) - names):
        raise ConfigError(f"{key}: path placeholder(s) {unknown} have no path parameter")
    if unused := sorted(names - set(placeholders)):
        raise ConfigError(f"{key}: path parameter(s) {unused} have no placeholder in {path!r}")


def _json_media(content: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for media_type, media in content.items():
        if isinstance(media_type, str) and _JSON_MEDIA.match(media_type):
            return _mapping(media, f"media type {media_type}")
    return None


def _output_schema(
    resolver: Resolver, op: Mapping[str, Any], key: str, dropped: list[str]
) -> dict[str, Any] | None:
    """The JSON schema of the lowest 2xx response; None when absent or too large (`dropped`)."""
    raw = _mapping(op.get("responses") or {}, f"{key}: responses")
    responses = {str(code): value for code, value in raw.items()}
    success = sorted(code for code in responses if code.startswith("2"))
    if not success:
        return None
    response = resolver.resolve(responses[success[0]], optional=True)
    if response is None:
        dropped.append(key)
        return None
    media = _json_media(response.get("content") or {}) if isinstance(response, Mapping) else None
    if media is None or not isinstance(media.get("schema"), Mapping):
        return None
    return normalise(media["schema"], resolver.derived)  # type: ignore[no-any-return]


class _InputBuilder:
    """Accumulates properties, required names and bindings for one operation."""

    def __init__(self, memo: dict[int, tuple[Any, Any]]) -> None:
        self.memo = memo  # the import's `normalise` memo
        self.properties: dict[str, Any] = {}
        self._taken: set[str] = set()  # = set(self.properties), kept up to date by `_add`
        self._counters: dict[str, int] = {}
        self.required: list[str] = []
        self.bindings: list[Binding] = []

    def add_parameter(self, param: Mapping[str, Any]) -> None:
        location, wire = str(param["in"]), str(param["name"])
        if location not in _PARAM_LOCATIONS:
            logger.warning("skipping parameter %s: unknown location %r", wire, location)
            return
        if location == "header" and wire.lower() in _IGNORED_HEADERS:
            return
        if location in ("header", "cookie") and not _TOKEN.match(wire):
            raise ConfigError(f"OpenAPI {location} parameter name {wire!r} is not a valid token")
        schema = dict(normalise(_parameter_schema(param), self.memo))
        if param.get("description") and "description" not in schema:
            schema["description"] = str(param["description"])
        prop = self._name(sanitise(wire), location)
        explode = bool(param.get("explode", param.get("style", "form") == "form"))
        self._add(prop, schema, bool(param.get("required")) or location == "path")
        self.bindings.append(Binding(prop, wire, location, explode))  # type: ignore[arg-type]

    def add_body(self, schema: Mapping[str, Any], required: bool) -> None:
        properties = schema.get("properties")
        flat = isinstance(properties, Mapping) and schema.get("type", "object") == "object"
        if not flat or any(k in schema for k in ("allOf", "anyOf", "oneOf")):
            prop = self._name("body", "body")
            self._add(prop, dict(schema), required)
            self.bindings.append(Binding(prop, "", "body_whole"))
            return
        body_required = set(schema.get("required", ())) if required else set()
        for wire, sub in properties.items():  # type: ignore[union-attr]
            if isinstance(sub, Mapping) and sub.get("readOnly"):
                continue  # set by the server, never sent
            prop = self._name(sanitise(wire), "body")
            self._add(prop, sub, wire in body_required)
            self.bindings.append(Binding(prop, str(wire), "body"))

    def schema(self) -> dict[str, Any]:
        root: dict[str, Any] = {"type": "object", "properties": self.properties}
        if self.required:
            root["required"] = self.required
        root["additionalProperties"] = False
        return root

    def _add(self, prop: str, schema: Any, required: bool) -> None:
        self.properties[prop] = schema
        self._taken.add(prop)
        if required:
            self.required.append(prop)

    def _name(self, base: str, location: str) -> str:
        """`base`, else `{location}_{base}` on a clash, else a numbered variant."""
        if base not in self._taken:
            return base
        return unique(f"{location}_{base}", self._taken, self._counters)


def _parameter_schema(param: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(param.get("schema"), Mapping):
        return param["schema"]  # type: ignore[no-any-return]
    media = next(iter((param.get("content") or {}).values()), None)
    if isinstance(media, Mapping) and isinstance(media.get("schema"), Mapping):
        return media["schema"]  # type: ignore[no-any-return]
    return {"type": "string"}


def normalise(schema: Any, memo: dict[int, tuple[Any, Any]] | None = None) -> Any:
    """OpenAPI 3.0 `nullable: true` as JSON-schema null, recursively.

    A node shared by several parents (a repeated `$ref`) is normalised once per `memo`;
    pass the import's memo (`Resolver.derived`) so shared nodes stay shared across schemas.
    """
    done: dict[int, tuple[Any, Any]] = {} if memo is None else memo
    if isinstance(schema, list | Mapping) and id(schema) in done:
        return done[id(schema)][1]
    if isinstance(schema, list):
        out_list = [normalise(item, done) for item in schema]
        done[id(schema)] = (schema, out_list)  # the source is kept so its id is not reused
        return out_list
    if not isinstance(schema, Mapping):
        return schema
    out: dict[str, Any] = {
        key: normalise(value, done)
        for key, value in schema.items()
        if key not in ("nullable", "properties")
    }
    properties = schema.get("properties")
    if isinstance(properties, Mapping):  # a name -> schema map: names are never keywords
        out["properties"] = {name: normalise(sub, done) for name, sub in properties.items()}
    elif "properties" in schema:
        out["properties"] = properties
    result: Any = out
    if schema.get("nullable") is True:
        kind = out.get("type")
        if isinstance(kind, str):
            out["type"] = [kind, "null"]
        elif "type" not in out:
            result = {"anyOf": [out, {"type": "null"}]}
    done[id(schema)] = (schema, result)
    return result
