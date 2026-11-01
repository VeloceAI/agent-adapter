"""Render the OpenAPI 3.1 document as plain 3.0.3: no oneOf, anyOf or allOf anywhere."""

from __future__ import annotations

import copy
from typing import Any

_REF = "#/components/schemas/"
_KEPT = frozenset(
    {
        "title",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxProperties",
        "minProperties",
        "required",
        "enum",
        "type",
        "description",
        "format",
        "default",
        "nullable",
        "readOnly",
        "writeOnly",
        "example",
        "externalDocs",
        "deprecated",
        "$ref",
    }
)
_TYPES = {bool: "boolean", int: "integer", float: "number", str: "string"}
PROBLEM_NOTE = "An RFC 9457 problem (application/problem+json)"


def downgrade(document: dict[str, Any], *, gateway: bool) -> dict[str, Any]:
    """`document` (3.1) as 3.0.3; `gateway` keeps only invoke operations, without auth blocks."""
    out = copy.deepcopy(document)
    out["openapi"] = "3.0.3"
    components = out["components"]
    if gateway:
        out["paths"] = {p: item for p, item in out["paths"].items() if p.startswith("/tools/")}
        out["paths"] = {p: item for p, item in out["paths"].items() if p.endswith("/invoke")}
        out.pop("security", None)
        components.pop("securitySchemes", None)
        components.pop("responses", None)
    converter = _Converter(components["schemas"])
    components["schemas"] = {name: converter.component(name) for name in converter.source}
    for item in out["paths"].values():
        for operation in item.values():
            _operation(operation, converter, gateway=gateway)
    for response in components.get("responses", {}).values():
        _content(response, converter)
    if gateway:
        components["schemas"] = _reachable(out["paths"], components["schemas"])
    return out


def _operation(operation: dict[str, Any], converter: _Converter, *, gateway: bool) -> None:
    if gateway:
        operation.pop("security", None)
    for parameter in operation.get("parameters", []):
        parameter["schema"] = converter.convert(parameter["schema"])
    if "requestBody" in operation:
        _content(operation["requestBody"], converter)
    responses = operation.get("responses", {})
    for status, response in list(responses.items()):
        if gateway and "$ref" in response:  # a gateway takes JSON bodies only
            responses[status] = {"description": PROBLEM_NOTE}
        else:
            _content(response, converter)


def _content(holder: dict[str, Any], converter: _Converter) -> None:
    for media in holder.get("content", {}).values():
        if "schema" in media:
            media["schema"] = converter.convert(media["schema"])


def _reachable(paths: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    """The schemas `paths` refer to, directly or through other schemas."""
    seen: set[str] = set()
    pending = _refs(paths)
    while pending:
        name = pending.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        pending.extend(_refs(schemas[name]))
    return {name: schema for name, schema in schemas.items() if name in seen}


def _refs(node: Any) -> list[str]:
    if isinstance(node, dict):
        ref = node.get("$ref")
        found = [ref.removeprefix(_REF)] if isinstance(ref, str) and ref.startswith(_REF) else []
        return found + [r for value in node.values() for r in _refs(value)]
    if isinstance(node, list):
        return [r for item in node for r in _refs(item)]
    return []


class _Converter:
    """Converts JSON Schema 2020-12 nodes to OpenAPI 3.0 Schema Objects."""

    def __init__(self, source: dict[str, Any]) -> None:
        self.source = source
        self._inlining: list[str] = []

    def component(self, name: str) -> dict[str, Any]:
        """The component `name`, converted; a reference back to it stays a ref."""
        self._inlining.append(name)
        try:
            return self.convert(self.source[name])
        finally:
            self._inlining.pop()

    def convert(self, node: Any) -> dict[str, Any]:
        """One schema node, its sub-schemas converted too."""
        if node is True or node == {}:
            return {}
        if node is False:
            return {"not": {}}
        if not isinstance(node, dict):
            return {}
        node = dict(node)
        nullable = False
        # anyOf next to oneOf means both must hold; 3.0 has no intersection, so take the union.
        while "anyOf" in node or "oneOf" in node:
            branches = [*node.pop("anyOf", []), *node.pop("oneOf", [])]
            if branches:
                node, null = self._union(node, branches)
                nullable = nullable or null
        if "allOf" in node:
            node = self._all_of(node, node.pop("allOf"))
        if isinstance(node.get("$ref"), str) and (len(node) > 1 or nullable):
            node = self._inlined(node)
        out = self._keywords(node)
        if (out.pop("nullable", False) or nullable) and "$ref" not in out:
            out["nullable"] = True  # a ref left by a cycle cannot carry it
        return out

    def _keywords(self, node: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "type":
                out.update(_type(value))
            elif key == "const":
                out["enum"] = [value]
            elif key == "properties" and isinstance(value, dict):
                out["properties"] = {name: self.convert(sub) for name, sub in value.items()}
            elif key == "items" and isinstance(value, dict | bool):
                if value is not False:
                    out["items"] = self.convert(value)
            elif key == "prefixItems" and isinstance(value, list):
                out.setdefault("items", self._flat(value) if value else {})
            elif key in ("additionalProperties", "not"):
                out[key] = (
                    value if isinstance(value, bool) and key != "not" else self.convert(value)
                )
            elif key == "examples" and isinstance(value, list) and value:
                out.setdefault("example", value[0])
            elif key in ("exclusiveMinimum", "exclusiveMaximum") and not isinstance(value, bool):
                continue  # numeric 3.1 bounds: see _bounds
            elif key in _KEPT or key.startswith("x-"):
                out.setdefault(key, value)
        if out.get("type") == "array" and "items" not in out:
            out["items"] = {}
        if not out.get("required"):
            out.pop("required", None)
        _bounds(node, out)
        return _enum_nulls(out)

    def _union(self, node: dict[str, Any], branches: list[Any]) -> tuple[dict[str, Any], bool]:
        """One schema for `branches` overlaid with `node`, and whether null is allowed."""
        nullable = False
        rest: list[Any] = []
        for branch in branches:
            if isinstance(branch, dict) and _is_null(branch):
                nullable = True
            else:
                rest.append(branch)
        if not rest:
            return {**node, "type": "null"}, nullable
        if len(rest) == 1:
            single = rest[0] if isinstance(rest[0], dict) else {}
            if "$ref" in single and nullable:
                return {**node, **single}, True  # inlined below, so `nullable` applies
            return _overlay(single, node), nullable
        flat = self._flat(rest)
        return _overlay(flat, node), nullable or bool(flat.pop("nullable", False))

    def _flat(self, branches: list[Any]) -> dict[str, Any]:
        """One converted schema that accepts every branch (a tagged union keeps its tag)."""
        resolved = [self._resolved(b) for b in branches]
        types = {s.get("type") for s in resolved}
        if len(types) != 1 or None in types:
            return {}
        kind = types.pop()
        if kind == "object":
            return _objects(resolved)
        merged: dict[str, Any] = {"type": kind}
        if all("enum" in s for s in resolved):
            merged["enum"] = _unique([v for s in resolved for v in s["enum"]])
        if kind == "array":
            items = [s.get("items", {}) for s in resolved]
            merged["items"] = items[0] if all(i == items[0] for i in items) else {}
        return merged

    def _resolved(self, branch: Any) -> dict[str, Any]:
        if isinstance(branch, dict) and isinstance(branch.get("$ref"), str):
            return self._inlined(branch)
        return self.convert(branch)

    def _all_of(self, node: dict[str, Any], branches: list[Any]) -> dict[str, Any]:
        merged = dict(node)
        for branch in branches:
            part = (
                self._inlined(branch) if isinstance(branch, dict) and "$ref" in branch else branch
            )
            if not isinstance(part, dict):
                continue
            props = {**part.get("properties", {}), **merged.get("properties", {})}
            required = _unique([*merged.get("required", []), *part.get("required", [])])
            merged = {**part, **merged}
            if props:
                merged["properties"] = props
            if required:
                merged["required"] = required
        return merged

    def _inlined(self, node: dict[str, Any]) -> dict[str, Any]:
        """The schema `node["$ref"]` names, under `node`'s other keys; a cycle stays a ref."""
        ref = node["$ref"]
        name = ref.removeprefix(_REF)
        if not ref.startswith(_REF) or name not in self.source or name in self._inlining:
            return {"$ref": ref}
        self._inlining.append(name)
        try:
            target = self.convert(self.source[name])
        finally:
            self._inlining.pop()
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        return _overlay(target, siblings)


def _objects(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """One object schema with every branch's properties; a shared one-value tag becomes an enum."""
    properties: dict[str, Any] = {}
    for schema in schemas:
        for name, sub in schema.get("properties", {}).items():
            properties[name] = sub if name not in properties else _either(properties[name], sub)
    required = set.intersection(*(set(s.get("required", [])) for s in schemas))
    merged: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        merged["required"] = sorted(required)
    return merged  # no discriminator: 3.0 allows one only beside oneOf/anyOf/allOf


def _bounds(node: dict[str, Any], out: dict[str, Any]) -> None:
    """Numeric exclusive bounds as 3.0 booleans; when both bounds are set the stricter wins."""
    for side, key in (("minimum", "exclusiveMinimum"), ("maximum", "exclusiveMaximum")):
        exclusive: Any = node.get(key)
        inclusive: Any = node.get(side)
        if not _number(exclusive):
            continue
        if _number(inclusive) and (
            inclusive > exclusive if side == "minimum" else inclusive < exclusive
        ):
            continue  # the inclusive bound already excludes the exclusive one
        out[side] = exclusive
        out[key] = True


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _either(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a == b:
        return a
    if a.get("type") == b.get("type") and "enum" in a and "enum" in b:
        return {**a, "enum": _unique([*a["enum"], *b["enum"]])}
    return {}


def _overlay(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """`base` with `extra`'s annotations (title, description, default) on top."""
    return {**base, **{k: v for k, v in extra.items() if k not in ("type", "$ref")}}


def _type(value: Any) -> dict[str, Any]:
    kinds = value if isinstance(value, list) else [value]
    rest = [k for k in kinds if k != "null"]
    out: dict[str, Any] = {"nullable": True} if "null" in kinds else {}
    if len(rest) == 1:
        out["type"] = rest[0]
    elif not rest:
        out["enum"] = [None]
    return out


def _enum_nulls(out: dict[str, Any]) -> dict[str, Any]:
    enum = out.get("enum")
    if isinstance(enum, list) and None in enum:
        out["nullable"] = True
        if any(v is not None for v in enum):
            out["enum"] = [v for v in enum if v is not None]
    if isinstance(enum, list) and "type" not in out:
        kinds = {_TYPES.get(type(v)) for v in out["enum"] if v is not None}
        if len(kinds) == 1 and None not in kinds:
            out["type"] = kinds.pop()
    return out


def _is_null(branch: dict[str, Any]) -> bool:
    return branch.get("type") == "null" or branch.get("const", ...) is None


def _unique(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
