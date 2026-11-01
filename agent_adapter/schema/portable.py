"""Check a JSON schema against the subset every supported provider accepts."""

from __future__ import annotations

from typing import Any

_REJECTED_KEYWORDS = {
    "pattern": "regex pattern",
    "patternProperties": "patternProperties",
    "minLength": "minLength",
    "maxLength": "maxLength",
    "prefixItems": "tuple",
    "oneOf": "oneOf union",
    "discriminator": "discriminated union",
    "uniqueItems": "uniqueItems",
    "minProperties": "minProperties",
    "maxProperties": "maxProperties",
}
_ALLOWED_FORMATS = frozenset({"date-time"})


def check_portable(schema: dict[str, Any]) -> list[str]:
    """Return the portability problems in `schema`; an empty list means portable."""
    problems: list[str] = []
    if schema.get("type") != "object":
        problems.append("$: root must be an object")
    _walk(schema, "$", schema.get("$defs", {}), problems, ())
    return problems


def _walk(
    node: dict[str, Any],
    path: str,
    defs: dict[str, Any],
    problems: list[str],
    stack: tuple[str, ...],
) -> None:
    ref = node.get("$ref")
    if ref is not None:
        name = ref.rsplit("/", 1)[-1]
        if not ref.startswith("#/$defs/") or name not in defs:
            problems.append(f"{path}: unresolvable $ref {ref}")
        elif name in stack:
            problems.append(f"{path}: recursive model {name} is not portable")
        else:
            _walk(defs[name], path, defs, problems, (*stack, name))
        return

    for keyword, label in _REJECTED_KEYWORDS.items():
        if keyword in node:
            problems.append(f"{path}: {label} is not portable")
    fmt = node.get("format")
    if fmt is not None and fmt not in _ALLOWED_FORMATS:
        problems.append(f"{path}: format {fmt!r} is not portable")
    if "enum" in node and not all(isinstance(v, str) for v in node["enum"]):
        problems.append(f"{path}: enum values must be strings")
    if "const" in node and not isinstance(node["const"], str):
        problems.append(f"{path}: enum values must be strings")

    if "anyOf" in node:
        branches = node["anyOf"]
        if len([b for b in branches if b.get("type") != "null"]) > 1:
            problems.append(f"{path}: union of several types is not portable")
        for i, branch in enumerate(branches):
            _walk(branch, f"{path}|{i}", defs, problems, stack)
        return
    if "allOf" in node:
        if len(node["allOf"]) > 1:
            problems.append(f"{path}: intersection (allOf) is not portable")
        for i, part in enumerate(node["allOf"]):
            _walk(part, f"{path}&{i}", defs, problems, stack)
        return

    kind = node.get("type")
    if kind is None:
        if "enum" not in node and "const" not in node:
            problems.append(f"{path}: untyped value (Any) is not portable")
        return
    if isinstance(kind, list):
        non_null = [k for k in kind if k != "null"]
        if not non_null:
            problems.append(f"{path}: type list without a non-null type is not portable")
            return
        if len(non_null) != 1 or len(kind) != 2:
            problems.append(f"{path}: union of several types is not portable")
            return
        kind = non_null[0]
    if kind == "object":
        extra = node.get("additionalProperties")
        if extra is True or isinstance(extra, dict):
            problems.append(f"{path}: open map (dict) is not portable")
        for name, sub in node.get("properties", {}).items():
            _walk(sub, f"{path}.{name}", defs, problems, stack)
    elif kind == "array":
        items = node.get("items")
        if isinstance(items, dict):
            _walk(items, f"{path}[]", defs, problems, stack)
        elif "prefixItems" not in node:  # a tuple is already reported once
            problems.append(f"{path}: list without an item type is not portable")
