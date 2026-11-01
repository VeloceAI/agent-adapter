"""Per-provider rewrites of a portable JSON schema."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from agent_adapter.core.errors import SchemaNotPortable

_GEMINI_KEYWORDS = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "enum",
        "format",
        "minimum",
        "maximum",
        "items",
        "minItems",
        "maxItems",
        "anyOf",
        "additionalProperties",
        "nullable",
    }
)

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

# Keywords holding child schemas: a name -> schema map, one schema, or a list of schemas.
CHILD_KEYS: tuple[tuple[str, Literal["map", "one", "list"]], ...] = (
    ("properties", "map"),
    ("items", "one"),
    ("additionalProperties", "one"),
    ("anyOf", "list"),
    ("allOf", "list"),
)


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every local $ref with its definition and drop $defs.

    Raises:
        SchemaNotPortable: a $ref is recursive or points nowhere.
    """
    defs: dict[str, Any] = schema.get("$defs", {})

    def resolve(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name in stack:
                raise SchemaNotPortable([f"recursive $ref {name!r} cannot be inlined"])
            if name not in defs:
                raise SchemaNotPortable([f"unresolvable $ref {node['$ref']!r}"])
            target = resolve(defs[name], (*stack, name))
            siblings = {k: resolve(v, stack) for k, v in node.items() if k != "$ref"}
            return {**target, **siblings}
        out = {k: resolve(v, stack) for k, v in node.items() if k != "$defs"}
        all_of = out.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
            del out["allOf"]
            out = {**all_of[0], **out}
        return out

    result: dict[str, Any] = resolve(schema, ())
    return result


def to_plain(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline refs, drop `title` keywords, turn `const` into a one-value `enum`."""
    return _map_nodes(inline_refs(schema), _plain_node)


def to_openai_parameters(schema: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Parameters for OpenAI tools; strict mode requires every field and forbids extras."""
    plain = to_plain(schema)
    return _map_nodes(_defaults_to_descriptions(plain), _strict_node) if strict else plain


def to_gemini_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """`parameters_json_schema` for Gemini: accepted keywords only, defaults in descriptions."""
    return _map_nodes(_defaults_to_descriptions(to_plain(schema)), _gemini_node)


def to_anthropic_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """`input_schema` for Anthropic tools."""
    return to_plain(schema)


def _map_nodes(node: dict[str, Any], fn: NodeFn) -> dict[str, Any]:
    """Apply `fn` to every schema node; property names are never treated as keywords."""
    out = fn(dict(node))
    for key, shape in CHILD_KEYS:
        child = out.get(key)
        if shape == "map" and isinstance(child, dict):
            out[key] = {name: _map_nodes(sub, fn) for name, sub in child.items()}
        elif shape == "one" and isinstance(child, dict):
            out[key] = _map_nodes(child, fn)
        elif shape == "list" and isinstance(child, list):
            out[key] = [_map_nodes(sub, fn) for sub in child]
    return out


def _plain_node(node: dict[str, Any]) -> dict[str, Any]:
    node.pop("title", None)
    if "const" in node:
        node["enum"] = [node.pop("const")]
    return node


def _strict_node(node: dict[str, Any]) -> dict[str, Any]:
    node.pop("default", None)
    if node.get("type") != "object":
        return node
    props = dict(node.get("properties", {}))
    required = set(node.get("required", []))
    for name, sub in props.items():
        if name not in required and not _is_nullable(sub):
            inner = {k: v for k, v in sub.items() if k != "description"}
            wrapper: dict[str, Any] = {"anyOf": [inner, {"type": "null"}]}
            if "description" in sub:
                wrapper["description"] = sub["description"]
            props[name] = wrapper
    node["properties"] = props
    node["required"] = list(props)
    node["additionalProperties"] = False
    return node


def _defaults_to_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Move each object property's `default` into its description; other defaults stay."""
    return _map_nodes(schema, _describe_property_defaults)


def _describe_property_defaults(node: dict[str, Any]) -> dict[str, Any]:
    props = node.get("properties")
    if isinstance(props, dict):
        node["properties"] = {name: _describe_default(sub) for name, sub in props.items()}
    return node


def _describe_default(prop: dict[str, Any]) -> dict[str, Any]:
    if "default" not in prop:
        return prop
    out = dict(prop)
    note = f"Default: {json.dumps(out.pop('default'))}."
    out["description"] = f"{out['description']} {note}" if out.get("description") else note
    return out


def _gemini_node(node: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k in _GEMINI_KEYWORDS}


def _is_nullable(node: dict[str, Any]) -> bool:
    kind = node.get("type")
    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True
    return any(branch.get("type") == "null" for branch in node.get("anyOf", []))
