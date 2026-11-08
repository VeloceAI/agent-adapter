"""Checks for `SubFlow` nodes (child graph, cycles, field mapping, types) and `NodeCache`."""

from __future__ import annotations

import math
import typing
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.core.tool import ToolSpec
from agent_adapter.flow.actions import Call, Map, NodeCache, SubFlow
from agent_adapter.flow.state import MergeRule, is_additive

if TYPE_CHECKING:
    from agent_adapter.flow.graph import FlowGraph, NodeSpec

_SCOPES = ("run", "thread", "global")
_VALIDATING: ContextVar[tuple[tuple[int, str], ...]] = ContextVar("aa_flow_validating", default=())

Pairs = list[tuple[str, str]]  # (to field, from field)


@contextmanager
def validating(flow: Any, name: str) -> Iterator[None]:
    """Mark `flow` as being validated, so a sub-flow that leads back to it is a cycle.

    Raises:
        FlowDefinitionError: `flow` is already being validated further up (a cycle).
    """
    stack = _VALIDATING.get()
    ids = [entry[0] for entry in stack]
    if id(flow) in ids:
        cycle = [entry[1] for entry in stack[ids.index(id(flow)) :]] + [name]
        raise FlowDefinitionError([f"sub-flow cycle {' -> '.join(cycle)}"], flow=name)
    token = _VALIDATING.set((*stack, (id(flow), name)))
    try:
        yield
    finally:
        _VALIDATING.reset(token)


def check_subflow(
    where: str,
    action: SubFlow,
    parent: type[BaseModel],
    rules: Mapping[str, MergeRule],
    found: list[str],
) -> FlowGraph | None:
    """The child's validated graph, after its mapping and types are checked; None if invalid."""
    validate = getattr(action.flow, "validate", None)
    if not callable(validate):
        found.append(f"{where}: SubFlow needs a Flow, got {type(action.flow).__name__}")
        return None
    try:
        child: FlowGraph = validate()
    except FlowDefinitionError as exc:
        name = getattr(action.flow, "name", "?")
        found.extend(f"{where}: sub-flow {name!r}: {problem}" for problem in exc.problems)
        return None
    problems: list[str] = []
    inputs = input_pairs(action, parent, child.state_model, rules, problems)
    outputs = output_pairs(action, parent, child.state_model, problems)
    if isinstance(action.into, str):
        _check_whole(action.into, parent, child.state_model, problems)
    for target, source in inputs:
        _check_types(parent, source, child.state_model, target, problems)
    for target, source in outputs:
        _check_types(parent, target, child.state_model, source, problems)
    for target, source in outputs:
        if (source, target) in inputs and is_additive(rules[target]):
            problems.append(
                f"field {target!r} goes in and comes back under merge rule "
                f"{rules[target].name!r}, so the parent's value would count twice"
            )
    found.extend(f"{where}: SubFlow {problem}" for problem in dict.fromkeys(problems))
    return child if not problems else None


def input_pairs(
    action: SubFlow,
    parent: type[BaseModel],
    child: type[BaseModel],
    rules: Mapping[str, MergeRule],
    problems: list[str] | None = None,
) -> Pairs:
    """(child field, parent field) copied in; [] for an `input` function."""
    spec = action.input
    if callable(spec):
        return []
    if spec is None:
        return [
            (name, name)
            for name in child.model_fields
            if name in parent.model_fields and not is_additive(rules[name])
        ]
    return _mapping(spec, "input", child, parent, problems)


def output_pairs(
    action: SubFlow,
    parent: type[BaseModel],
    child: type[BaseModel],
    problems: list[str] | None = None,
) -> Pairs:
    """(parent field, child field) written back; [] for a whole-state field or a function."""
    spec = action.into
    if callable(spec) or isinstance(spec, str):
        return []
    if spec is None:
        return [(name, name) for name in parent.model_fields if name in child.model_fields]
    return _mapping(spec, "into", parent, child, problems)


def _mapping(
    spec: Any,
    label: str,
    to_model: type[BaseModel],
    from_model: type[BaseModel],
    problems: list[str] | None,
) -> Pairs:
    out = problems if problems is not None else []
    if not isinstance(spec, Mapping):
        out.append(f"{label} must be a field mapping, a function or None, got {spec!r}")
        return []
    pairs: Pairs = []
    for target, source in spec.items():
        if not isinstance(target, str) or target not in to_model.model_fields:
            out.append(f"{label}: {target!r} is not a field of {to_model.__name__}")
        elif not isinstance(source, str) or source not in from_model.model_fields:
            out.append(f"{label}: {source!r} is not a field of {from_model.__name__}")
        else:
            pairs.append((target, source))
    return pairs


def _check_whole(
    field: str, parent: type[BaseModel], child: type[BaseModel], problems: list[str]
) -> None:
    info = parent.model_fields.get(field)
    if info is None:
        problems.append(f"into {field!r} is not a field of {parent.__name__}")
    elif not _accepts(info.annotation, child):
        problems.append(f"into {field!r} cannot hold a {child.__name__}")


def _accepts(annotation: Any, model: type[BaseModel]) -> bool:
    if annotation is Any or annotation is model:
        return True
    members = typing.get_args(annotation) if typing.get_origin(annotation) in _UNIONS else ()
    return any(member is Any or member is model for member in members)


_UNIONS = (typing.Union, type(int | None))


def _check_types(
    parent: type[BaseModel], p_field: str, child: type[BaseModel], c_field: str, out: list[str]
) -> None:
    left = parent.model_fields[p_field].annotation
    right = child.model_fields[c_field].annotation
    if left is Any or right is Any or left == right:
        return
    out.append(
        f"field {p_field!r} ({_name(left)}) and child field {c_field!r} ({_name(right)}) "
        "have different types; map them with input= / into="
    )


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation).replace("typing.", "")


def check_cache(where: str, node: NodeSpec, fields: Mapping[str, Any], found: list[str]) -> None:
    """Check `cache=`: options, and no pause gate or approval-gated tool on the node."""
    cache = node.cache
    if not isinstance(cache, NodeCache):
        found.append(f"{where}: cache must be a NodeCache, got {type(cache).__name__}")
        return
    found.extend(f"{where}: cache {problem}" for problem in _cache_options(cache, fields))
    if node.pause is not False:
        found.append(f"{where}: cache cannot be used on a node with a pause gate")
    if any(tool.confirm for tool in _static_tools(node.action)):
        found.append(f"{where}: cache cannot be used on a node whose tool needs approval")
    child = node.child
    if child is not None and _gated(child):
        found.append(f"{where}: cache cannot be used on a sub-flow with pause gates or approvals")


def _cache_options(cache: NodeCache, fields: Mapping[str, Any]) -> Iterator[str]:
    ttl = cache.ttl
    bad_ttl = isinstance(ttl, bool) or not isinstance(ttl, int | float)
    if ttl is not None and (bad_ttl or not math.isfinite(ttl) or ttl <= 0):
        yield f"ttl must be a positive number or None, got {ttl!r}"
    if cache.key is not None and not callable(cache.key):
        yield f"key must be callable, got {cache.key!r}"
    if cache.key is not None and cache.reads is not None:
        yield "takes key= or reads=, not both"
    reads = cache.reads
    if reads is not None:
        if isinstance(reads, str) or not isinstance(reads, Sequence) or not reads:
            yield f"reads must be a non-empty list of state fields, got {reads!r}"
        else:
            yield from (f"reads {r!r}, not a state field" for r in reads if r not in fields)
    if cache.scope not in _SCOPES:
        yield f"scope must be one of {list(_SCOPES)}, got {cache.scope!r}"


def _static_tools(action: Any) -> list[ToolSpec]:
    tool = action.tool if isinstance(action, Call | Map) else None
    return [tool] if isinstance(tool, ToolSpec) else []


def _gated(graph: FlowGraph) -> bool:
    """True when a node of `graph` (or of a sub-flow inside it) has a gate or a confirm tool."""
    for node in graph.nodes.values():
        if node.pause is not False or any(t.confirm for t in _static_tools(node.action)):
            return True
        if node.child is not None and _gated(node.child):
            return True
    return False


def tool_names(graph: FlowGraph) -> Iterator[tuple[str, str, str]]:
    """(flow, node, tool name) of every cached node naming its tool by string, sub-flows too."""
    for name, node in graph.nodes.items():
        tool = node.action.tool if isinstance(node.action, Call | Map) else None
        if node.cache is not None and isinstance(tool, str):
            yield graph.name, name, tool
        if node.child is not None:
            yield from tool_names(node.child)
