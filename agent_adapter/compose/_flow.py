"""A template compiled to a `Flow`: one node per blank, layered by what each prompt uses."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel

from agent_adapter.compose import _fill
from agent_adapter.compose.blanks import agent_identity
from agent_adapter.compose.template import Template, names
from agent_adapter.core.context import ToolContext
from agent_adapter.core.types import CacheScope
from agent_adapter.flow import Flow, Merge, NodeCache

BEGIN = "aa_begin"
EDITOR, VERIFIER = _fill.EDITOR, _fill.VERIFIER


class ComposeState(BaseModel):
    """The flow state: the inputs, every filled blank (by name), revisions and issues."""

    inputs: dict[str, Any] = {}
    values: Annotated[list[dict[str, Any]], Merge.keyed("blank")] = []
    revisions: Annotated[list[dict[str, Any]], Merge.append] = []
    issues: Annotated[list[dict[str, Any]], Merge.append] = []
    redone: Annotated[list[str], Merge.append] = []


def values_of(state: ComposeState) -> dict[str, Any]:
    return {entry["blank"]: entry["value"] for entry in state.values}


def build_flow(
    plan: _fill.Plan, *, cache_scope: CacheScope | None, cache_ttl: float | None
) -> Flow[ComposeState]:
    """The template as a flow; a blank waits for the blanks its prompt uses.

    Each node starts in the superstep after the deepest blank it waits for, so an edge comes
    only from the blanks one layer up; blanks with nobody after them end the flow. The
    editor, then the verifier, follow the last layer.
    """
    template = plan.template
    flow: Flow[ComposeState] = Flow(template.name, state=ComposeState)
    flow.node(BEGIN, _begin)
    model_id = agent_identity(plan.model)
    layers = template.layers()

    def cache_for(key: Any) -> NodeCache | None:
        return None if cache_scope is None else NodeCache(cache_ttl, key=key, scope=cache_scope)

    for name in template.blanks:
        key = functools.partial(_blank_key, plan, name, model_id)
        flow.node(name, functools.partial(_blank_node, name), cache=cache_for(key))
    flow.edge(BEGIN, layers[0])
    for name in template.blanks:
        depth = template.depth[name]
        for dep in template.deps[name]:
            if template.depth[dep] == depth - 1:
                flow.edge(dep, name)
    ends = [name for name in template.blanks if not template.users(name)]
    last = layers[-1]
    for node, worker, part in (
        (EDITOR, _editor_node, plan.editor),
        (VERIFIER, _verifier_node, plan.verifier),
    ):
        if part is None:
            continue
        key = functools.partial(_after_key, template, part.identity())
        flow.node(node, worker, cache=cache_for(key))
        for name in last:
            flow.edge(name, node)
        ends = [name for name in ends if name not in last]
        last = [node]
    return flow.end(*dict.fromkeys([*ends, *last]))


def _begin(state: ComposeState) -> None:
    return None


async def _blank_node(name: str, state: ComposeState, ctx: ToolContext) -> Mapping[str, Any]:
    plan = _fill.plan_of(ctx)
    plan.ran.add(name)
    blank = plan.template.blanks[name]
    _fill.started(ctx, plan.template, blank)
    entry = await _fill.fill_blank(plan, blank, values_of(state), state.inputs, ctx)
    cost = _fill.stage_cost(ctx, _fill.stage(plan.template, name))
    await _fill.filled(ctx, plan, _fill.blank_result(plan.template, entry, cached=False, cost=cost))
    return {"values": [entry]}


async def _editor_node(state: ComposeState, ctx: ToolContext) -> Mapping[str, Any]:
    plan = _fill.plan_of(ctx)
    plan.ran.add(EDITOR)
    values, revisions = await plan.editor.revise(plan, state, ctx)
    return {"values": values, "revisions": revisions}


async def _verifier_node(state: ComposeState, ctx: ToolContext) -> Mapping[str, Any]:
    plan = _fill.plan_of(ctx)
    plan.ran.add(VERIFIER)
    values, issues = await plan.verifier.check(plan, state, ctx)
    return {"values": values, "issues": issues, "redone": [entry["blank"] for entry in values]}


def _blank_key(plan: _fill.Plan, name: str, model_id: Any, state: ComposeState) -> Any:
    """What a blank's answer depends on: its settings, the values and inputs it uses, its
    sources and how they are picked."""
    template, blank = plan.template, plan.template.blanks[name]
    values = values_of(state)
    used = names(blank.prompt)
    pool = _fill.pool(plan.sources, blank)
    return {
        "blank": blank.fingerprint(),
        "model": model_id,
        "doc": [template.instructions, template.format],
        "values": {n: values.get(n) for n in used if n in template.blanks},
        "inputs": {n: state.inputs.get(n) for n in used if n not in template.blanks},
        "sources": [[doc.id, _digest(doc.text), doc.source.uri] for doc in pool],
        "pick": [plan.k, None if blank.sources else _identity(plan.retriever)],
    }


def _after_key(template: Template, part_id: Any, state: ComposeState) -> Any:
    return {
        "part": part_id,
        "template": template.text,
        "inputs": state.inputs,
        "values": values_of(state),
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _identity(retriever: Any) -> Any:
    if retriever is None:
        return None
    return getattr(retriever, "name", None) or type(retriever).__qualname__
