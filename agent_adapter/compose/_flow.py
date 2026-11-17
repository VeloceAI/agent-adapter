"""A template compiled to a `Flow`: one node per blank, layered by what each prompt uses."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel

from agent_adapter.compose import _fill
from agent_adapter.compose.blanks import agent_identity
from agent_adapter.compose.template import Template
from agent_adapter.core.context import ToolContext
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import CacheScope
from agent_adapter.flow import Flow, Merge, NodeCache

BEGIN, EDITOR = "aa_begin", "aa_editor"


class ComposeState(BaseModel):
    """The flow state: the inputs, every filled blank (by name) and the editor's revisions."""

    inputs: dict[str, Any] = {}
    values: Annotated[list[dict[str, Any]], Merge.keyed("blank")] = []
    revisions: Annotated[list[dict[str, Any]], Merge.append] = []


def values_of(state: ComposeState) -> dict[str, Any]:
    return {entry["blank"]: entry["value"] for entry in state.values}


def build_flow(
    template: Template,
    *,
    model: LLMPort | None,
    editor: Any,
    cache_scope: CacheScope | None,
    cache_ttl: float | None,
) -> Flow[ComposeState]:
    """The template as a flow; a blank waits for the blanks its prompt uses.

    Each node starts in the superstep after the deepest blank it waits for, so an edge comes
    only from the blanks one layer up; blanks with nobody after them end the flow.
    """
    flow: Flow[ComposeState] = Flow(template.name, state=ComposeState)
    flow.node(BEGIN, _begin)
    model_id = agent_identity(model)
    layers = template.layers()
    for name in template.blanks:
        cache = None
        if cache_scope is not None:
            key = functools.partial(_blank_key, template, name, model_id)
            cache = NodeCache(cache_ttl, key=key, scope=cache_scope)
        flow.node(name, functools.partial(_blank_node, name), cache=cache)
    flow.edge(BEGIN, layers[0])
    for name in template.blanks:
        depth = template.depth[name]
        for dep in template.deps[name]:
            if template.depth[dep] == depth - 1:
                flow.edge(dep, name)
    leaves = [name for name in template.blanks if not template.users(name)]
    if editor is None:
        return flow.end(*leaves)
    cache = None
    if cache_scope is not None:
        key = functools.partial(_editor_key, template, editor.identity())
        cache = NodeCache(cache_ttl, key=key, scope=cache_scope)
    flow.node(EDITOR, _editor_node, cache=cache)
    for name in layers[-1]:
        flow.edge(name, EDITOR)
    return flow.end(*[name for name in leaves if name not in layers[-1]], EDITOR)


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


def _blank_key(template: Template, name: str, model_id: Any, state: ComposeState) -> Any:
    blank = template.blanks[name]
    messages = _fill.first_messages(template, blank, values_of(state), state.inputs)
    sent = [[m.role, m.content] for m in messages]
    return {"blank": blank.fingerprint(), "model": model_id, "messages": sent}


def _editor_key(template: Template, editor_id: Any, state: ComposeState) -> Any:
    values = values_of(state)
    return {
        "editor": editor_id,
        "template": template.text,
        "inputs": state.inputs,
        "values": values,
    }
