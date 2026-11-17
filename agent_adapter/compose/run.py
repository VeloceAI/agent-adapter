"""`compose()`: fill a template's blanks as a flow, edit the draft, and assemble the result."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

from agent_adapter.compose import _fill
from agent_adapter.compose._flow import ComposeState, build_flow
from agent_adapter.compose.editor import Editor
from agent_adapter.compose.results import BlankResult, ComposeResult, Revision
from agent_adapter.compose.template import Template
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import CacheScope
from agent_adapter.exec.defaults import default_executor
from agent_adapter.exec.executor import Executor
from agent_adapter.flow import Flow, FlowRunner

logger = logging.getLogger(__name__)


async def compose(
    template: Template,
    *,
    model: LLMPort | None = None,
    inputs: Mapping[str, Any] | None = None,
    editor: Editor | None = None,
    executor: Executor | None = None,
    runner: FlowRunner | None = None,
    ctx: ToolContext | None = None,
    thread_id: str = "",
    concurrency: int | None = 4,
    on_blank: _fill.BlankCallback | None = None,
    cache_scope: CacheScope | None = "thread",
    cache_ttl: float | None = None,
) -> ComposeResult:
    """Fill every blank (independent ones in parallel), let `editor` revise, return the document.

    `model` fills each blank that has no `agent` of its own. `on_blank(result)` (sync or
    async) gets each blank as it fills, for a live preview. Each blank is cached by its
    prompt and the values it uses (`cache_scope`, on the executor's cache), so a second run
    with one blank changed re-runs only it and the blanks that use it; the default scope
    "thread" needs a `thread_id` (or a `ctx` with one). Pass `runner=` for a `FlowRunner`
    with checkpoints or other settings; its own `concurrency` then applies.

    Raises:
        ToolUsageError: an input is missing, a `ctx` has no run id, or a bad option.
        FlowError: a blank's model or agent call failed.
    """
    inputs = dict(inputs or {})
    missing = template.missing_inputs(inputs)
    if missing:
        raise ToolUsageError(f"template {template.name!r} needs inputs {missing}")
    orphans = [name for name, blank in template.blanks.items() if blank.agent is None]
    if model is None and orphans:
        raise ToolUsageError(f"blanks {orphans} have no agent; pass model= to compose()")
    if editor is not None and not isinstance(editor, Editor):
        raise ToolUsageError(f"editor must be an Editor, got {type(editor).__name__}")
    if runner is not None:
        executor = runner.executor
    executor = executor or default_executor()
    runner = runner or FlowRunner(executor, concurrency=concurrency)
    plan = _fill.Plan(template, model, executor, on_blank=on_blank, editor=editor)
    if executor.cache is None and (ctx is None or ctx.cache is None):
        cache_scope = None
    flow = build_flow(
        template, model=model, editor=editor, cache_scope=cache_scope, cache_ttl=cache_ttl
    )
    if ctx is not None:
        if not ctx.run_id:
            raise ToolUsageError("ctx needs a run id; use executor.context() or a tool's ctx")
        return await _compose(runner, flow, plan, inputs, _with_plan(ctx, plan))
    async with executor.run_scope(thread_id=thread_id) as scoped:
        return await _compose(runner, flow, plan, inputs, _with_plan(scoped, plan))


def _with_plan(ctx: ToolContext, plan: _fill.Plan) -> ToolContext:
    return dataclasses.replace(ctx, extras={**ctx.extras, _fill.PLAN_KEY: plan})


async def _compose(
    runner: FlowRunner,
    flow: Flow[ComposeState],
    plan: _fill.Plan,
    inputs: dict[str, Any],
    ctx: ToolContext,
) -> ComposeResult:
    template = plan.template
    result = await runner.run(flow, {"inputs": inputs}, ctx)
    if result.status != "done":
        logger.warning("compose %s ended %s", template.name, result.status)
    state = result.state
    revisions = tuple(Revision.from_dict(item) for item in state.revisions)
    edited = {revision.blank for revision in revisions if revision.accepted}
    entries = {entry["blank"]: entry for entry in state.values}
    blanks: dict[str, BlankResult] = {}
    for name in template.blanks:
        entry = entries.get(name, {"blank": name, "value": None, "problems": ["not filled"]})
        cached = name not in plan.ran
        cost = 0.0 if cached else _fill.stage_cost(ctx, _fill.stage(template, name))
        found = _fill.blank_result(template, entry, cached=cached, cost=cost, edited=name in edited)
        blanks[name] = found
        if cached:
            await _fill.filled(ctx, plan, found)
        elif name in edited:
            await _fill.call_back(plan.on_blank, found)
    markdown = template.render({name: found.markdown for name, found in blanks.items()}, inputs)
    editor_cost = _fill.stage_cost(ctx, _fill.stage(template, "editor"))
    return ComposeResult(
        markdown=markdown,
        blanks=blanks,
        revisions=revisions,
        cost=sum(found.cost_usd for found in blanks.values()) + editor_cost,
        run_id=ctx.run_id,
        problems={name: found.problems for name, found in blanks.items() if found.problems},
    )
