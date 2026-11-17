"""`compose()`: fill a template's blanks as a flow, edit, verify and assemble the document."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_adapter.compose import _assemble, _fill
from agent_adapter.compose._flow import ComposeState, build_flow
from agent_adapter.compose.blanks import Blank
from agent_adapter.compose.citations import Citations, CiteFn, Style, as_citations
from agent_adapter.compose.editor import Editor
from agent_adapter.compose.formats import Format
from agent_adapter.compose.results import BlankResult, ComposeResult, Revision
from agent_adapter.compose.sources import TOP_K
from agent_adapter.compose.template import Template
from agent_adapter.compose.verifier import Verifier
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import CacheScope
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec.defaults import default_executor
from agent_adapter.exec.executor import Executor
from agent_adapter.flow import Flow, FlowRunner
from agent_adapter.hooks.builtin.budget import BudgetHook
from agent_adapter.retrieval.documents import Document, as_document
from agent_adapter.retrieval.port import RetrieverPort
from agent_adapter.runtime.sync import run_sync

logger = logging.getLogger(__name__)


async def compose(
    template: Template | str | Path,
    *,
    model: LLMPort | None = None,
    inputs: Mapping[str, Any] | None = None,
    sources: Sequence[Document | str | Mapping[str, Any]] = (),
    retriever: RetrieverPort | None = None,
    top_k: int = TOP_K,
    citations: Citations | Style | CiteFn = "numeric",
    editor: Editor | bool = True,
    verifier: Verifier | bool = True,
    max_usd: float | None = None,
    blanks: Sequence[Blank] = (),
    format: Format | None = None,
    executor: Executor | None = None,
    runner: FlowRunner | None = None,
    ctx: ToolContext | None = None,
    thread_id: str = "",
    concurrency: int | None = 4,
    on_blank: _fill.BlankCallback | None = None,
    cache_scope: CacheScope | None = "thread",
    cache_ttl: float | None = None,
) -> ComposeResult:
    """Fill every blank (independent ones in parallel), edit, verify, and return the document.

    `template` is a `Template`, a file path (format from the extension) or template text.
    `model` fills each blank with no `agent` of its own, and is the default editor and
    verifier. Each blank sees only its hint, the blanks it uses and the best `top_k`
    passages of `sources` (or of `retriever`); `citations` draws what it cites. `max_usd`
    stops the run once it costs that much. `on_blank(result)` (sync or async) gets each
    blank as it fills. Blanks are cached by what they depend on (`cache_scope` on the
    executor's cache; "thread" needs a `thread_id`).

    Raises:
        ToolUsageError: an input is missing, a `ctx` has no run id, or a bad option.
        FlowError: a blank's model or agent call failed.
        BudgetExceeded: the run went over `max_usd`.
    """
    template = _template(template, blanks, format)
    inputs = dict(inputs or {})
    missing = template.missing_inputs(inputs)
    if missing:
        raise ToolUsageError(f"template {template.name!r} needs inputs {missing}")
    orphans = [name for name, blank in template.blanks.items() if blank.agent is None]
    if model is None and orphans:
        raise ToolUsageError(f"blanks {orphans} have no agent; pass model= to compose()")
    style = as_citations(citations)
    docs = tuple(as_document(item) for item in sources)
    if runner is not None:
        if max_usd is not None:
            raise ToolUsageError("max_usd needs executor=, not runner=; add a BudgetHook to it")
        executor = runner.executor
    executor = executor or default_executor()
    if runner is None:
        executor = _metered(executor, max_usd, ctx)
    runner = runner or FlowRunner(executor, concurrency=concurrency)
    plan = _fill.Plan(
        template,
        model,
        executor,
        sources=docs,
        retriever=retriever,
        k=top_k,
        on_blank=on_blank,
        editor=_editor(editor, model),
        verifier=_verifier(verifier, model, bool(docs) or retriever is not None),
    )
    if executor.cache is None and (ctx is None or ctx.cache is None):
        cache_scope = None
    flow = build_flow(plan, cache_scope=cache_scope, cache_ttl=cache_ttl)
    if ctx is not None:
        if not ctx.run_id:
            raise ToolUsageError("ctx needs a run id; use executor.context() or a tool's ctx")
        return await _compose(runner, flow, plan, inputs, style, _with_plan(ctx, plan))
    async with executor.run_scope(thread_id=thread_id) as scoped:
        return await _compose(runner, flow, plan, inputs, style, _with_plan(scoped, plan))


def compose_sync(template: Template | str | Path, **options: Any) -> ComposeResult:
    """`compose()` from sync code (it runs the coroutine on a private loop if one is running)."""
    return run_sync(compose(template, **options))


def _template(
    given: Template | str | Path, blanks: Sequence[Blank], fmt: Format | None
) -> Template:
    options: dict[str, Any] = {} if fmt is None else {"format": fmt}
    if isinstance(given, Template):
        if blanks or fmt is not None:
            raise ToolUsageError("blanks= and format= go to Template(), not with a Template")
        return given
    if isinstance(given, str) and ("{{" in given or "\n" in given):
        return Template(given, blanks, **options)
    if isinstance(given, (str, Path)):
        return Template.load(given, blanks, **options)
    raise ToolUsageError(f"template must be a Template, a path or text, got {type(given).__name__}")


def _metered(executor: Executor, max_usd: float | None, ctx: ToolContext | None) -> Executor:
    """The executor with a ledger (for `usage`) and the `max_usd` budget hook."""
    changes: dict[str, Any] = {}
    if executor.ledger is None and ctx is None:
        changes["ledger"] = Ledger()
    if max_usd is not None:
        changes["hooks"] = [*executor.hooks, BudgetHook(max_usd)]
    return executor.replace(**changes) if changes else executor


def _editor(given: Editor | bool | None, model: LLMPort | None) -> Editor | None:
    if given is True:
        return Editor(model) if model is not None else None
    if given is False or given is None:
        return None
    if isinstance(given, Editor):
        return given
    raise ToolUsageError(f"editor must be an Editor or a bool, got {type(given).__name__}")


def _verifier(given: Verifier | bool | None, model: LLMPort | None, cited: bool) -> Verifier | None:
    """The verifier node's settings; None when it has no model call to make."""
    if given is True:
        given = Verifier(model=model if cited else None)
    if given is False or given is None:
        return None
    if not isinstance(given, Verifier):
        raise ToolUsageError(f"verifier must be a Verifier or a bool, got {type(given).__name__}")
    return given if given.model is not None else None


def _with_plan(ctx: ToolContext, plan: _fill.Plan) -> ToolContext:
    return dataclasses.replace(ctx, extras={**ctx.extras, _fill.PLAN_KEY: plan})


async def _compose(
    runner: FlowRunner,
    flow: Flow[ComposeState],
    plan: _fill.Plan,
    inputs: dict[str, Any],
    style: Citations,
    ctx: ToolContext,
) -> ComposeResult:
    template = plan.template
    result = await runner.run(flow, {"inputs": inputs}, ctx)
    if result.status != "done":
        logger.warning("compose %s ended %s", template.name, result.status)
    state = result.state
    revisions = tuple(Revision.from_dict(item) for item in state.revisions)
    changed = {r.blank for r in revisions if r.accepted} | set(state.redone)
    entries = {entry["blank"]: entry for entry in state.values}
    blanks: dict[str, BlankResult] = {}
    for name in template.blanks:
        entry = entries.get(name, {"blank": name, "value": None, "problems": ["not filled"]})
        cached = name not in plan.ran
        cost = 0.0 if cached else _fill.stage_cost(ctx, _fill.stage(template, name))
        found = _fill.blank_result(
            template, entry, cached=cached, cost=cost, edited=name in changed
        )
        blanks[name] = found
        if cached:
            await _fill.filled(ctx, plan, found)
        elif name in changed:
            await _fill.call_back(plan.on_blank, found)
    text, cited, issues = _assemble.assemble(template, entries, inputs, plan.sources, style)
    for item in state.issues:
        issues[item["blank"]] = (*issues.get(item["blank"], ()), item["issue"])
    usage = _assemble.usage(ctx, template)
    return ComposeResult(
        text=text,
        blanks=blanks,
        revisions=revisions,
        cost=sum(part.cost_usd for part in usage.values()),
        run_id=ctx.run_id,
        format=template.format,
        citations=tuple(cited),
        issues=issues,
        usage=usage,
    )
