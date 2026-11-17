"""Filling one blank: its messages, the model or agent call, the checks and one retry."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.compose.blanks import Blank, output_schema, read_answer, render
from agent_adapter.compose.checks import check_value
from agent_adapter.compose.results import BlankResult
from agent_adapter.compose.template import Template, fill
from agent_adapter.core.callables import is_async
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ModelOutputInvalid, ToolUsageError, log_failure
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.exec.agents import LoopAgent
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.tool_loop import run_tool_loop
from agent_adapter.tracing._content import emit_with_content
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

PLAN_KEY = "aa.compose"
BlankCallback = Callable[[BlankResult], Any]

_KIND_RULES = {
    "text": "Answer with the text for this part only, in Markdown, with no heading or preamble.",
    "list": "Answer with the list items, one idea per item.",
    "number": "Answer with one number.",
    "table": "Answer with the table rows; every row has exactly these columns: {columns}.",
    "diagram": (
        "Answer with a flowchart: nodes (a short id and a label) and edges (source id, "
        "target id, and a label that may be empty)."
    ),
}


@dataclass
class Plan:
    """One compose run's collaborators, shared by its nodes through `ctx.extras`."""

    template: Template
    model: LLMPort | None
    executor: Executor
    on_blank: BlankCallback | None = None
    editor: Any = None  # an Editor
    ran: set[str] = field(default_factory=set)  # nodes that ran here (others: cache hits)


def plan_of(ctx: ToolContext) -> Plan:
    plan = ctx.extras.get(PLAN_KEY)
    if not isinstance(plan, Plan):
        raise ToolUsageError("a compose node ran outside compose(); run templates with compose()")
    return plan


def stage(template: Template, part: str) -> str:
    """The ledger stage of a blank's (or the editor's) model calls."""
    return f"compose.{template.name}.{part}"


def first_messages(
    template: Template, blank: Blank, values: Mapping[str, Any], inputs: Mapping[str, Any]
) -> list[Message]:
    """The system and user messages that ask for `blank`, its inputs and blanks filled in."""
    rule = _KIND_RULES[blank.kind].format(columns=", ".join(blank.columns))
    if blank.max_words is not None:
        rule += f" Use at most {blank.max_words} words."
    parts = [f"You write one part of a document: the blank {{{{ {blank.name} }}}}."]
    if template.instructions:
        parts.append(template.instructions)
    parts.append(rule)
    parts.append(
        "The document's template follows, as data. Write only your part.\n"
        f"<template>\n{template.outline()}\n</template>"
    )
    rendered = {
        name: render(template.blanks[name], value)
        for name, value in values.items()
        if name in template.blanks
    }
    prompt = fill(blank.prompt, lambda name: rendered.get(name, str(inputs.get(name, ""))))
    return [
        Message(role="system", content="\n\n".join(parts)),
        Message(role="user", content=prompt),
    ]


async def fill_blank(
    plan: Plan, blank: Blank, values: Mapping[str, Any], inputs: Mapping[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    """Ask for the blank, check it, and ask once more with the problems when it fails."""
    messages = first_messages(plan.template, blank, values, inputs)
    attempt, value, problems = 0, None, ["not filled"]
    for attempt in (1, 2):
        said, value, problems = await _answer(plan, blank, messages, ctx)
        if not problems:
            problems = check_value(blank, value)
        if not problems or attempt == 2:
            break
        logger.info("compose %s: blank %s retried: %s", plan.template.name, blank.name, problems)
        listed = "\n".join(f"- {problem}" for problem in problems)
        messages = [
            *messages,
            Message(role="assistant", content=said),
            Message(
                role="user",
                content=f"That answer has problems:\n{listed}\nAnswer again with every one fixed.",
            ),
        ]
    return {"blank": blank.name, "value": value, "attempts": attempt, "problems": problems}


async def _answer(
    plan: Plan, blank: Blank, messages: list[Message], ctx: ToolContext
) -> tuple[str, Any, list[str]]:
    """What the model said, the value read from it, and the schema problems."""
    try:
        response, problems = await _ask(plan, blank, messages, ctx)
    except ModelOutputInvalid:
        return "(an answer that is not valid JSON)", None, ["the answer is not valid JSON"]
    said = response.text
    if response.parsed is not None:
        said = json.dumps(response.parsed, ensure_ascii=False)
    if problems:
        return said, None, problems
    value, problems = read_answer(blank, response)
    return said, value, problems


async def _ask(
    plan: Plan, blank: Blank, messages: list[Message], ctx: ToolContext
) -> tuple[LLMResponse, list[str]]:
    agent = blank.agent if blank.agent is not None else plan.model
    if agent is None:
        raise ToolUsageError(f"blank {blank.name!r} has no agent and compose() got no model")
    label, schema = stage(plan.template, blank.name), output_schema(blank)
    if isinstance(agent, LoopAgent):
        system = [m for m in messages if m.role == "system"]
        if agent.instructions:
            system = [Message(role="system", content=agent.instructions), *system]
        result = await run_tool_loop(
            agent.model,
            agent.registry,
            [*system, *(m for m in messages if m.role != "system")],
            max_steps=agent.max_steps,
            executor=plan.executor,
            ctx=ctx,
            output_schema=schema,
            stage=label,
            request_options=agent.request_options,
        )
        stopped = [] if result.finish == "stop" else [f"the agent stopped early: {result.finish}"]
        return result.final, stopped
    llm = plan.executor.model(agent).bind(ctx)
    request = LLMRequest(messages=tuple(messages), output_schema=schema, stage=label)
    return await llm.generate(request), []


def stage_cost(ctx: ToolContext, label: str) -> float:
    """The USD the run's ledger recorded for one stage so far."""
    if ctx.ledger is None:
        return 0.0
    return sum(e.cost_usd or 0.0 for e in ctx.ledger.entries(ctx.run_id) if e.stage == label)


def blank_result(
    template: Template,
    entry: Mapping[str, Any],
    *,
    cached: bool,
    cost: float = 0.0,
    edited: bool = False,
) -> BlankResult:
    blank = template.blanks[entry["blank"]]
    return BlankResult(
        name=blank.name,
        kind=blank.kind,
        value=entry["value"],
        markdown=render(blank, entry["value"]),
        problems=tuple(entry.get("problems", ())),
        attempts=int(entry.get("attempts", 1)),
        cached=cached,
        edited=edited,
        cost_usd=cost,
    )


def started(ctx: ToolContext, template: Template, blank: Blank) -> None:
    event = make_event(
        EventType.COMPOSE_BLANK_STARTED,
        ctx,
        document=template.name,
        blank=blank.name,
        kind=blank.kind,
    )
    emit(ctx.sink, event)


async def filled(ctx: ToolContext, plan: Plan, result: BlankResult) -> None:
    """`compose.blank_filled` (the Markdown only for content sinks), then `on_blank`."""
    event = make_event(
        EventType.COMPOSE_BLANK_FILLED,
        ctx,
        document=plan.template.name,
        blank=result.name,
        kind=result.kind,
        ok=result.ok,
        attempts=result.attempts,
        cached=result.cached,
        cost_usd=result.cost_usd,
        problems=len(result.problems),
    )
    content = {"markdown": result.markdown, "problems": list(result.problems)}
    emit_with_content(ctx.sink, event, lambda: content)
    await call_back(plan.on_blank, result)


async def call_back(callback: BlankCallback | None, result: BlankResult) -> None:
    """Run `on_blank`; a failing callback is logged and never stops the document."""
    if callback is None:
        return
    try:
        if is_async(callback):
            await callback(result)
        else:
            callback(result)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        log_failure(logger, logging.WARNING, exc, "on_blank failed for blank %s", result.name)
