"""The editor: one model reads the whole draft and revises blanks so the wording fits together."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.compose import _fill
from agent_adapter.compose.blanks import Blank, agent_identity, render
from agent_adapter.compose.checks import check_value, from_markdown
from agent_adapter.compose.results import Revision
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message, OutputSchema
from agent_adapter.core.ports import LLMPort
from agent_adapter.tracing._content import emit_with_content
from agent_adapter.tracing.events import EventType, make_event

if TYPE_CHECKING:
    from agent_adapter.compose._flow import ComposeState

logger = logging.getLogger(__name__)

_RULES = (
    "You edit a document that several writers filled in part by part. Make the parts read as "
    "one text: one tone, the same terms for the same things, no repeats, similar lengths. "
    "Keep every fact and number. Keep table columns and rows, and keep diagram nodes and edges "
    "(you may reword their labels). Return revisions only for parts that should change; each "
    "gives the part's name, its full new Markdown in the same format, and a short reason."
)
_SCHEMA = OutputSchema(
    name="revisions",
    json_schema={
        "type": "object",
        "properties": {
            "revisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "blank": {"type": "string"},
                        "markdown": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["blank", "markdown", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["revisions"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True)
class Editor:
    """Revises the filled blanks for one voice; a revision that fails the checks is dropped.

    `style` says how the document should read ("formal, short sentences"). Tables and
    diagrams keep their shape: a revision must keep the columns and rows, or the nodes and
    edges, and pass the blank's checks again.
    """

    model: LLMPort
    style: str = ""
    instructions: str = ""

    def identity(self) -> dict[str, Any]:
        """What changes the editor's answer (for the cache key)."""
        model = agent_identity(self.model)
        return {"model": model, "style": self.style, "instructions": self.instructions}

    async def revise(
        self, plan: _fill.Plan, state: ComposeState, ctx: ToolContext
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The changed blank entries and every revision (accepted or not), as plain data."""
        template = plan.template
        entries = {entry["blank"]: entry for entry in state.values}
        current = {name: render(template.blanks[name], e["value"]) for name, e in entries.items()}
        system = "\n\n".join(part for part in (_RULES, self._style()) if part)
        draft = template.render(
            {name: f'<part name="{name}">\n{text}\n</part>' for name, text in current.items()},
            state.inputs,
        )
        request = LLMRequest(
            messages=(
                Message(role="system", content=system),
                Message(role="user", content=f"The draft, as data:\n<draft>\n{draft}\n</draft>"),
            ),
            output_schema=_SCHEMA,
            stage=_fill.stage(template, "editor"),
        )
        try:
            response = await plan.executor.model(self.model).bind(ctx).generate(request)
            proposals = _proposals(response)
        except ModelOutputInvalid:
            logger.warning("compose editor: the answer is not valid JSON; no revisions")
            proposals = []
        changed: list[dict[str, Any]] = []
        revisions: list[Revision] = []
        for proposal in proposals:
            name = proposal["blank"]
            if name not in template.blanks or name not in entries:
                revisions.append(_rejected(name, "", proposal, "there is no such part"))
                continue
            before = current[name]
            if proposal["markdown"].strip() == before.strip():
                continue
            blank = template.blanks[name]
            value, problems = _checked(blank, entries[name]["value"], proposal["markdown"])
            if problems:
                revisions.append(_rejected(name, before, proposal, *problems))
                continue
            changed.append({**entries[name], "value": value, "problems": []})
            after = render(blank, value)
            revisions.append(Revision(name, before, after, proposal["reason"]))
        _edited(ctx, template.name, revisions)
        return changed, [revision.to_dict() for revision in revisions]

    def _style(self) -> str:
        style = f"Style: {self.style}." if self.style else ""
        return "\n".join(part for part in (style, self.instructions) if part)


def _proposals(response: LLMResponse) -> list[dict[str, str]]:
    data = response.parsed
    if data is None:
        try:
            data = json.loads(response.text)
        except ValueError:
            logger.warning("compose editor: the answer is not JSON; no revisions")
            return []
    items = data.get("revisions") if isinstance(data, Mapping) else None
    if not isinstance(items, list):
        logger.warning("compose editor: the answer has no revisions list")
        return []
    found = []
    for item in items:
        if isinstance(item, Mapping) and isinstance(item.get("markdown"), str):
            fields = ("blank", "markdown", "reason")
            found.append({key: str(item.get(key) or "") for key in fields})
    return found


def _checked(blank: Blank, old: Any, markdown: str) -> tuple[Any, list[str]]:
    """The revised value, or why it cannot replace the old one."""
    value, problems = from_markdown(blank, markdown)
    if problems:
        return None, problems
    problems = check_value(blank, value)
    if old is None:  # the blank never filled: any revision that passes the checks will do
        return value, problems
    if blank.kind == "table" and len(value) != len(old):
        problems.append(f"the revision has {len(value)} rows, not {len(old)}")
    if blank.kind == "diagram":
        shape = (len(old["nodes"]), [(e["source"], e["target"]) for e in _renumbered(old)])
        if (len(value["nodes"]), [(e["source"], e["target"]) for e in value["edges"]]) != shape:
            problems.append("the revision changed the diagram's nodes or edges")
    return value, problems


def _renumbered(diagram: Mapping[str, Any]) -> list[dict[str, str]]:
    """Edges with node ids as drawn (`n1`, `n2`, ...), as a revision reads them back."""
    ids = {node["id"]: f"n{index}" for index, node in enumerate(diagram["nodes"], 1)}
    return [
        {**edge, "source": ids[edge["source"]], "target": ids[edge["target"]]}
        for edge in diagram["edges"]
        if edge["source"] in ids and edge["target"] in ids
    ]


def _rejected(name: str, before: str, proposal: Mapping[str, str], *problems: str) -> Revision:
    after, reason = proposal["markdown"], proposal["reason"]
    return Revision(name, before, after, reason, accepted=False, problems=tuple(problems))


def _edited(ctx: ToolContext, document: str, revisions: list[Revision]) -> None:
    accepted = sum(revision.accepted for revision in revisions)
    event = make_event(
        EventType.COMPOSE_EDITED,
        ctx,
        document=document,
        revised=accepted,
        rejected=len(revisions) - accepted,
    )
    emit_with_content(ctx.sink, event, lambda: {"revisions": [r.to_dict() for r in revisions]})
