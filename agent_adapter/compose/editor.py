"""The editor: one call reads the text blanks (only them) and rewords the ones that clash."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.compose import _fill
from agent_adapter.compose.blanks import agent_identity, word_count
from agent_adapter.compose.checks import check_value
from agent_adapter.compose.citations import cite_ids, plain_words
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
    "You edit parts of one document that different writers wrote. The parts come as JSON, "
    "part id to text. Make them read as one text: one tone, the same terms for the same "
    "things, no repeats. Keep every fact and number, and keep each [words](cite:ID) marker "
    "(you may change its words). Return only the parts you change, each with its full new "
    "text and a short reason."
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
                        "text": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["blank", "text", "reason"],
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
    """Rewords text blanks so they read as one voice; tables, lists and diagrams stay as filled.

    `style` says how the document should read ("formal, short sentences"). The editor sees
    only the text blanks, returns only the ones it changes, and is skipped when there are
    fewer than two text blanks or under `min_words` words in all. A revision that changes
    which sources a blank cites, or fails the blank's checks, is rejected and recorded.
    """

    model: LLMPort
    style: str = ""
    instructions: str = ""
    min_words: int = 150

    def identity(self) -> dict[str, Any]:
        """What changes the editor's answer (for the cache key)."""
        model = agent_identity(self.model)
        return {"model": model, "style": self.style, "instructions": self.instructions}

    def texts(self, plan: _fill.Plan, entries: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
        """The text blanks it would read, or nothing when editing is not worth a call."""
        found = {
            name: str(entry["value"])
            for name, entry in entries.items()
            if plan.template.blanks[name].kind == "text" and entry["value"] is not None
        }
        words = sum(word_count(plain_words(text)) for text in found.values())
        return found if len(found) >= 2 and words >= self.min_words else {}

    def messages(self, texts: Mapping[str, str]) -> list[Message]:
        style = f"Style: {self.style}." if self.style else ""
        system = "\n\n".join(part for part in (_RULES, style, self.instructions) if part)
        data = json.dumps(dict(texts), ensure_ascii=False)
        return [Message(role="system", content=system), Message(role="user", content=data)]

    async def revise(
        self, plan: _fill.Plan, state: ComposeState, ctx: ToolContext
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The changed blank entries and every revision (accepted or not), as plain data."""
        template = plan.template
        entries = {entry["blank"]: entry for entry in state.values}
        texts = self.texts(plan, entries)
        if not texts:
            return [], []
        request = LLMRequest(
            messages=tuple(self.messages(texts)),
            output_schema=_SCHEMA,
            stage=_fill.stage(template, _fill.EDITOR),
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
            name, after = proposal["blank"], proposal["text"]
            if name not in texts:
                revisions.append(_rejected(name, "", proposal, "there is no such text part"))
                continue
            before = texts[name]
            if after.strip() == before.strip():
                continue
            problems = _locked(before, after)
            problems += check_value(template.blanks[name], after, set(cite_ids(before)))
            if problems:
                revisions.append(_rejected(name, before, proposal, *problems))
                continue
            changed.append({**entries[name], "value": after, "problems": []})
            revisions.append(Revision(name, before, after, proposal["reason"]))
        _edited(ctx, template.name, revisions)
        return changed, [revision.to_dict() for revision in revisions]


def _locked(before: str, after: str) -> list[str]:
    """A revision must cite the same sources as the text it replaces."""
    old, new = set(cite_ids(before)), set(cite_ids(after))
    if old == new:
        return []
    lost, added = sorted(old - new), sorted(new - old)
    said = [f"drops citations {lost}" if lost else "", f"adds citations {added}" if added else ""]
    return ["the revision " + " and ".join(part for part in said if part)]


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
    fields = ("blank", "text", "reason")
    return [
        {key: str(item.get(key) or "") for key in fields}
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    ]


def _rejected(name: str, before: str, proposal: Mapping[str, str], *problems: str) -> Revision:
    after, reason = proposal["text"], proposal["reason"]
    return Revision(name, before, after, reason, accepted=False, problems=tuple(problems))


def _edited(ctx: ToolContext, document: str, revisions: Sequence[Revision]) -> None:
    accepted = sum(revision.accepted for revision in revisions)
    event = make_event(
        EventType.COMPOSE_EDITED,
        ctx,
        document=document,
        revised=accepted,
        rejected=len(revisions) - accepted,
    )
    emit_with_content(ctx.sink, event, lambda: {"revisions": [r.to_dict() for r in revisions]})
