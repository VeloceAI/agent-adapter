"""The verifier: an optional model checks cited phrases against their sources, and your rules."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_adapter.compose import _fill
from agent_adapter.compose._assemble import known_sources
from agent_adapter.compose.blanks import agent_identity
from agent_adapter.compose.citations import cited_phrases, plain_words, texts_of
from agent_adapter.compose.sources import window
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelOutputInvalid
from agent_adapter.core.messages import LLMRequest, Message, OutputSchema
from agent_adapter.core.ports import LLMPort

if TYPE_CHECKING:
    from agent_adapter.compose._flow import ComposeState

logger = logging.getLogger(__name__)

_RULES = (
    "You check a document. Each claim gives a phrase, the part it is in, and an excerpt of the "
    "source it cites. Report a claim the excerpt does not support. When rules are given, "
    "report each part that breaks one. Report only real problems; say each in one sentence."
)
_SCHEMA = OutputSchema(
    name="issues",
    json_schema={
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"blank": {"type": "string"}, "issue": {"type": "string"}},
                    "required": ["blank", "issue"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["issues"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True)
class Verifier:
    """Checks the finished document; on by default.

    Code checks always run: the document is well formed, every blank is filled, every cited
    id is known and the reference numbers match. With a `model` (a small one will do), one
    call checks each cited phrase against about `window` characters of its source, and the
    `rules`; each blank with an issue is filled again once, alone, then checked again.
    """

    model: LLMPort | None = None
    rules: Sequence[str] = ()
    window: int = 300

    def identity(self) -> dict[str, Any]:
        """What changes the verifier's answer (for the cache key)."""
        return {"model": agent_identity(self.model), "rules": list(self.rules), "w": self.window}

    def claims(
        self, plan: _fill.Plan, entries: Mapping[str, Mapping[str, Any]], names: Sequence[str]
    ) -> list[dict[str, str]]:
        """Each cited phrase in `names`, with a short excerpt of the source it cites."""
        known = known_sources(plan.template, plan.sources, entries)
        found = []
        for name in names:
            entry = entries[name]
            kept = entry.get("sources") or {}
            for text in texts_of(plan.template.blanks[name].kind, entry["value"]):
                for phrase, source_id in cited_phrases(text):
                    source = kept.get(source_id, {}).get("text") or getattr(
                        known.get(source_id), "text", ""
                    )
                    excerpt = window(source, phrase, self.window)
                    found.append({"blank": name, "phrase": phrase, "source": excerpt})
        return found

    def messages(
        self, plan: _fill.Plan, entries: Mapping[str, Mapping[str, Any]], names: Sequence[str]
    ) -> list[Message]:
        """The one request for `names`, or no messages when there is nothing to ask."""
        claims = self.claims(plan, entries, names)
        if not claims and not self.rules:
            return []
        data: dict[str, Any] = {"claims": claims}
        if self.rules:
            data["rules"] = list(self.rules)
            data["parts"] = {
                name: plain_words(" ".join(texts_of(plan.template.blanks[name].kind, value)))
                for name in names
                if (value := entries[name]["value"]) is not None
            }
        user = json.dumps(data, ensure_ascii=False)
        return [Message(role="system", content=_RULES), Message(role="user", content=user)]

    async def review(
        self,
        plan: _fill.Plan,
        entries: Mapping[str, Mapping[str, Any]],
        names: Sequence[str],
        ctx: ToolContext,
    ) -> dict[str, list[str]]:
        """The issues the model found, by blank."""
        messages = self.messages(plan, entries, names) if self.model is not None else []
        if not messages or self.model is None:
            return {}
        stage = _fill.stage(plan.template, _fill.VERIFIER)
        request = LLMRequest(messages=tuple(messages), output_schema=_SCHEMA, stage=stage)
        try:
            response = await plan.executor.model(self.model).bind(ctx).generate(request)
        except ModelOutputInvalid:
            logger.warning("compose verifier: the answer is not valid JSON; no issues")
            return {}
        data = response.parsed if response.parsed is not None else _json(response.text)
        found: dict[str, list[str]] = {}
        for item in data.get("issues", []) if isinstance(data, Mapping) else []:
            if isinstance(item, Mapping) and item.get("blank") in names and item.get("issue"):
                found.setdefault(str(item["blank"]), []).append(str(item["issue"]))
        return found

    async def check(
        self, plan: _fill.Plan, state: ComposeState, ctx: ToolContext
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Redone blank entries, and the issues still left after one redo."""
        entries = {entry["blank"]: entry for entry in state.values}
        names = [n for n in plan.template.reading_order() if entries.get(n, {}).get("value")]
        issues = await self.review(plan, entries, names, ctx)
        if not issues:
            return [], []
        values = {name: entry["value"] for name, entry in entries.items()}
        blanks = plan.template.blanks

        async def redo(name: str) -> dict[str, Any]:
            before = entries[name]["value"]
            return await _fill.fill_blank(
                plan, blanks[name], values, state.inputs, ctx, issues=issues[name], before=before
            )

        redone = await asyncio.gather(*(redo(name) for name in issues))
        logger.info("compose %s: verifier redid %s", plan.template.name, list(issues))
        entries.update((entry["blank"], entry) for entry in redone)
        left = await self.review(plan, entries, list(issues), ctx)
        flat = [{"blank": name, "issue": issue} for name, said in left.items() for issue in said]
        return redone, flat


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None
