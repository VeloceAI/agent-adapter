"""The finished document: blanks drawn in the template's format, citations numbered, checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_adapter.compose import _fill
from agent_adapter.compose.citations import Citations, Marks, finish
from agent_adapter.compose.formats import problems, render
from agent_adapter.compose.results import Citation, StageUsage
from agent_adapter.compose.template import REFERENCES, Template
from agent_adapter.core.context import ToolContext
from agent_adapter.retrieval.documents import Document, Source

DOCUMENT = "document"
_SLOT = "\x00references\x00"


def assemble(
    template: Template,
    entries: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Any],
    sources: Sequence[Document],
    style: Citations,
) -> tuple[str, list[Citation], dict[str, tuple[str, ...]]]:
    """The document, its citations, and every issue left (by blank, "document" for the text)."""
    marks = Marks(template.format)
    rendered: dict[str, str] = {}
    for name, entry in entries.items():
        if name in template.blanks:
            blank = template.blanks[name]
            rendered[name] = render(blank, entry["value"], template.format, marks.piece(name))
    slot = _SLOT if template.has_references() else None
    if slot is not None:
        rendered[REFERENCES] = slot
    text = template.render(rendered, inputs)
    known = known_sources(template, sources, entries)
    text, citations, unknown = finish(text, marks, known, style, slot=slot)
    issues: dict[str, tuple[str, ...]] = {}
    for name in template.blanks:
        left = entries.get(name, {}).get("problems", ["not filled"])
        if left:
            issues[name] = tuple(left)
    whole = [*problems(text, template.format), *unknown, *_numbering(citations)]
    if whole:
        issues[DOCUMENT] = tuple(whole)
    return text, citations, issues


def known_sources(
    template: Template, sources: Sequence[Document], entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Document]:
    """Every source a blank may have cited: the document's, the blanks' own, the retrieved."""
    found = {doc.id: doc for doc in sources}
    for blank in template.blanks.values():
        found.update((doc.id, doc) for doc in blank.sources)
    for entry in entries.values():
        for source_id, kept in (entry.get("sources") or {}).items():
            if source_id not in found:
                where = Source(uri=kept.get("uri", ""), title=kept.get("title", ""))
                found[source_id] = Document(id=source_id, text=kept.get("text", ""), source=where)
    return found


def _numbering(citations: Sequence[Citation]) -> list[str]:
    numbers = sorted({citation.number for citation in citations})
    if numbers != list(range(1, len(numbers) + 1)):
        return [f"the reference numbers {numbers} skip a number"]
    return []


def usage(ctx: ToolContext, template: Template) -> dict[str, StageUsage]:
    """Calls, tokens and cost per stage, read from the run's ledger."""
    stages = {_fill.stage(template, name): "blanks" for name in template.blanks}
    stages[_fill.stage(template, _fill.EDITOR)] = "editor"
    stages[_fill.stage(template, _fill.VERIFIER)] = "verifier"
    totals = {part: [0, 0, 0, 0.0] for part in ("blanks", "editor", "verifier")}
    if ctx.ledger is not None:
        for entry in ctx.ledger.entries(ctx.run_id):
            part = stages.get(entry.stage)
            if part is None:
                continue
            total = totals[part]
            total[0] += 0 if entry.cache_hit else 1
            total[1] += entry.input_tokens
            total[2] += entry.output_tokens
            total[3] += entry.cost_usd or 0.0
    return {
        part: StageUsage(int(calls), int(tokens_in), int(tokens_out), float(cost))
        for part, (calls, tokens_in, tokens_out, cost) in totals.items()
    }
