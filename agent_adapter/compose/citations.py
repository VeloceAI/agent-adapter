"""Citations: `[words](cite:id)` markers, numbering in code, and pluggable renderers."""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_adapter.compose.formats import Format, escape
from agent_adapter.compose.results import Citation
from agent_adapter.retrieval.documents import Document

Style = Literal["numeric", "linked", "footnote", "html", "plain"]
STYLES: tuple[Style, ...] = ("numeric", "linked", "footnote", "html", "plain")
CiteFn = Callable[[str, Document, int], str]
ReferencesFn = Callable[[Sequence[tuple[int, Document]]], str]
MARKER = re.compile(r"\[([^\[\]]*)\]\(cite:([^()\s]+)\)|\[\[([^\[\]\s]+)\]\]")
_SENTINEL = re.compile("\x00(\\d+)\x00")
_SENTENCE_END = re.compile(r"[.!?]\s")
_LISTED = {"numeric": True, "linked": False, "footnote": True, "html": False, "plain": False}


@dataclass(frozen=True)
class Citations:
    """How cited words appear: a built-in `style`, or your own `cite(words, source, number)`.

    `references` adds a References list: None takes the style's default (numeric and
    footnote list them), True or False forces it, or a function `(pairs) -> str` writes it.
    """

    style: Style = "numeric"
    cite: CiteFn | None = None
    references: bool | ReferencesFn | None = None

    def listed(self) -> bool:
        if self.references is None:
            return self.cite is None and _LISTED[self.style]
        return bool(self.references)


def as_citations(value: Citations | Style | CiteFn) -> Citations:
    if isinstance(value, Citations):
        return value
    if isinstance(value, str):
        if value not in STYLES:
            raise ValueError(f"citations must be one of {list(STYLES)}, got {value!r}")
        return Citations(style=value)
    if callable(value):
        return Citations(cite=value)
    raise ValueError(f"citations must be a style name, a Citations or a function, got {value!r}")


def cite_ids(text: str) -> list[str]:
    """The source ids cited in `text`, in order."""
    return [match.group(2) or match.group(3) for match in MARKER.finditer(text)]


def plain_words(text: str) -> str:
    """`text` with each marker replaced by its words."""
    return MARKER.sub(lambda match: match.group(1) or "", text)


def cited_phrases(text: str) -> list[tuple[str, str]]:
    """Each citation in `text` as (the words it backs, the source id); bare ids back the
    sentence before them."""
    return [
        (
            match.group(1) or plain_words(_sentence_before(text, match.start())),
            match.group(2) or match.group(3),
        )
        for match in MARKER.finditer(text)
    ]


def texts_of(kind: str, value: Any) -> list[str]:
    """Every piece of model text inside a value (text, items, cells, labels)."""
    if value is None or kind == "number":
        return []
    if kind == "text":
        return [str(value)]
    if kind == "list":
        return list(value)
    if kind == "table":
        return [cell for row in value for cell in row.values()]
    return [n["label"] for n in value["nodes"]] + [e["label"] for e in value["edges"]]


@dataclass
class _Mark:
    words: str
    source_id: str
    blank: str
    text: str


@dataclass
class Marks:
    """Markers lifted out of the blanks while drawing, replaced by sentinels until numbered."""

    fmt: Format
    marks: list[_Mark] = field(default_factory=list)

    def piece(self, blank: str) -> Callable[[str], str]:
        """A drawing function for `blank`: escape its text, keep each marker as a sentinel."""

        def turn(text: str) -> str:
            out, at = [], 0
            for match in MARKER.finditer(text):
                out.append(escape(text[at : match.start()], self.fmt))
                words = match.group(1) or ""
                said = plain_words(words or _sentence_before(text, match.start()))
                self.marks.append(_Mark(words, match.group(2) or match.group(3), blank, said))
                out.append(f"\x00{len(self.marks) - 1}\x00")
                at = match.end()
            out.append(escape(text[at:], self.fmt))
            return "".join(out)

        return turn


def _sentence_before(text: str, end: int) -> str:
    head = text[:end].rstrip()
    starts = [m.end() for m in _SENTENCE_END.finditer(head)]
    return head[starts[-1] if starts else 0 :][-200:].strip()


def finish(
    document: str,
    marks: Marks,
    sources: Mapping[str, Document],
    style: Citations,
    *,
    slot: str | None,
) -> tuple[str, list[Citation], list[str]]:
    """Number the citations in reading order, draw them and the References list.

    `slot` is the sentinel standing for `{{ references }}` (None: append the list at the end).
    Returns the document, the citations and any issues (unknown ids).
    """
    numbers: dict[str, int] = {}
    by_uri: dict[str, int] = {}
    listed: list[tuple[int, Document]] = []
    citations: list[Citation] = []
    issues: list[str] = []

    def number_of(source: Document) -> int:
        if source.id in numbers:
            return numbers[source.id]
        uri = source.source.uri
        found = by_uri.get(uri) if uri else None
        if found is None:
            found = len(listed) + 1
            listed.append((found, source))
            if uri:
                by_uri[uri] = found
        numbers[source.id] = found
        return found

    def draw(match: re.Match[str]) -> str:
        mark = marks.marks[int(match.group(1))]
        source = sources.get(mark.source_id)
        if source is None:
            issues.append(f"blank {mark.blank!r} cites an unknown source {mark.source_id!r}")
            return escape(mark.words, marks.fmt)
        number = number_of(source)
        citations.append(Citation(number, source, mark.blank, mark.text))
        if style.cite is not None:
            return style.cite(mark.words, source, number)
        return _cite(style.style, mark.words, source, number, marks.fmt)

    body = _SENTINEL.sub(draw, document)
    refs = ""
    if listed and style.listed():
        writer = style.references
        refs = writer(listed) if callable(writer) else _references(style.style, listed, marks.fmt)
    if slot is not None:
        return body.replace(slot, refs), citations, issues
    return _append(body, refs, style.style, marks.fmt), citations, issues


def _cite(style: Style, words: str, source: Document, n: int, fmt: Format) -> str:
    shown, uri = escape(words, fmt), source.source.uri
    if fmt == "markdown":
        shown = shown.replace("|", "\\|")
    gap = " " if shown else ""
    if style == "plain":
        return shown
    if style == "linked" and uri:
        if fmt == "markdown":
            return f"[{shown}]({uri})" if shown else f"[[{n}]]({uri})"
        if fmt == "html":
            return f'<a href="{html.escape(uri)}">{shown or n}</a>'
        if fmt == "text":
            return f"{shown}{gap}({uri})"
    if style == "html" and fmt in ("html", "markdown"):
        title = html.escape(source.source.title or source.id)
        href = f' href="{html.escape(uri)}"' if uri else ""
        return f'<a class="aa-cite"{href} title="{title}">{html.escape(words) or n}</a>'
    if style == "footnote":
        if fmt == "markdown":
            return f"{shown}[^{n}]"
        if fmt == "html":
            return f'{shown}<sup><a href="#aa-ref-{n}">{n}</a></sup>'
        return f"{shown}[{n}]"
    if fmt == "xml":
        return f'<cite ref="{n}">{shown}</cite>'
    return f"{shown}{gap}[{n}]"


def _references(style: Style, listed: Sequence[tuple[int, Document]], fmt: Format) -> str:
    def name(doc: Document) -> str:
        return doc.source.title or doc.id

    if fmt == "html":
        items = "".join(
            f'<li id="aa-ref-{n}">'
            + (
                f'<a href="{html.escape(d.source.uri)}">{html.escape(name(d))}</a>'
                if d.source.uri
                else html.escape(name(d))
            )
            + "</li>"
            for n, d in listed
        )
        return f'<ol class="aa-references">{items}</ol>'
    if fmt == "xml":
        items = "".join(
            f'<source n="{n}" id="{escape(d.id, fmt)}" uri="{escape(d.source.uri, fmt)}">'
            f"{escape(name(d), fmt)}</source>"
            for n, d in listed
        )
        return f"<references>{items}</references>"
    if fmt == "text":
        return "\n".join(
            f"[{n}] {name(d)}" + (f" - {d.source.uri}" if d.source.uri else "") for n, d in listed
        )
    link = {n: (f"[{name(d)}]({d.source.uri})" if d.source.uri else name(d)) for n, d in listed}
    if style == "footnote":
        return "\n".join(f"[^{n}]: {link[n]}" for n, _ in listed)
    return "\n".join(f"{n}. {link[n]}" for n, _ in listed)


def _append(body: str, refs: str, style: Style, fmt: Format) -> str:
    if not refs:
        return body
    if fmt == "html":
        block = f'<section class="aa-references"><h2>References</h2>{refs}</section>'
        at = body.lower().rfind("</body>")
        return body[:at] + block + body[at:] if at >= 0 else body + block
    if fmt == "xml":
        at = body.rstrip().rfind("</")
        return body[:at] + refs + body[at:] if at > 0 else body + refs
    if fmt == "text":
        return body.rstrip() + "\n\nReferences\n" + refs + "\n"
    heading = "" if style == "footnote" else "## References\n\n"
    return body.rstrip() + "\n\n" + heading + refs + "\n"
