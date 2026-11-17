"""Output formats: detecting one, escaping model text, drawing values, and well-formed checks."""

from __future__ import annotations

import html
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from xml.sax.saxutils import escape as xml_escape
from xml.sax.saxutils import quoteattr

from agent_adapter.flow.diagram import _quote

if TYPE_CHECKING:
    from agent_adapter.compose.blanks import Blank

Format = Literal["markdown", "html", "xml", "text"]
FORMATS: tuple[Format, ...] = ("markdown", "html", "xml", "text")
Piece = Callable[[str], str]
_SPACE = re.compile(r"\s+")
_EXTENSIONS: dict[str, Format] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".txt": "text",
}
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}
_OPTIONAL_END = {"p", "li", "td", "th", "tr", "thead", "tbody", "html", "body", "head", "dt", "dd"}


def detect(text: str, path: str | Path | None = None) -> Format:
    """The format from a file extension, else from the content (Markdown by default)."""
    if path is not None:
        found = _EXTENSIONS.get(Path(path).suffix.lower())
        if found is not None:
            return found
    head = text.lstrip()[:200].lower()
    if head.startswith("<?xml"):
        return "xml"
    if head.startswith(("<!doctype html", "<html")) or re.search(r"<(body|div|p|h1)\b", head):
        return "html"
    return "markdown"


def escape(text: str, fmt: Format) -> str:
    """`text` safe to place in a document of `fmt` (HTML and XML are escaped)."""
    if fmt == "html":
        return html.escape(text, quote=True)
    if fmt == "xml":
        return xml_escape(text, {'"': "&quot;"})
    return text


def render(blank: Blank, value: Any, fmt: Format = "markdown", piece: Piece | None = None) -> str:
    """A blank's value drawn in `fmt`; `piece` turns each model text (default: escape it)."""
    if value is None:
        return ""
    turn: Piece = piece if piece is not None else (lambda text: escape(text, fmt))
    if blank.kind == "text":
        return turn(str(value))
    if blank.kind == "number":
        return turn(number(value))
    if blank.kind == "list":
        return _list([turn(item) for item in value], fmt)
    if blank.kind == "table":
        cells = [[turn(row.get(column, "")) for column in blank.columns] for row in value]
        return _table(blank.columns, cells, fmt)
    return _diagram(blank.direction, value, fmt)


def number(value: float) -> str:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _list(items: Sequence[str], fmt: Format) -> str:
    if fmt == "html":
        return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
    if fmt == "xml":
        return "".join(f"<item>{item}</item>" for item in items)
    return "\n".join(f"- {item}" for item in items)


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]], fmt: Format) -> str:
    if fmt == "html":
        head = "".join(f"<th>{escape(c, fmt)}</th>" for c in columns)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    if fmt == "xml":
        return "".join(
            "<row>"
            + "".join(
                f"<cell column={quoteattr(col)}>{c}</cell>"
                for col, c in zip(columns, row, strict=True)
            )
            + "</row>"
            for row in rows
        )
    if fmt == "text":
        return "\n".join(" | ".join(_flat(c) for c in line) for line in [columns, *rows])

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(_flat(c).replace("|", "\\|") for c in cells) + " |"

    lines = [line(columns), "| " + " | ".join("---" for _ in columns) + " |"]
    return "\n".join([*lines, *(line(row) for row in rows)])


def _flat(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _diagram(direction: str, value: Mapping[str, Any], fmt: Format) -> str:
    from agent_adapter.compose.citations import plain_words

    ids = {node["id"]: f"n{index}" for index, node in enumerate(value["nodes"], 1)}
    labels = {
        node["id"]: _flat(plain_words(node["label"] or node["id"])) for node in value["nodes"]
    }
    edges = [
        (ids[e["source"]], ids[e["target"]], _flat(plain_words(e["label"])))
        for e in value["edges"]
        if e["source"] in ids and e["target"] in ids
    ]
    if fmt == "xml":
        nodes = "".join(
            f"<node id={quoteattr(ids[k])}>{escape(label, fmt)}</node>"
            for k, label in labels.items()
        )
        links = "".join(
            f"<edge from={quoteattr(a)} to={quoteattr(b)}>{escape(label, fmt)}</edge>"
            for a, b, label in edges
        )
        return f"<diagram direction={quoteattr(direction)}>{nodes}{links}</diagram>"
    if fmt == "text":
        names = {ids[k]: label for k, label in labels.items()}
        lines = [
            f"{names[a]} -> {names[b]}" + (f" ({label})" if label else "") for a, b, label in edges
        ]
        return "\n".join(lines or list(names.values()))
    lines = [f"flowchart {direction}"]
    lines.extend(f"  {ids[k]}[{_quote(label)}]" for k, label in labels.items())
    for a, b, label in edges:
        lines.append(f"  {a} -->|{_quote(label)}| {b}" if label else f"  {a} --> {b}")
    source = "\n".join(lines)
    if fmt == "html":
        return f'<pre class="mermaid">\n{html.escape(source, quote=False)}\n</pre>'
    return f"```mermaid\n{source}\n```"


def problems(document: str, fmt: Format) -> list[str]:
    """Why the finished document is not well formed (empty when it is)."""
    if fmt == "xml":
        try:
            ET.fromstring(document.encode("utf-8"))
        except ET.ParseError as exc:
            return [f"the XML is not well formed: {exc}"]
        return []
    if fmt == "html":
        checker = _TagChecker()
        checker.feed(document)
        checker.close()
        return checker.found + [f"<{tag}> is never closed" for tag in checker.open_tags()]
    if fmt == "markdown":
        return _table_shapes(document)
    return []


class _TagChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.found: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        while self.stack and self.stack[-1] != tag and self.stack[-1] in _OPTIONAL_END:
            self.stack.pop()
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.found.append(f"</{tag}> closes nothing open")

    def open_tags(self) -> list[str]:
        return [tag for tag in self.stack if tag not in _OPTIONAL_END]


def _table_shapes(document: str) -> list[str]:
    found: list[str] = []
    widths: list[int] = []
    fence = False
    for number_, line in enumerate([*document.splitlines(), ""], 1):
        if line.strip().startswith("```"):
            fence = not fence
        row = line.strip()
        if not fence and row.startswith("|") and row.endswith("|") and len(row) > 1:
            widths.append(len(re.split(r"(?<!\\)\|", row[1:-1])))
            continue
        if widths and len(set(widths)) > 1:
            shapes, end = sorted(set(widths)), number_ - 1
            found.append(f"the Markdown table ending on line {end} has rows of {shapes} cells")
        widths = []
    return found
