"""Checks a filled blank must pass, and reading an edited blank's Markdown back into a value."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from agent_adapter.compose.blanks import Blank, render, word_count

MAX_DIAGRAM_NODES = 60
_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_NODE = re.compile(r'^\s*(n\d+)\["(.*)"\]\s*$')
_EDGE = re.compile(r'^\s*(n\d+)\s*-->\s*(?:\|"(.*)"\|\s*)?(n\d+)\s*$')
_ENTITIES = {"#quot;": '"', "#lt;": "<", "#gt;": ">", "#amp;": "&"}


def check_value(blank: Blank, value: Any) -> list[str]:
    """Every problem with a blank's value: required, word limit, columns, diagram links."""
    if is_empty(blank, value):
        return ["the answer is empty"] if blank.required else []
    found: list[str] = []
    if blank.kind == "number" and not math.isfinite(float(value)):
        found.append("the number is not finite")
    if blank.kind == "table":
        found.extend(_table_problems(blank, value))
    if blank.kind == "diagram":
        found.extend(_diagram_problems(value))
    if blank.max_words is not None and blank.kind in ("text", "list"):
        words = word_count(render(blank, value))
        if words > blank.max_words:
            found.append(f"the answer has {words} words; the limit is {blank.max_words}")
    return found


def is_empty(blank: Blank, value: Any) -> bool:
    """True when nothing usable was written."""
    if value is None:
        return True
    if blank.kind == "text":
        return not str(value).strip()
    if blank.kind == "list":
        return not any(item for item in value)
    if blank.kind == "table":
        return not value
    if blank.kind == "diagram":
        return not value.get("nodes")
    return False


def _table_problems(blank: Blank, rows: list[Mapping[str, str]]) -> list[str]:
    found = []
    wanted = set(blank.columns)
    for index, row in enumerate(rows, 1):
        missing = [c for c in blank.columns if c not in row]
        extra = sorted(set(row) - wanted)
        if missing:
            found.append(f"row {index} lacks columns {missing}")
        if extra:
            found.append(f"row {index} has unknown columns {extra}")
    return found


def _diagram_problems(value: Mapping[str, Any]) -> list[str]:
    nodes, edges = value["nodes"], value["edges"]
    found = []
    if len(nodes) > MAX_DIAGRAM_NODES:
        found.append(f"the diagram has {len(nodes)} nodes; the limit is {MAX_DIAGRAM_NODES}")
    ids = [node["id"] for node in nodes]
    if any(not node_id for node_id in ids):
        found.append("a node has no id")
    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
        found.append(f"node ids repeat: {duplicates}")
    known = set(ids)
    for index, edge in enumerate(edges, 1):
        unknown = [end for end in (edge["source"], edge["target"]) if end not in known]
        if unknown:
            found.append(f"edge {index} links unknown nodes {unknown}")
    return found


def from_markdown(blank: Blank, text: str) -> tuple[Any, list[str]]:
    """An edited blank's Markdown read back into a value, and why it could not be read."""
    text = text.strip()
    if blank.kind == "text":
        return text, []
    if blank.kind == "number":
        try:
            number = float(text.replace(",", ""))
        except ValueError:
            return None, ["the revision is not a number"]
        return (int(number) if number.is_integer() else number), []
    if blank.kind == "list":
        items = []
        for line in text.splitlines():
            if not line.strip():
                continue
            match = _ITEM.match(line)
            if match is None:
                return None, ["the revision is not a Markdown list"]
            items.append(match.group(1).strip())
        return items, []
    if blank.kind == "table":
        return _read_table(blank, text)
    return _read_mermaid(text)


def _read_table(blank: Blank, text: str) -> tuple[Any, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not all(line.startswith("|") and line.endswith("|") for line in lines):
        return None, ["the revision is not a Markdown table"]
    header = _cells(lines[0])
    if header != list(blank.columns):
        return None, [f"the revision changed the columns to {header}"]
    rows = []
    for index, line in enumerate(lines[2:], 1):
        cells = _cells(line)
        if len(cells) != len(header):
            return None, [f"row {index} of the revision has {len(cells)} cells, not {len(header)}"]
        rows.append(dict(zip(header, cells, strict=True)))
    return rows, []


def _cells(line: str) -> list[str]:
    parts = re.split(r"(?<!\\)\|", line.strip()[1:-1])
    return [part.strip().replace("\\|", "|") for part in parts]


def _read_mermaid(text: str) -> tuple[Any, list[str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0].strip() != "```mermaid" or lines[-1].strip() != "```":
        return None, ["the revision is not a mermaid block"]
    if not lines[1].strip().startswith("flowchart "):
        return None, ["the revision is not a mermaid flowchart"]
    nodes, edges = [], []
    for line in lines[2:-1]:
        node, edge = _NODE.match(line), _EDGE.match(line)
        if node is not None:
            nodes.append({"id": node.group(1), "label": _unquote(node.group(2))})
        elif edge is not None:
            label = _unquote(edge.group(2) or "")
            edges.append({"source": edge.group(1), "target": edge.group(3), "label": label})
        else:
            return None, [f"the revision has a line the checks cannot read: {line.strip()[:60]!r}"]
    return {"nodes": nodes, "edges": edges}, []


def _unquote(text: str) -> str:
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return text
