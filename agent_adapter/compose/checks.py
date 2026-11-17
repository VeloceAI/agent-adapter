"""Checks a filled blank must pass: required, word limit, columns, diagram links, citations."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

from agent_adapter.compose.blanks import Blank, word_count
from agent_adapter.compose.citations import cite_ids, plain_words, texts_of

MAX_DIAGRAM_NODES = 60


def check_value(blank: Blank, value: Any, known: Collection[str] | None = None) -> list[str]:
    """Every problem with a blank's value; `known` are the source ids it may cite."""
    if is_empty(blank, value):
        return ["the answer is empty"] if blank.required else []
    found: list[str] = []
    if blank.kind == "number" and not math.isfinite(float(value)):
        found.append("the number is not finite")
    if blank.kind == "table":
        found.extend(_table_problems(blank, value))
    if blank.kind == "diagram":
        found.extend(_diagram_problems(value))
    texts = texts_of(blank.kind, value)
    if blank.max_words is not None and blank.kind in ("text", "list"):
        words = sum(word_count(plain_words(text)) for text in texts)
        if words > blank.max_words:
            found.append(f"the answer has {words} words; the limit is {blank.max_words}")
    cited = [source for text in texts for source in cite_ids(text)]
    unknown = sorted({source for source in cited if source not in (known or ())})
    if unknown:
        found.append(f"the answer cites unknown sources {unknown}; cite only the listed ids")
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
