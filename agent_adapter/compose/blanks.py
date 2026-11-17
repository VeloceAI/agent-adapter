"""Blanks: what each part of a template asks for, its output schema, and its Markdown."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any, Literal

from agent_adapter.core.messages import LLMResponse, OutputSchema
from agent_adapter.core.ports import LLMPort
from agent_adapter.exec.agents import LoopAgent
from agent_adapter.flow.diagram import _quote

Kind = Literal["text", "list", "number", "table", "diagram"]
KINDS: tuple[Kind, ...] = ("text", "list", "number", "table", "diagram")
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Blank:
    """One `{{ name }}` of a template: the prompt that fills it and the shape of its answer.

    `kind` is "text", "list", "number", "table" (rows with `columns`) or "diagram" (nodes
    and edges, drawn as Mermaid). `agent` fills it instead of the document's model: a model
    port, or a `LoopAgent` with tools.
    """

    name: str
    prompt: str
    kind: Kind = "text"
    _: KW_ONLY
    agent: LLMPort | LoopAgent | None = None
    max_words: int | None = None
    required: bool = True
    columns: Sequence[str] = ()
    direction: Literal["TD", "LR"] = "TD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))

    def problems(self) -> list[str]:
        """What is wrong with this blank's definition (empty when it is fine)."""
        where = f"blank {self.name!r}"
        found = []
        if not isinstance(self.name, str) or not NAME.match(self.name):
            found.append(f"{where}: the name must be letters, digits or _, starting with a letter")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            found.append(f"{where}: the prompt is empty")
        if self.kind not in KINDS:
            found.append(f"{where}: kind must be one of {list(KINDS)}, got {self.kind!r}")
        if self.kind == "table" and not self.columns:
            found.append(f"{where}: a table needs columns=")
        if len(set(self.columns)) != len(self.columns) or not all(
            isinstance(c, str) and c.strip() for c in self.columns
        ):
            found.append(f"{where}: columns must be distinct, non-empty strings")
        if self.max_words is not None and (
            not isinstance(self.max_words, int) or isinstance(self.max_words, bool)
        ):
            found.append(f"{where}: max_words must be an int or None")
        elif self.max_words is not None and self.max_words < 1:
            found.append(f"{where}: max_words must be >= 1")
        if self.direction not in ("TD", "LR"):
            found.append(f"{where}: direction must be 'TD' or 'LR'")
        return found

    def fingerprint(self) -> dict[str, Any]:
        """What changes this blank's answer: its settings and who fills it (for the cache key)."""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "kind": self.kind,
            "max_words": self.max_words,
            "required": self.required,
            "columns": list(self.columns),
            "direction": self.direction,
            "agent": agent_identity(self.agent),
        }


def agent_identity(agent: LLMPort | LoopAgent | None) -> Any:
    """A model's or a loop agent's identity as plain data."""
    if agent is None:
        return None
    if isinstance(agent, LoopAgent):
        tools = sorted(agent.registry.names())
        return {
            "model": agent_identity(agent.model),
            "instructions": agent.instructions,
            "tools": tools,
            "max_steps": agent.max_steps,
        }
    return {"provider": str(agent.provider), "model": str(agent.model)}


def output_schema(blank: Blank) -> OutputSchema | None:
    """The structured-output schema the answer must match; None for free text."""
    if blank.kind == "text":
        return None
    if blank.kind == "number":
        body: dict[str, Any] = {"value": {"type": "number"}}
    elif blank.kind == "list":
        body = {"items": {"type": "array", "items": {"type": "string"}}}
    elif blank.kind == "table":
        cells = {column: {"type": "string"} for column in blank.columns}
        row = _object(cells)
        body = {"rows": {"type": "array", "items": row}}
    else:
        node = _object({"id": {"type": "string"}, "label": {"type": "string"}})
        edge = _object(
            {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "label": {"type": "string"},
            }
        )
        body = {
            "nodes": {"type": "array", "items": node},
            "edges": {"type": "array", "items": edge},
        }
    return OutputSchema(name=f"{blank.kind}_{blank.name}"[:64], json_schema=_object(body))


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def read_answer(blank: Blank, response: LLMResponse) -> tuple[Any, list[str]]:
    """The blank's value from a reply, and the schema problems found while reading it."""
    if blank.kind == "text":
        return response.text.strip(), []
    data = response.parsed
    if data is None:
        try:
            data = json.loads(_unfenced(response.text))
        except ValueError:
            return None, ["the answer is not JSON matching the schema"]
    return from_json(blank, data)


def from_json(blank: Blank, data: Any) -> tuple[Any, list[str]]:
    """The blank's value from its structured answer, and what does not match the schema."""
    if not isinstance(data, Mapping):
        return None, ["the answer is not a JSON object"]
    if blank.kind == "number":
        value = data.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None, ["'value' is not a number"]
        return value, []
    if blank.kind == "list":
        items = data.get("items")
        if not isinstance(items, list):
            return None, ["'items' is not a list"]
        return [str(item).strip() for item in items], []
    if blank.kind == "table":
        rows = data.get("rows")
        if not isinstance(rows, list) or not all(isinstance(r, Mapping) for r in rows):
            return None, ["'rows' is not a list of objects"]
        return [{str(k): _cell(v) for k, v in row.items()} for row in rows], []
    nodes, edges = data.get("nodes"), data.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None, ["'nodes' and 'edges' must be lists"]
    if not all(isinstance(item, Mapping) for item in [*nodes, *edges]):
        return None, ["every node and edge must be an object"]
    clean_nodes = [{"id": _cell(n.get("id")), "label": _cell(n.get("label"))} for n in nodes]
    clean_edges = [
        {
            "source": _cell(e.get("source")),
            "target": _cell(e.get("target")),
            "label": _cell(e.get("label")),
        }
        for e in edges
    ]
    return {"nodes": clean_nodes, "edges": clean_edges}, []


def _cell(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _unfenced(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    return stripped


def render(blank: Blank, value: Any) -> str:
    """The value as Markdown; tables and diagrams are always drawn by this code."""
    if value is None:
        return ""
    if blank.kind == "text":
        return str(value)
    if blank.kind == "number":
        return _number(value)
    if blank.kind == "list":
        return "\n".join(f"- {item}" for item in value)
    if blank.kind == "table":
        return _table(blank.columns, value)
    return _mermaid(blank.direction, value)


def _number(value: float) -> str:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _table(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(_table_cell(cell) for cell in cells) + " |"

    lines = [line(columns), "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend(line([row.get(column, "") for column in columns]) for row in rows)
    return "\n".join(lines)


def _table_cell(text: str) -> str:
    return _SPACE.sub(" ", text).replace("|", "\\|").strip()


def _mermaid(direction: str, value: Mapping[str, Any]) -> str:
    ids = {node["id"]: f"n{index}" for index, node in enumerate(value["nodes"], 1)}
    lines = ["```mermaid", f"flowchart {direction}"]
    for node in value["nodes"]:
        label = _SPACE.sub(" ", node["label"] or node["id"]).strip()
        lines.append(f"  {ids[node['id']]}[{_quote(label)}]")
    for edge in value["edges"]:
        source, target = ids.get(edge["source"]), ids.get(edge["target"])
        if source is None or target is None:
            continue
        label = _SPACE.sub(" ", edge["label"]).strip()
        arrow = f" -->|{_quote(label)}| " if label else " --> "
        lines.append(f"  {source}{arrow}{target}")
    lines.append("```")
    return "\n".join(lines)


def word_count(text: str) -> int:
    """Words in `text`, split on whitespace."""
    return len(text.split())
