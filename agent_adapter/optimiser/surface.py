"""The editable surface: path grammar, caps, dimensions, tree validation and tree diffs."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import OptimiserError, log_failure
from agent_adapter.optimiser._schema_check import schema_problems
from agent_adapter.optimiser.model import Edit, HarnessTree, SurfaceCaps

logger = logging.getLogger(__name__)

_IDENT = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
_HOOK = r"[A-Za-z_][A-Za-z0-9_.\-]{0,63}"
_SECTION = r"[A-Za-z_][A-Za-z0-9_\-]{0,63}"

# (pattern key, regex); the groups are (name, field) or (name, node).
_GRAMMAR: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tool/*/description", re.compile(rf"tool/({_IDENT})/(description)")),
    ("tool/*/prompt_hint", re.compile(rf"tool/({_IDENT})/(prompt_hint)")),
    ("hook/*/config.json", re.compile(rf"hook/({_HOOK})/(config\.json)")),
    ("prompt/system", re.compile(r"prompt/(system)()")),
    ("prompt/layout/*", re.compile(rf"prompt/(layout)/({_SECTION})")),
    ("flow/*/*/prompt", re.compile(rf"flow/({_IDENT})/({_IDENT})/prompt")),
)

# Characters per pattern; hosts may tighten or widen these in `caps()`.
DEFAULT_CAPS: Mapping[str, int] = {
    "tool/*/description": 1_200,
    "tool/*/prompt_hint": 600,
    "hook/*/config.json": 4_000,
    "prompt/system": 12_000,
    "prompt/layout/*": 6_000,
    "flow/*/*/prompt": 6_000,
}

Dimension = Literal["procedure", "knowledge"]
DIMENSION_KINDS: Mapping[str, frozenset[str]] = {
    "procedure": frozenset({"tool", "hook"}),
    "knowledge": frozenset({"prompt", "flow"}),
}
_SHOWN = {"procedure": "tool/*, hook/*", "knowledge": "prompt/*, flow/*"}


@dataclass(frozen=True)
class ParsedPath:
    """A component path split into kind, name, field and (for flows) node."""

    kind: str
    name: str
    field: str
    node: str = ""
    pattern: str = ""


class Validates(Protocol):
    """Anything with a host-style `validate(tree) -> problems`."""

    def validate(self, tree: HarnessTree) -> list[str]: ...


def parse_path(path: str) -> ParsedPath:
    """Split a component path.

    Raises:
        OptimiserError: the path is not in the grammar (escapes, unknown kinds or fields).
    """
    for pattern, regex in _GRAMMAR:
        match = regex.fullmatch(path)
        if match is None:
            continue
        kind = pattern.split("/", 1)[0]
        first, second = match.groups()
        if kind == "flow":
            return ParsedPath(kind, first, "prompt", node=second, pattern=pattern)
        return ParsedPath(kind, first, second, pattern=pattern)
    raise OptimiserError(f"{path}: not an editable path")


def cap_key(path: str) -> str:
    """The caps pattern a path falls under.

    Raises:
        OptimiserError: the path is not in the grammar.
    """
    return parse_path(path).pattern


def dimension_problems(edits: Iterable[Edit], dim: Dimension) -> list[str]:
    """Edits a proposer of `dim` may not make (bad paths or another dimension's kinds)."""
    allowed = DIMENSION_KINDS[dim]
    shown = _SHOWN[dim]
    problems: list[str] = []
    for edit in edits:
        try:
            kind = parse_path(edit.path).kind
        except OptimiserError as exc:
            problems.append(str(exc))
            continue
        if kind not in allowed:
            problems.append(f"{edit.path}: the {dim} editor may only edit {shown} paths")
    return problems


def validate_tree(tree: HarnessTree, caps: SurfaceCaps, host: Validates | None = None) -> list[str]:
    """Every problem with `tree`: grammar, kinds, caps, hook config JSON, then the host's own."""
    problems: list[str] = []
    for path in sorted(tree.files):
        problems += _path_problems(path, tree.files[path], caps)
    if host is not None:
        problems += _host_problems(host, tree)
    return problems


def _path_problems(path: str, text: str, caps: SurfaceCaps) -> list[str]:
    try:
        parsed = parse_path(path)
    except OptimiserError as exc:
        return [str(exc)]
    if parsed.kind not in caps.kinds:
        return [f"{path}: kind {parsed.kind!r} is not editable here"]
    problems: list[str] = []
    cap = caps.max_chars.get(parsed.pattern)
    if cap is not None and len(text) > cap:
        problems.append(f"{path}: {len(text)} chars, cap is {cap}")
    if parsed.kind == "hook":
        problems += [f"{path}: {p}" for p in hook_config_problems(text, caps, parsed.name)]
    return problems


def hook_config_problems(text: str, caps: SurfaceCaps, hook: str) -> list[str]:
    """Problems with one hook's config text: JSON object, then the hook's schema if known."""
    try:
        value = json.loads(text)
    except ValueError as exc:
        return [f"config is not valid JSON ({exc})"]
    if not isinstance(value, dict):
        return ["config must be a JSON object"]
    schema = caps.hook_schemas.get(hook)
    return schema_problems(value, schema) if schema else []


def _host_problems(host: Validates, tree: HarnessTree) -> list[str]:
    try:
        return [str(problem) for problem in host.validate(tree)]
    except Exception as exc:
        if is_control_flow(exc):
            raise
        log_failure(logger, logging.WARNING, exc, "host.validate raised; the tree is rejected")
        return [f"host validation failed: {type(exc).__name__}"]


def edits_between(a: HarnessTree, b: HarnessTree) -> dict[str, str | None]:
    """The edits that turn `a` into `b` (`a.with_edits(...) == b`)."""
    edits: dict[str, str | None] = {p: None for p in a.files if p not in b.files}
    edits.update({p: t for p, t in b.files.items() if a.files.get(p) != t})
    return dict(sorted(edits.items()))


def always_on_chars(tree: HarnessTree) -> int:
    """Characters sent with every request: system prompt, layout sections and tool hints."""
    return sum(
        len(text)
        for path, text in tree.files.items()
        if path.startswith("prompt/") or path.endswith("/prompt_hint")
    )
