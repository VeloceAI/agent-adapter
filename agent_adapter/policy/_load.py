"""Read a policy file (YAML, TOML or JSON) and validate it; errors name the file, line and key."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.errors import MissingDependency
from agent_adapter.policy.errors import PolicyError
from agent_adapter.policy.model import PolicySpec

logger = logging.getLogger(__name__)

MAX_BYTES = 1_000_000
MAX_DEPTH = 32  # a policy needs about 5; deeper input is refused before it is built
YAML_HINT = "YAML policies need pyyaml: pip install pyyaml (or use .toml or .json)"
Lines = dict[tuple[str | int, ...], int]


def read_file(path: Path) -> tuple[dict[str, Any], Lines]:
    """The data in a policy file and the line of each key path (YAML exact, others best effort).

    Raises:
        PolicyError: unreadable, too large, an unknown suffix, a parse error or duplicate keys.
        MissingDependency: a YAML file without pyyaml.
    """
    source = str(path)
    suffix = path.suffix.lower()
    if suffix not in (".yaml", ".yml", ".toml", ".json"):
        raise PolicyError("a policy file ends in .yaml, .yml, .toml or .json", source=source)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PolicyError(f"cannot read the file ({type(exc).__name__})", source=source) from exc
    if len(raw) > MAX_BYTES:
        raise PolicyError(f"a policy file is at most {MAX_BYTES} bytes", source=source)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyError("a policy file must be UTF-8", source=source) from exc
    try:
        if suffix in (".yaml", ".yml"):
            data, lines = _yaml(text, source)
        elif suffix == ".toml":
            data, lines = _toml(text, source), {}
        else:
            data, lines = _json(text, source), {}
    except RecursionError:
        raise PolicyError("the file nests too deeply", source=source) from None
    _check_depth(data, source)
    if not isinstance(data, dict):
        raise PolicyError("the top level of a policy must be a mapping", source=source)
    return data, lines or _guess_lines(text, data)


def _check_depth(data: Any, source: str) -> None:
    """Refuse data nested deeper than MAX_DEPTH (checked without recursion).

    Raises:
        PolicyError: too deep.
    """
    stack: list[tuple[Any, int]] = [(data, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise PolicyError(f"a policy nests at most {MAX_DEPTH} levels", source=source)
        items = (
            value.values()
            if isinstance(value, Mapping)
            else value
            if isinstance(value, list)
            else ()
        )
        stack.extend((item, depth + 1) for item in items)


def validate(data: Any, source: str = "", lines: Lines | None = None) -> PolicySpec:
    """`data` as a `PolicySpec`.

    Raises:
        PolicyError: the first invalid or unknown key, with its line when known.
    """
    if isinstance(data, PolicySpec):
        return data
    if not isinstance(data, Mapping):
        raise PolicyError("a policy is a mapping, a PolicySpec or a file path", source=source)
    try:
        return PolicySpec.model_validate(dict(data))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = tuple(first["loc"])
        more = len(exc.errors()) - 1
        # The message only, never the input value: errors reach logs and terminals.
        message = first["msg"].removeprefix("Value error, ")
        if more:
            message += f" (and {more} more)"
        raise PolicyError(
            message, source=source, key=_key(loc), line=_line(lines or {}, loc)
        ) from None


def _key(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        elif not _is_union_tag(part):
            out += f".{part}" if out else str(part)
    return out


def _is_union_tag(part: Any) -> bool:
    """Pydantic adds the union member's type name ('literal[True]', 'dict[str,...]') to locs."""
    return isinstance(part, str) and ("[" in part or part in ("str", "bool", "is-instance"))


def _line(lines: Lines, loc: tuple[Any, ...]) -> int:
    path = tuple(part for part in loc if not _is_union_tag(part))
    while path:
        if path in lines:
            return lines[path]
        path = path[:-1]
    return 0


def _yaml(text: str, source: str) -> tuple[Any, Lines]:
    try:
        import yaml
    except ImportError as exc:
        raise MissingDependency(YAML_HINT) from exc
    try:
        _no_aliases(yaml, text, source)  # before anything expands them (alias bombs, loops)
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        data = yaml.safe_load(text)
    except RecursionError:
        raise PolicyError("the YAML nests too deeply", source=source) from None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else 0
        raise PolicyError("not valid YAML", source=source, line=line) from None
    lines: Lines = {}
    if node is not None:
        _walk_yaml(node, (), lines, source)
    return data, lines


def _no_aliases(yaml: Any, text: str, source: str) -> None:
    """Refuse YAML anchors, aliases and deep nesting, read from the event stream.

    Raises:
        PolicyError: an anchor, an alias or too deep a nesting, with its line.
    """
    depth = 0
    for event in yaml.parse(text, Loader=yaml.SafeLoader):
        line = event.start_mark.line + 1
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
            raise PolicyError(
                "YAML anchors and aliases are not supported in a policy", source=source, line=line
            )
        if isinstance(event, yaml.CollectionStartEvent):
            depth += 1
            if depth > MAX_DEPTH:
                raise PolicyError(
                    f"a policy nests at most {MAX_DEPTH} levels", source=source, line=line
                )
        elif isinstance(event, yaml.CollectionEndEvent):
            depth -= 1


def _walk_yaml(node: Any, path: tuple[str | int, ...], lines: Lines, source: str) -> None:
    lines[path] = node.start_mark.line + 1
    if node.id == "mapping":
        seen: set[str] = set()
        for key, value in node.value:
            name = str(key.value)
            if name in seen:
                raise PolicyError(
                    "a key appears twice", source=source, key=name, line=key.start_mark.line + 1
                )
            seen.add(name)
            _walk_yaml(value, (*path, name), lines, source)
            lines[(*path, name)] = key.start_mark.line + 1
    elif node.id == "sequence":
        for index, item in enumerate(node.value):
            _walk_yaml(item, (*path, index), lines, source)


def _toml(text: str, source: str) -> Any:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        found = re.search(r"line (\d+)", str(exc))
        line = int(found.group(1)) if found else 0
        raise PolicyError("not valid TOML", source=source, line=line) from None


def _json(text: str, source: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        twice = sorted({key for key in keys if keys.count(key) > 1})
        if twice:
            raise PolicyError("a key appears twice", source=source, key=twice[0])
        return dict(pairs)

    try:
        return json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise PolicyError("not valid JSON", source=source, line=exc.lineno) from None


def _guess_lines(text: str, data: Mapping[str, Any]) -> Lines:
    """The first line naming each key path's last key, after its parent's line (TOML, JSON)."""
    rows = text.splitlines()
    lines: Lines = {}

    def find(name: str, after: int) -> int:
        pattern = re.compile(rf"""^\s*(\[\[?)?["']?([\w.]*\.)?{re.escape(name)}["']?\s*(\]|[=:])""")
        for number in range(after, len(rows)):
            if pattern.search(rows[number]):
                return number + 1
        return 0

    def walk(value: Any, path: tuple[str | int, ...], after: int) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                line = find(str(key), after)
                if line:
                    lines[(*path, str(key))] = line
                walk(item, (*path, str(key)), line or after)
        elif isinstance(value, list):
            name = path[-1] if path else ""
            start = after
            for index, item in enumerate(value):
                header = find(str(name), start) if index else start
                if not header:  # an inline list: later items cannot be told apart by line
                    return
                lines[(*path, index)] = header
                walk(item, (*path, index), header)
                start = header

    walk(data, (), 0)
    return lines
