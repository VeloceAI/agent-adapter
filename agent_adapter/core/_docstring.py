"""Parse Google-style docstrings into a description and per-parameter text."""

from __future__ import annotations

import inspect
import re
from collections.abc import Collection

_SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Return|Raises|Yields|Examples?|Notes?|Attributes):\s*$"
)
_PARAM_RE = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_ARG_SECTIONS = frozenset({"Args", "Arguments", "Parameters"})


def parse_docstring(
    doc: str | None, params: Collection[str] | None = None
) -> tuple[str, dict[str, str]]:
    """Return (description, {param: text}) from a Google-style docstring.

    Args:
        doc: The docstring.
        params: Known parameter names. When given, a `word: text` line whose word is not a
            parameter continues the current parameter instead of starting a new one.
    """
    if not doc:
        return "", {}
    description: list[str] = []
    found: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    param_indent: int | None = None
    for line in inspect.cleandoc(doc).splitlines():
        header = None if line.startswith((" ", "\t")) else _SECTION_RE.match(line.strip())
        if header:
            section, current, param_indent = header.group(1), None, None
            continue
        if section is None:
            description.append(line.rstrip())
            continue
        if section not in _ARG_SECTIONS or not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        match = _PARAM_RE.match(line.strip())
        if match and params is not None and match.group(1) not in params and current is not None:
            match = None
        if match and (param_indent is None or indent <= param_indent):
            param_indent = indent
            current = match.group(1)
            found[current] = match.group(2).strip()
        elif current is not None:
            found[current] = f"{found[current]} {line.strip()}".strip()
    return "\n".join(description).strip(), found
