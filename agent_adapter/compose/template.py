"""Markdown templates with `{{ name }}` blanks: parsing, the order blanks fill in, rendering."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_adapter.compose.blanks import NAME, Blank
from agent_adapter.core.errors import FlowDefinitionError

_SLOT = re.compile(r"(\\?)\{\{(.*?)\}\}", re.DOTALL)
RESERVED = ("aa_begin", "aa_editor")


def slots(text: str) -> tuple[list[str], list[str]]:
    """The `{{ name }}` names in `text` in order, and the slots that are not names.

    `\\{{` writes literal braces.
    """
    names: list[str] = []
    bad: list[str] = []
    for match in _SLOT.finditer(text):
        if match.group(1):
            continue
        inner = match.group(2).strip()
        if NAME.match(inner):
            names.append(inner)
        else:
            bad.append(match.group(0))
    return names, bad


def fill(text: str, value_of: Callable[[str], str]) -> str:
    """`text` with every `{{ name }}` replaced by `value_of(name)` and `\\{{` unescaped."""

    def swap(match: re.Match[str]) -> str:
        if match.group(1):
            return match.group(0)[1:]
        inner = match.group(2).strip()
        return value_of(inner) if NAME.match(inner) else match.group(0)

    return _SLOT.sub(swap, text)


class Template:
    """A Markdown document with blanks; any other `{{ name }}` is an input given to `compose`.

    A blank's prompt may use other blanks and inputs too: a blank used in a prompt fills
    first, and blanks that do not depend on each other fill in parallel.

    Raises:
        FlowDefinitionError: every problem found (bad names, duplicates, cycles), together.
    """

    def __init__(
        self,
        text: str,
        blanks: Sequence[Blank],
        *,
        name: str = "document",
        instructions: str = "",
    ) -> None:
        self.text = text
        self.name = name
        self.instructions = instructions
        self.blanks: dict[str, Blank] = {}
        problems: list[str] = []
        if not NAME.match(name):
            problems.append(f"template name {name!r} must be letters, digits or _")
        for blank in blanks:
            if not isinstance(blank, Blank):
                problems.append(f"blanks must be Blank objects, got {type(blank).__name__}")
                continue
            problems.extend(blank.problems())
            if blank.name in self.blanks:
                problems.append(f"blank {blank.name!r} is defined twice")
            if blank.name in RESERVED:
                problems.append(f"blank name {blank.name!r} is reserved")
            self.blanks[blank.name] = blank
        in_text, bad = slots(text)
        problems.extend(f"{slot} in the template is not a blank name" for slot in bad)
        self.deps: dict[str, tuple[str, ...]] = {}
        used = set(in_text)
        for blank in self.blanks.values():
            found, bad = slots(blank.prompt)
            problems.extend(f"{slot} in blank {blank.name!r} is not a name" for slot in bad)
            self.deps[blank.name] = tuple(dict.fromkeys(n for n in found if n in self.blanks))
            used.update(found)
        problems.extend(
            f"blank {name!r} is not used in the template or any prompt"
            for name in self.blanks
            if name not in used
        )
        self.inputs: tuple[str, ...] = tuple(
            dict.fromkeys(
                n for n in [*in_text, *_prompt_names(self.blanks)] if n not in self.blanks
            )
        )
        self.depth: dict[str, int] = {}
        if not problems:
            problems.extend(self._order())
        if problems:
            raise FlowDefinitionError(problems, flow=name)

    def _order(self) -> list[str]:
        """Each blank's depth (longest chain of blanks it waits for); a cycle is a problem."""
        state: dict[str, int] = {}  # 1 = on the current path, 2 = done
        path: list[str] = []
        problems: list[str] = []

        def visit(name: str) -> int:
            if state.get(name) == 2:
                return self.depth[name]
            if state.get(name) == 1:
                cycle = " -> ".join([*path[path.index(name) :], name])
                problems.append(f"blanks wait for each other: {cycle}")
                return 0
            state[name] = 1
            path.append(name)
            deepest = max((visit(dep) + 1 for dep in self.deps[name]), default=0)
            path.pop()
            state[name] = 2
            self.depth[name] = deepest
            return deepest

        for name in self.blanks:
            visit(name)
        return problems[:1]

    def layers(self) -> list[list[str]]:
        """Blank names grouped by depth: every blank in a layer can fill in parallel."""
        found: list[list[str]] = [[] for _ in range(max(self.depth.values(), default=-1) + 1)]
        for name in self.blanks:
            found[self.depth[name]].append(name)
        return found

    def users(self, name: str) -> list[str]:
        """The blanks whose prompts use `name`."""
        return [other for other, deps in self.deps.items() if name in deps]

    def missing_inputs(self, inputs: Mapping[str, Any]) -> list[str]:
        """The inputs the template uses that `inputs` does not give."""
        return [name for name in self.inputs if name not in inputs]

    def render(self, rendered: Mapping[str, str], inputs: Mapping[str, Any]) -> str:
        """The document with blanks (already Markdown) and inputs filled in."""
        return fill(self.text, lambda name: _value(name, rendered, inputs))

    def outline(self, limit: int = 4000) -> str:
        """The template text, cut to `limit` characters, to show a model where its blank sits."""
        return self.text if len(self.text) <= limit else self.text[:limit] + "\n..."

    def __repr__(self) -> str:
        return f"Template({self.name!r}, blanks={list(self.blanks)})"


def _prompt_names(blanks: Mapping[str, Blank]) -> list[str]:
    return [name for blank in blanks.values() for name in slots(blank.prompt)[0]]


def _value(name: str, rendered: Mapping[str, str], inputs: Mapping[str, Any]) -> str:
    if name in rendered:
        return rendered[name]
    return str(inputs.get(name, ""))
