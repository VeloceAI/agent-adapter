"""Templates with `{{ name }}` blanks in any format: parsing, hints, fill order, rendering."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_adapter.compose import hints
from agent_adapter.compose.blanks import NAME, Blank
from agent_adapter.compose.formats import FORMATS, Format, detect, escape
from agent_adapter.core.errors import FlowDefinitionError

_SLOT = re.compile(r"(\\?)\{\{(.*?)\}\}", re.DOTALL)
REFERENCES = "references"
RESERVED = ("aa_begin", "aa_editor", "aa_verify", REFERENCES)


def slots(text: str) -> tuple[list[hints.Hint], list[str]]:
    """The slots in `text` in order, as hints, and the slots whose name is not a name.

    `\\{{` writes literal braces.
    """
    found: list[hints.Hint] = []
    bad: list[str] = []
    for match in _SLOT.finditer(text):
        if match.group(1):
            continue
        hint = hints.parse(match.group(2))
        if NAME.match(hint.name):
            found.append(hint)
        else:
            bad.append(match.group(0))
    return found, bad


def names(text: str) -> list[str]:
    return [hint.name for hint in slots(text)[0]]


def fill(text: str, value_of: Callable[[str], str]) -> str:
    """`text` with every slot replaced by `value_of(name)` and `\\{{` unescaped."""

    def swap(match: re.Match[str]) -> str:
        if match.group(1):
            return match.group(0)[1:]
        name = hints.parse(match.group(2)).name
        return value_of(name) if NAME.match(name) else match.group(0)

    return _SLOT.sub(swap, text)


class Template:
    """A document (Markdown, HTML, XML or text) with blanks.

    A slot with a hint (`{{ summary | "two sentences" }}`) or a `Blank` of that name is a
    blank; `{{ references }}` marks where the References list goes; any other name is an
    input given to `compose`. A blank's prompt may use other blanks: those fill first, and
    blanks that do not depend on each other fill in parallel. `format` is detected from
    the text when not given.

    Raises:
        FlowDefinitionError: every problem found (bad names, duplicates, cycles), together.
    """

    def __init__(
        self,
        text: str,
        blanks: Sequence[Blank] = (),
        *,
        name: str = "document",
        instructions: str = "",
        format: Format | None = None,
    ) -> None:
        self.text = text
        self.name = name
        self.instructions = instructions
        self.format: Format = format or detect(text)
        self.blanks: dict[str, Blank] = {}
        problems: list[str] = []
        if not NAME.match(name):
            problems.append(f"template name {name!r} must be letters, digits or _")
        if self.format not in FORMATS:
            problems.append(f"format must be one of {list(FORMATS)}, got {self.format!r}")
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
        for hint in in_text:
            problems.extend(hint.problems)
            if hint.marked and hint.name not in self.blanks:
                self.blanks[hint.name] = _from_hint(hint)
                problems.extend(self.blanks[hint.name].problems())
        self.deps: dict[str, tuple[str, ...]] = {}
        used = {hint.name for hint in in_text}
        for blank in self.blanks.values():
            found, bad = slots(blank.prompt)
            problems.extend(f"{slot} in blank {blank.name!r} is not a name" for slot in bad)
            listed = [h.name for h in found]
            self.deps[blank.name] = tuple(dict.fromkeys(n for n in listed if n in self.blanks))
            used.update(listed)
        problems.extend(
            f"blank {name!r} is not used in the template or any prompt"
            for name in self.blanks
            if name not in used
        )
        every = [h.name for h in in_text] + [
            n for b in self.blanks.values() for n in names(b.prompt)
        ]
        self.inputs: tuple[str, ...] = tuple(
            dict.fromkeys(n for n in every if n not in self.blanks and n != REFERENCES)
        )
        self.depth: dict[str, int] = {}
        if not problems:
            problems.extend(self._order())
        if problems:
            raise FlowDefinitionError(problems, flow=name)

    @classmethod
    def load(cls, path: str | Path, blanks: Sequence[Blank] = (), **options: Any) -> Template:
        """A template from a file; its format comes from the extension, its name from the stem."""
        file = Path(path)
        text = file.read_text(encoding="utf-8")
        options.setdefault("format", detect(text, file))
        stem = re.sub(r"\W", "_", file.stem).strip("_")
        options.setdefault("name", stem if NAME.match(stem) else "document")
        return cls(text, blanks, **options)

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

    def reading_order(self) -> list[str]:
        """Blank names in the order they first appear in the text, then helper blanks."""
        shown = [n for n in dict.fromkeys(names(self.text)) if n in self.blanks]
        return shown + [n for n in self.blanks if n not in shown]

    def missing_inputs(self, inputs: Mapping[str, Any]) -> list[str]:
        """The inputs the template uses that `inputs` does not give."""
        return [name for name in self.inputs if name not in inputs]

    def has_references(self) -> bool:
        return REFERENCES in names(self.text)

    def render(self, rendered: Mapping[str, str], inputs: Mapping[str, Any]) -> str:
        """The document with blanks (already drawn) and inputs (escaped) filled in."""

        def value_of(name: str) -> str:
            if name in rendered:
                return rendered[name]
            return escape(str(inputs.get(name, "")), self.format)

        return fill(self.text, value_of)

    def __repr__(self) -> str:
        return f"Template({self.name!r}, format={self.format!r}, blanks={list(self.blanks)})"


def _from_hint(hint: hints.Hint) -> Blank:
    return Blank(
        hint.name,
        hint.prompt(),
        hint.kind,  # type: ignore[arg-type]
        max_words=hint.max_words,
        required=hint.required,
        columns=hint.columns,
        direction=hint.direction,  # type: ignore[arg-type]
    )
