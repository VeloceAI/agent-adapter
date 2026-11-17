"""Blanks written inside the template: `{{ name | "hint" | table(A, B) | words(40) }}`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CALL = re.compile(r"^([a-z]+)\s*(?:\((.*)\))?$", re.DOTALL)
_KINDS = ("text", "list", "number", "table", "diagram")


@dataclass(frozen=True)
class Hint:
    """What a slot says about its blank; `marked` is False for a bare `{{ name }}`."""

    name: str
    marked: bool = False
    hint: str = ""
    kind: str = "text"
    columns: tuple[str, ...] = ()
    direction: str = "TD"
    max_words: int | None = None
    required: bool = True
    uses: tuple[str, ...] = ()
    problems: tuple[str, ...] = field(default=())

    def prompt(self) -> str:
        """The blank's prompt: the hint (or a default), then the blanks it uses."""
        text = self.hint or f"Write the {self.name.replace('_', ' ')}."
        if self.uses:
            text += "\n\nBuild on:\n" + "\n\n".join(f"{{{{ {name} }}}}" for name in self.uses)
        return text


def parse(inner: str) -> Hint:
    """A slot's inside, as a `Hint` (its problems listed, never raised)."""
    parts = _split(inner)
    name = parts[0].strip()
    if len(parts) == 1:
        return Hint(name)
    found: dict[str, object] = {}
    problems: list[str] = []
    for raw in parts[1:]:
        part = raw.strip()
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
            found["hint"] = part[1:-1].strip()
            continue
        match = _CALL.match(part)
        if match is None:
            problems.append(f"{{{{ {name} }}}}: cannot read {part!r}")
            continue
        word, args = match.group(1), [a.strip() for a in (match.group(2) or "").split(",")]
        args = [a for a in args if a]
        if word in _KINDS:
            found["kind"] = word
            if word == "table":
                found["columns"] = tuple(args)
            elif word == "diagram" and args:
                found["direction"] = args[0].upper()
        elif word == "words" and len(args) == 1 and args[0].isdigit():
            found["max_words"] = int(args[0])
        elif word == "optional" and not args:
            found["required"] = False
        elif word == "uses" and args:
            found["uses"] = tuple(args)
        else:
            problems.append(f"{{{{ {name} }}}}: unknown option {part!r}")
    return Hint(name, True, problems=tuple(problems), **found)  # type: ignore[arg-type]


def _split(inner: str) -> list[str]:
    """`inner` split on `|` outside quotes."""
    parts: list[str] = []
    current: list[str] = []
    quote = ""
    for char in inner:
        if quote:
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "|":
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts
