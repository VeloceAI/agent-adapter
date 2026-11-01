"""A system prompt's `<reference>` blocks: the only part of it a summarising step may shorten."""

from __future__ import annotations

import re

_BLOCK = re.compile(r"(?is)(<reference>)(.*?)(</reference>)")


def reference_spans(text: str) -> list[tuple[int, int]]:
    """(start, stop) of the inside of each `<reference>...</reference>` block, in order."""
    return [match.span(2) for match in _BLOCK.finditer(text)]


def without_references(text: str) -> str:
    """`text` with every reference block emptied: the instruction lines, verbatim."""
    return _BLOCK.sub(lambda m: m.group(1) + m.group(3), text)


def keeps_instructions(old: str, new: str) -> bool:
    """True when `new` differs from `old` only inside its reference blocks."""
    return len(reference_spans(old)) == len(reference_spans(new)) and without_references(
        old
    ) == without_references(new)
