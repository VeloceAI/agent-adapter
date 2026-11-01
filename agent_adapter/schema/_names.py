"""Unique names: a numbered variant per clash, found in amortised O(1)."""

from __future__ import annotations

# Numbered variants of capped names share one counter per stem of `max_len - _SUFFIX_ROOM`
# characters, so long names that agree on that stem do not re-probe each other.
_SUFFIX_ROOM = 8


def unique(
    name: str, taken: set[str], counters: dict[str, int], *, max_len: int | None = None
) -> str:
    """`name`, else the first free `{name}_N` from this stem's counter (`N` from 2).

    `taken` only grows, so every variant below the counter is still taken; the counter is
    left at the variant returned, so it is offered again if the caller does not take it.
    With `max_len`, a variant is cut to fit (`{name[:max_len - len("_N")]}_N`).
    """
    if name not in taken:
        return name
    stem = name if max_len is None else name[: max_len - _SUFFIX_ROOM]
    n = counters.get(stem, 2)
    while True:
        suffix = f"_{n}"
        head = name if max_len is None else name[: max_len - len(suffix)]
        candidate = f"{head}{suffix}"
        if candidate not in taken:
            counters[stem] = n
            return candidate
        n += 1
