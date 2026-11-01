"""Tool and property names: sanitised, unique, numbered on a clash in amortised O(1)."""

from __future__ import annotations

import re

from agent_adapter.schema._names import unique as _unique

MAX_NAME = 64  # tool names, and numbered variants of any name


def sanitise(name: str) -> str:
    """`name` as `[A-Za-z0-9_]`, never empty and never starting with a digit."""
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not clean or clean[0].isdigit():
        clean = f"p_{clean}"
    return clean


def tool_name(prefix: str, key: str) -> str:
    """`prefix + key` sanitised for a tool name (at most `MAX_NAME` characters)."""
    clean = re.sub(r"[^0-9A-Za-z_]", "_", f"{prefix}{key}").strip("_")
    if not clean or clean[0].isdigit():
        clean = f"op_{clean}"
    return clean[:MAX_NAME]


def unique(name: str, taken: set[str], counters: dict[str, int]) -> str:
    """`schema._names.unique` capped at `MAX_NAME` characters."""
    return _unique(name, taken, counters, max_len=MAX_NAME)
