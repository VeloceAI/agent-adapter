"""Board entries and claims: the value types every store reads and writes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_adapter.core.errors import ToolUsageError

Kind = Literal["status", "finding", "claim", "handoff"]
KINDS: tuple[str, ...] = ("status", "finding", "claim", "handoff")
MAX_TEXT = 2_000  # characters of one entry's text; longer text is cut
MAX_KEY = 200
MAX_BY = 100
DEFAULT_MAX_ENTRIES = 500
DEFAULT_MAX_BYTES = 256 * 1024


@dataclass(frozen=True)
class Entry:
    """One board post: who wrote it, what kind, an optional key, the text and JSON data.

    `seq` is the entry's place on its board (1, 2, ...), set by the store; `read(since=)`
    and `watch()` use it.
    """

    id: str
    at: float
    by: str
    kind: str
    text: str
    key: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The entry as plain JSON data (a snapshot row)."""
        return {
            "id": self.id,
            "at": self.at,
            "by": self.by,
            "kind": self.kind,
            "key": self.key,
            "text": self.text,
            "data": dict(self.data),
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Entry:
        """An entry from `to_dict` data.

        Raises:
            ValueError: a missing field or one of the wrong type.
        """
        if not isinstance(row, Mapping):
            raise ValueError("a board entry must be a mapping")
        text = {name: row.get(name, "") for name in ("id", "by", "kind", "key", "text")}
        if not all(isinstance(value, str) for value in text.values()) or not text["id"]:
            raise ValueError("a board entry needs str id, by, kind, key and text")
        if text["kind"] not in KINDS:
            raise ValueError(f"unknown board entry kind {text['kind']!r}")
        at, seq, data = row.get("at"), row.get("seq", 0), row.get("data") or {}
        if isinstance(at, bool) or not isinstance(at, int | float) or not math.isfinite(at):
            raise ValueError("a board entry needs a finite number `at`")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError("a board entry's seq must be an int >= 0")
        if not isinstance(data, Mapping):
            raise ValueError("a board entry's data must be a mapping")
        return cls(seq=seq, at=float(at), data=dict(data), **text)

    def size(self) -> int:
        """Bytes this entry counts against the board's `max_bytes`."""
        return len(dumps(self))


@dataclass(frozen=True)
class Claim:
    """The outcome of `Board.claim`: who holds the key, and whether that is the caller."""

    key: str
    holder: str
    won: bool


def dumps(entry: Entry) -> str:
    """The entry as compact JSON text (how stores keep it)."""
    return json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":"))


def loads(text: str | bytes) -> Entry:
    """An entry from `dumps` text.

    Raises:
        ValueError: not an entry's JSON.
    """
    return Entry.from_dict(json.loads(text))


def check_kind(kind: object) -> str:
    """Refuse an unknown entry kind.

    Raises:
        ToolUsageError: `kind` is not one of KINDS.
    """
    if kind not in KINDS:
        raise ToolUsageError(f"board entry kind must be one of {list(KINDS)}, got {kind!r}")
    return str(kind)


def check_short(name: str, value: object, limit: int, *, empty: bool = True) -> str:
    """Refuse a non-str, too long or (unless allowed) empty key or author.

    Raises:
        ToolUsageError: the value breaks one of those rules.
    """
    if not isinstance(value, str) or len(value) > limit or (not empty and not value):
        need = "a" if empty else "a non-empty"
        raise ToolUsageError(f"board {name} must be {need} str of at most {limit} chars")
    return value


def check_data(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """`data` as a plain dict that JSON can hold.

    Raises:
        ToolUsageError: not a mapping, or a value JSON cannot encode.
    """
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ToolUsageError(f"board entry data must be a mapping, got {type(data).__name__}")
    try:  # the round trip leaves only plain JSON types (tuples become lists)
        plain: dict[str, Any] = json.loads(json.dumps(dict(data), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ToolUsageError(f"board entry data must be JSON data: {exc}") from exc
    return plain


def cut(text: str, limit: int = MAX_TEXT) -> str:
    """`text` cut to `limit` characters, marked when it was cut."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def check_caps(max_entries: object, max_bytes: object) -> None:
    """Refuse caps that are not positive ints.

    Raises:
        ToolUsageError: either cap is not an int >= 1.
    """
    for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ToolUsageError(f"board {name} must be an int >= 1, got {value!r}")
