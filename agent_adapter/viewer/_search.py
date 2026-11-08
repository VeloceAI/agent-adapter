"""Run search: names folded line by line per segment, and a query matched against them."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.tracing.sinks import CONTENT_KEYS

MAX_TEXT = 64 * 1024  # characters of captured content kept per segment for search
MAX_NAMES = 2_000  # distinct names kept per segment
MAX_QUERY = 200  # characters of one query
_NAME_KEYS = ("tool", "flow", "node", "provider", "model", "error_kind", "stage", "retriever")


@dataclass
class SearchFold:
    """The searchable names of one segment, and its captured content (when it was recorded)."""

    names: set[str] = field(default_factory=set)
    text: list[str] = field(default_factory=list)
    text_chars: int = 0

    def add(self, record: Mapping[str, Any]) -> None:
        """Fold one parsed line in."""
        if record.get("kind") == "meta":
            self._name(record.get("run_id"))
            return
        if record.get("kind") != "event":
            return
        self._name(record.get("type"))
        data = record.get("data")
        if not isinstance(data, Mapping):
            return
        for key in _NAME_KEYS:
            self._name(data.get(key))
        for key in CONTENT_KEYS & data.keys():
            self._content(data[key])

    def copy(self) -> SearchFold:
        """A snapshot that later `add` calls do not change."""
        return SearchFold(set(self.names), list(self.text), self.text_chars)

    def _name(self, value: Any) -> None:
        if isinstance(value, str) and value and len(self.names) < MAX_NAMES:
            self.names.add(value.lower())

    def _content(self, value: Any) -> None:
        room = MAX_TEXT - self.text_chars
        if room <= 0 or value is None:
            return
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        kept = text[:room].lower()
        self.text.append(kept)
        self.text_chars += len(kept)


@dataclass(frozen=True)
class Query:
    """A parsed search: words that must all match, plus `tag:` and `status:` filters."""

    words: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.words or self.tags or self.statuses)


def parse_query(text: str, tag: str = "") -> Query:
    """`text` split into words; `tag:x` and `status:x` words filter; `tag` adds one more tag."""
    words: list[str] = []
    tags = [" ".join(tag.split()).lower()] if tag.strip() else []
    statuses: list[str] = []
    for word in text[:MAX_QUERY].lower().split():
        if word.startswith("tag:") and len(word) > 4:
            tags.append(word[4:])
        elif word.startswith("status:") and len(word) > 7:
            statuses.append(word[7:])
        else:
            words.append(word)
    return Query(tuple(words), tuple(tags), tuple(statuses))


def matches(
    summary: Mapping[str, Any],
    folds: Iterable[SearchFold],
    tags: Sequence[str],
    query: Query,
) -> bool:
    """True when the run passes every filter and holds every word."""
    if query.statuses and str(summary.get("status", "")) not in query.statuses:
        return False
    if any(tag not in tags for tag in query.tags):
        return False
    if not query.words:
        return True
    folds = list(folds)
    fields = [str(summary.get(key, "")).lower() for key in ("key", "run_id", "status", "framework")]
    fields += [str(flow).lower() for flow in summary.get("flows") or ()]
    fields += list(tags)
    for fold in folds:
        fields += fold.names
    texts = [text for fold in folds for text in fold.text]
    return all(_found(word, fields, texts) for word in query.words)


def _found(word: str, fields: list[str], texts: list[str]) -> bool:
    return any(word in field for field in fields) or any(word in text for text in texts)
