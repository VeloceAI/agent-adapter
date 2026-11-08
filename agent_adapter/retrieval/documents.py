"""`Document` and `Source`, the filter type, and the checks every retriever shares."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from agent_adapter.core.errors import ToolUsageError

MAX_K = 100
MAX_QUERY_CHARS = 8_000
MAX_FILTERS = 32

FilterValue = str | int | float | bool
Filters = Mapping[str, FilterValue]

# Metadata keys a backend commonly uses for a document's location and title, in order.
URI_KEYS = ("uri", "url", "source", "file_path", "path")
TITLE_KEYS = ("title", "file_name", "name")


class Source(BaseModel):
    """Where a document came from, for citations: a link or path and a title."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = ""
    title: str = ""


class Document(BaseModel):
    """One retrieved passage: a stable id, its text, a score (higher is better) and its source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    text: str
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None
    source: Source = Source()
    metadata: dict[str, Any] = Field(default_factory=dict)

    def cited(
        self, *, include_metadata: bool = False, max_chars: int | None = None
    ) -> dict[str, Any]:
        """The JSON shape a model sees: id, text, score and source (metadata only if asked)."""
        text = self.text if max_chars is None else self.text[:max_chars]
        shown: dict[str, Any] = {"id": self.id, "text": text, "score": self.score}
        shown["source"] = self.source.model_dump()
        if include_metadata:
            shown["metadata"] = dict(self.metadata)
        return shown


def document_id(text: str, uri: str = "") -> str:
    """A stable id for a document its backend gave none: a digest of its source and text."""
    digest = hashlib.sha256(f"{uri}\x00{text}".encode()).hexdigest()
    return f"doc_{digest[:16]}"


def source_from(metadata: Mapping[str, Any]) -> Source:
    """A `Source` from the first non-empty string under `URI_KEYS` and `TITLE_KEYS`."""
    return Source(uri=_first(metadata, URI_KEYS), title=_first(metadata, TITLE_KEYS))


def score_of(value: object) -> float | None:
    """`value` as a finite float score, else None (backends without scores, NaN, bools)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _first(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def as_document(item: Any) -> Document:
    """A `Document` from a Document, a str, or a mapping with `text` (plus the other fields).

    Raises:
        ToolUsageError: anything else, or a mapping that is not a valid document.
    """
    if isinstance(item, Document):
        return item
    if isinstance(item, str):
        return Document(id=document_id(item), text=item)
    if isinstance(item, Mapping):
        fields = dict(item)
        text = fields.get("text")
        if not isinstance(text, str):
            raise ToolUsageError("a retrieved mapping needs a str 'text'")
        metadata = fields.get("metadata") or {}
        try:
            if fields.get("source") is None:
                fields["source"] = source_from(metadata if isinstance(metadata, Mapping) else {})
            source = Source.model_validate(fields["source"])
            fields["source"] = source
            fields["id"] = fields.get("id") or document_id(text, source.uri)
            return Document.model_validate(fields)
        except ValueError as exc:
            raise ToolUsageError("a retrieved mapping is not a valid document") from exc
    raise ToolUsageError(f"a retriever must return documents, got {type(item).__name__}")


def check_query(query: object) -> str:
    """`query`, stripped.

    Raises:
        ToolUsageError: not a str, empty, or longer than `MAX_QUERY_CHARS`.
    """
    if not isinstance(query, str):
        raise ToolUsageError(f"query must be a str, got {type(query).__name__}")
    query = query.strip()
    if not query:
        raise ToolUsageError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolUsageError(f"query must be at most {MAX_QUERY_CHARS} characters")
    return query


def check_k(k: object) -> int:
    """`k` when it is an int from 1 to `MAX_K`.

    Raises:
        ToolUsageError: anything else.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ToolUsageError(f"k must be an int from 1 to {MAX_K}, got {k!r}")
    return k


def check_filters(filters: object) -> dict[str, FilterValue]:
    """`filters` as a dict of metadata key to the exact value it must have.

    Raises:
        ToolUsageError: not a mapping of non-empty str keys to str, int, finite float or bool.
    """
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise ToolUsageError(f"filters must be a mapping, got {type(filters).__name__}")
    if len(filters) > MAX_FILTERS:
        raise ToolUsageError(f"at most {MAX_FILTERS} filters")
    clean: dict[str, FilterValue] = {}
    for key, value in filters.items():
        if not isinstance(key, str) or not key:
            raise ToolUsageError("filter keys must be non-empty str")
        if not isinstance(value, str | int | float) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ToolUsageError(f"filter {key!r} must be a str, int, finite float or bool")
        clean[key] = value
    return clean


def matches(document: Document, filters: Mapping[str, FilterValue]) -> bool:
    """True when every filter key is in the metadata with the same value (bools match bools)."""
    for key, wanted in filters.items():
        if key not in document.metadata:
            return False
        found = document.metadata[key]
        if isinstance(found, bool) != isinstance(wanted, bool) or found != wanted:
            return False
    return True
