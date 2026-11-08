import math

import pytest
from pydantic import ValidationError

from agent_adapter.errors import ToolUsageError
from agent_adapter.retrieval import MAX_K, Document, Source
from agent_adapter.retrieval.documents import (
    MAX_FILTERS,
    MAX_QUERY_CHARS,
    as_document,
    check_filters,
    check_k,
    check_query,
    document_id,
    matches,
    score_of,
    source_from,
)


def test_document_cited_shows_id_text_score_and_source_only() -> None:
    document = Document(id="d1", text="abcdef", score=0.5, source=Source(uri="u", title="t"))
    document = document.model_copy(update={"metadata": {"scope": "north"}})
    assert document.cited() == {
        "id": "d1",
        "text": "abcdef",
        "score": 0.5,
        "source": {"uri": "u", "title": "t"},
    }
    shown = document.cited(include_metadata=True, max_chars=3)
    assert shown["text"] == "abc"
    assert shown["metadata"] == {"scope": "north"}


@pytest.mark.parametrize(
    "bad", [{"id": "", "text": "x"}, {"id": "a", "text": "x", "score": math.nan}]
)
def test_document_rejects_empty_id_and_nan_score(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Document.model_validate(bad)


def test_document_id_is_stable_and_depends_on_source() -> None:
    assert document_id("hello") == document_id("hello")
    assert document_id("hello", "a") != document_id("hello", "b")
    assert document_id("hello").startswith("doc_")


def test_source_from_takes_first_non_empty_string() -> None:
    assert source_from({"url": "", "source": "s.md", "file_name": "S", "title": 3}) == Source(
        uri="s.md", title="S"
    )
    assert source_from({}) == Source()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), (0.25, 0.25), (True, None), (math.inf, None), ("0.5", None), (None, None)],
)
def test_score_of_keeps_finite_numbers_only(value: object, expected: float | None) -> None:
    assert score_of(value) == expected


def test_as_document_accepts_documents_strings_and_mappings() -> None:
    same = Document(id="a", text="t")
    assert as_document(same) is same
    assert as_document("plain").id == document_id("plain")
    mapped = as_document({"text": "t", "metadata": {"url": "https://x", "title": "X"}})
    assert mapped.source == Source(uri="https://x", title="X")
    assert mapped.id == document_id("t", "https://x")
    given = as_document({"id": "k", "text": "t", "source": {"uri": "u"}, "score": 2})
    assert (given.id, given.source.uri, given.score) == ("k", "u", 2.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"title": "no text"},
        {"text": "t", "source": 3},
        {"text": "t", "metadata": "x"},
        {"text": "t", "extra": 1},
        42,
        b"bytes",
    ],
)
def test_as_document_rejects_other_shapes(bad: object) -> None:
    with pytest.raises(ToolUsageError):
        as_document(bad)


def test_check_query_strips_and_bounds() -> None:
    assert check_query("  refund  ") == "refund"
    for bad in (3, "", "   ", "x" * (MAX_QUERY_CHARS + 1)):
        with pytest.raises(ToolUsageError):
            check_query(bad)


def test_check_k_bounds() -> None:
    assert check_k(1) == 1
    assert check_k(MAX_K) == MAX_K
    for bad in (0, MAX_K + 1, True, 2.0, "3"):
        with pytest.raises(ToolUsageError):
            check_k(bad)


def test_check_filters_accepts_scalars_only() -> None:
    assert check_filters(None) == {}
    assert check_filters({"a": "x", "b": 1, "c": 0.5, "d": False}) == {
        "a": "x",
        "b": 1,
        "c": 0.5,
        "d": False,
    }
    bad_cases: list[object] = [
        ["a"],
        {"": "x"},
        {1: "x"},
        {"a": None},
        {"a": ["x"]},
        {"a": math.nan},
        {str(i): i for i in range(MAX_FILTERS + 1)},
    ]
    for bad in bad_cases:
        with pytest.raises(ToolUsageError):
            check_filters(bad)


def test_matches_needs_every_key_with_the_same_value_and_kind() -> None:
    document = Document(id="a", text="t", metadata={"scope": "north", "level": 1, "open": True})
    assert matches(document, {})
    assert matches(document, {"scope": "north", "level": 1.0})
    assert not matches(document, {"scope": "south"})
    assert not matches(document, {"missing": "x"})
    assert not matches(document, {"level": True})  # 1 is not True here
    assert not matches(document, {"open": 1})
    assert not matches(document, {"level": "1"})
