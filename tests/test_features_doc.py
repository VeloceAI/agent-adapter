"""The Features page: every row names a feature and links a guide that exists."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
FEATURES = DOCS / "features.md"
HEADER = ["Feature", "Guide"]
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)\s]*)?\)")
CODE = re.compile(r"`[^`]*`")


def _table_lines() -> list[list[str]]:
    rows = []
    for line in FEATURES.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        # A `|` inside a code span is not a cell separator.
        plain = CODE.sub(lambda m: m.group().replace("|", "/"), line)
        rows.append([cell.strip() for cell in plain.strip().strip("|").split("|")])
    return rows


def _feature_rows() -> list[list[str]]:
    return [
        cells
        for cells in _table_lines()
        if cells != HEADER and not all(set(cell) <= set("-:") for cell in cells)
    ]


def test_features_doc_lists_rows_in_every_section() -> None:
    assert len(_feature_rows()) >= 80


def test_features_doc_every_table_has_feature_and_guide_columns() -> None:
    malformed = [cells for cells in _table_lines() if len(cells) != len(HEADER)]
    assert malformed == []


def test_features_doc_has_no_status_or_internal_columns() -> None:
    text = FEATURES.read_text(encoding="utf-8")
    for marker in ("status-planned", "status-ok", "| Spec |", "| Status |", "§"):
        assert marker not in text, marker


def test_features_doc_every_row_links_a_doc_page() -> None:
    unlinked = [row[0] for row in _feature_rows() if not LINK.search(row[1])]
    assert unlinked == []


@pytest.mark.parametrize(
    "target", sorted(set(LINK.findall(FEATURES.read_text(encoding="utf-8")))), ids=str
)
def test_features_doc_link_target_exists(target: str) -> None:
    assert (FEATURES.parent / target).resolve().is_file(), target
