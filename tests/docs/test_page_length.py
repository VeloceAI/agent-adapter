"""Every published docs page, the README and the codebase map stay short enough to read."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MAX_LINES = 150
# The README is the project's front page and CODEBASE.md the map for contributors: each is
# one document a reader goes through end to end, so they get a larger budget than a page.
LONG_PAGES = {"README.md": 350, "CODEBASE.md": 350}
PAGES = [ROOT / name for name in LONG_PAGES] + [
    page for page in sorted(DOCS.rglob("*.md")) if "superpowers" not in page.relative_to(DOCS).parts
]


def reading_lines(text: str) -> int:
    """Lines a reader sees: front matter (search title and description) is not on the page."""
    lines = text.splitlines()
    if lines and lines[0] == "---" and "---" in lines[1:]:
        lines = lines[lines.index("---", 1) + 1 :]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return len(lines)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.relative_to(ROOT).as_posix())
def test_page_stays_within_its_line_budget(page: Path) -> None:
    budget = LONG_PAGES.get(page.relative_to(ROOT).as_posix(), MAX_LINES)
    lines = reading_lines(page.read_text(encoding="utf-8"))
    assert lines <= budget, f"{lines} lines (budget {budget}): split the page or tighten it"
