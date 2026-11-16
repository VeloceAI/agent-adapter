"""Run the media how-to example offline and check its page."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "how-to" / "media-in-tool-results.md"


def test_example_prints_what_the_page_shows(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(ROOT / "examples" / "howto_media.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    page = PAGE.read_text(encoding="utf-8")
    for line in lines[:4]:
        assert line in page
    assert lines[4] == "screenshot of https://example.com: image/png 'page.png' (2 KB)"


def test_page_is_short_and_in_the_nav() -> None:
    assert len(PAGE.read_text(encoding="utf-8").splitlines()) <= 150
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "how-to/media-in-tool-results.md" in nav
