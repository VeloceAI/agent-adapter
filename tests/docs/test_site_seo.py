"""The docs site has a `site_url` (so the sitemap fills) and an llms.txt whose links all resolve."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
LLMS = ROOT / "docs" / "llms.txt"
LINK = re.compile(r"\]\((https?://[^)]+)\)")


def site_url() -> str:
    """The `site_url` set in mkdocs.yml."""
    match = re.search(r"^site_url:\s*(\S+)\s*$", CONFIG, re.M)
    assert match, "mkdocs.yml needs a site_url, or sitemap.xml stays empty"
    return match.group(1)


def _page(url: str) -> str:
    path = url.removeprefix(site_url()).strip("/")
    for candidate in (f"{path}.md", f"{path}/index.md") if path else ("index.md",):
        if (ROOT / "docs" / candidate).is_file():
            return candidate
    raise AssertionError(f"llms.txt links {url}, which is no docs page")


def test_site_url_is_absolute_with_a_trailing_slash() -> None:
    assert re.fullmatch(r"https://[^/\s]+/(\S*/)?", site_url())


def test_llms_txt_starts_with_a_title_and_a_summary() -> None:
    lines = LLMS.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Agent Adapter"
    assert lines[2].startswith("> ")


def test_every_llms_txt_link_is_a_page_in_the_nav() -> None:
    links = LINK.findall(LLMS.read_text(encoding="utf-8"))
    assert links
    for url in links:
        assert url.startswith(site_url()), url
        assert f" {_page(url)}\n" in CONFIG, f"{url}: page is not in the mkdocs.yml nav"
