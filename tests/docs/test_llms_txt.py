"""docs/llms.txt is generated from the mkdocs.yml nav, so it lists every page and cannot drift.

Regenerate it with `python -m tests.docs.test_llms_txt`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CONFIG = ROOT / "mkdocs.yml"
LLMS = DOCS / "llms.txt"
SUMMARY = (
    "> A framework-independent Python SDK (3.11+). Write a tool once as a Python function and"
    " use it with any model API (OpenAI, Gemini, Anthropic), any agent framework (LangGraph,"
    " OpenAI Agents SDK, Pydantic AI, CrewAI, ...), MCP, A2A, AG-UI or HTTP. One executor adds"
    " validation, hooks, human approval, budgets, caching and tracing."
)
IMPORT_RULES = (
    "Import rules: everyday names come from `agent_adapter`; every other error from"
    " `agent_adapter.errors`; everything else from its subpackage (`agent_adapter.exec`,"
    " `.hooks`, `.approval`, ...); framework exports from"
    " `agent_adapter.integrations.<framework>`. Never import from `agent_adapter.core` or a"
    " `_private` module."
)
HEADER = f"# Agent Adapter\n\n{SUMMARY}\n\n{IMPORT_RULES}\n"
# scripts/gen_agent_guide.py writes it: install, recipes and every public name in one file.
FULL_GUIDE = "Full guide for AI coding agents (install, recipes, every public name):"
SKIP = re.compile(r"^(```|\||:material|<|!?\[|---|-\s|\*|=+|#|>|!!!|\?\?\?|\d+\.\s)")
SENTENCE_END = re.compile(r"(?<=\.)\s+(?=[A-Z`*(])")
SNIPPET = re.compile(r'^--8<-- "([^"]+)"$')
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MAX_SUMMARY = 160


def _config() -> tuple[str, list[Any]]:
    """`site_url` and the nav; the nav is parsed alone because the rest uses Python tags."""
    text = CONFIG.read_text(encoding="utf-8")
    site = re.search(r"^site_url:\s*(\S+)\s*$", text, re.M)
    assert site, "mkdocs.yml needs a site_url"
    nav = yaml.safe_load(text[text.index("\nnav:\n") :])["nav"]
    return site.group(1), nav


def _url(site: str, page: str) -> str:
    path = page.removesuffix(".md")
    path = path.removesuffix("index") if path == "index" or path.endswith("/index") else path + "/"
    return site + path


def _summary(page: str) -> tuple[str, str]:
    """The page's H1 and the first sentence of its first prose paragraph."""
    lines = (DOCS / page).read_text(encoding="utf-8").splitlines()
    if lines and (included := SNIPPET.match(lines[0])):
        lines = (ROOT / included.group(1)).read_text(encoding="utf-8").splitlines()
    heading = next(line[2:].strip() for line in lines if line.startswith("# "))
    blocks: list[list[str]] = [[]]
    fenced = False
    for line in lines[lines.index(f"# {heading}") + 1 :]:
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif fenced:
            continue
        elif stripped:
            blocks[-1].append(stripped)
        elif blocks[-1]:
            blocks.append([])
    prose = next((b for b in blocks if b and not SKIP.match(b[0])), [])
    text = LINK.sub(r"\1", " ".join(prose))
    sentence = SENTENCE_END.split(text, maxsplit=1)[0].rstrip(".")
    if len(sentence) > MAX_SUMMARY:
        sentence = sentence[: MAX_SUMMARY - 3].rsplit(" ", 1)[0] + "..."
    return heading, sentence


def _entries(items: list[Any]) -> list[tuple[str | None, str]]:
    """Flatten a nav subtree into `(nav title or None, page)` in nav order."""
    found: list[tuple[str | None, str]] = []
    for item in items:
        if isinstance(item, str):
            found.append((None, item))
            continue
        ((title, value),) = item.items()
        if isinstance(value, str):
            found.append((title, value))
        else:
            found.extend(_entries(value))
    return found


def render() -> str:
    """llms.txt exactly as it should be on disk."""
    site, nav = _config()
    out = [HEADER, f"{FULL_GUIDE} <{site}llms-full.txt>\n"]
    for item in nav:
        ((section, value),) = item.items()
        out.append(f"## {section}\n")
        for title, page in _entries([{section: value}] if isinstance(value, str) else value):
            heading, sentence = _summary(page)
            line = f"- [{title or heading}]({_url(site, page)})"
            out.append(f"{line}: {sentence}" if sentence else line)
        out.append("")
    return "\n".join(out)


def _nav_pages() -> set[str]:
    return {page for _, page in _entries(_config()[1])}


def test_llms_txt_matches_the_nav() -> None:
    assert LLMS.read_text(encoding="utf-8") == render(), (
        "regenerate: python -m tests.docs.test_llms_txt"
    )


def test_llms_txt_links_every_published_page() -> None:
    published = {
        p.relative_to(DOCS).as_posix()
        for p in DOCS.rglob("*.md")
        if "superpowers" not in p.relative_to(DOCS).parts
    }
    assert published == _nav_pages(), "every docs page must be in the mkdocs.yml nav"
    text = LLMS.read_text(encoding="utf-8")
    site = _config()[0]
    for page in published:
        assert f"]({_url(site, page)})" in text, page


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    LLMS.write_text(render(), encoding="utf-8", newline="\n")
