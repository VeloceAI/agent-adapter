"""Assemble the two guide tiers and write (or check) their copies."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.agent_guide import api_index, install, recipes, sections

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PACKAGED = ROOT / "agent_adapter" / "_guide"
# Concise tier: the repo root (AGENTS.md convention) and the default `agent-adapter guide`.
CONCISE = (ROOT / "AGENTS.md", PACKAGED / "AGENTS.md")
# Full tier (adds the API index): the docs site and `agent-adapter guide --full`.
FULL = (ROOT / "docs" / "llms-full.txt", PACKAGED / "llms-full.txt")


def _parts(*, full: bool) -> list[str]:
    """Safety rules come early so an agent that reads only the top still gets them."""
    return [
        sections.title(full=full),
        sections.what_it_is(),
        install.render(full=full),
        sections.import_rules(),
        sections.safety(),
        recipes.render(full=full),
        sections.errors(full=full),
        sections.cli(),
        api_index.render() if full else sections.full_api_pointer(),
    ]


def render_concise() -> str:
    """`AGENTS.md`: everything but the API index."""
    return "\n".join(part.rstrip("\n") + "\n" for part in _parts(full=False))


def render_full() -> str:
    """`llms-full.txt`: the concise guide plus the troubleshooting tables and the API index."""
    return "\n".join(part.rstrip("\n") + "\n" for part in _parts(full=True))


def outputs() -> list[tuple[Path, Callable[[], str]]]:
    """Every output file and the function that renders it."""
    return [(p, render_concise) for p in CONCISE] + [(p, render_full) for p in FULL]


def _changed() -> list[tuple[Path, str]]:
    texts = {render_concise: render_concise(), render_full: render_full()}
    return [
        (path, texts[fn])
        for path, fn in outputs()
        if not path.is_file() or path.read_text(encoding="utf-8") != texts[fn]
    ]


def stale() -> list[Path]:
    """The outputs whose content differs from what the generator renders."""
    return [path for path, _ in _changed()]


def write() -> list[Path]:
    """Write every output that changed; return them."""
    changed = _changed()
    for path, text in changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        logger.info("wrote %s", path.relative_to(ROOT).as_posix())
    return [path for path, _ in changed]


def main(argv: Sequence[str] | None = None) -> int:
    """`--check` exits 1 when an output is stale; otherwise write them all."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--check"]:
        old = stale()
        for path in old:
            logger.error("stale: %s", path.relative_to(ROOT).as_posix())
        return 1 if old else 0
    if args:
        logger.error("usage: python scripts/gen_agent_guide.py [--check]")
        return 2
    write()
    return 0
