"""`agent-adapter guide`: print or write the packaged guide for AI coding agents."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TextIO

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

FOLDER = Path(__file__).resolve().parents[1] / "_guide"  # shipped in the wheel
CONCISE, FULL = FOLDER / "AGENTS.md", FOLDER / "llms-full.txt"

# --section name -> the heading it starts at; it runs to the next `## ` heading.
SECTIONS: dict[str, str] = {
    "install": "## Install",
    "safety": "## Safety rules",
    "recipes": "## I want to",
    "errors": "## Common errors",
    "cli": "## CLI commands",
    "api": "## API index",
}


def guide_text(*, full: bool = False) -> str:
    """The packaged guide: concise `AGENTS.md`, or the full text with the API index."""
    return (FULL if full else CONCISE).read_text(encoding="utf-8")


def section(text: str, name: str) -> str:
    """One `## ` section of `text`; ConfigError for an unknown name or a missing heading."""
    if name not in SECTIONS:
        raise ConfigError(f"unknown section {name!r}; choose one of {sorted(SECTIONS)}")
    lines = text.splitlines(keepends=True)
    fenced = False
    start: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
        if fenced or not line.startswith("## "):
            continue
        if start is not None:
            return "".join(lines[start:index]).rstrip() + "\n"
        if line.startswith(SECTIONS[name]):
            start = index
    if start is None:
        raise ConfigError(f"the packaged guide has no {SECTIONS[name]!r} section")
    return "".join(lines[start:]).rstrip() + "\n"


def emit(out: TextIO, text: str) -> None:
    """Write `text` as UTF-8 even when the console or pipe uses a legacy code page."""
    buffer = getattr(out, "buffer", None)
    if buffer is None:  # a StringIO or another text-only stream
        out.write(text)
        return
    out.flush()
    buffer.write(text.encode("utf-8"))
    buffer.flush()


def _write(path: Path, text: str, *, force: bool) -> None:
    """Write `text`; without `force` the file must not exist yet (like `init`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= os.O_TRUNC if force else os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ConfigError(f"not overwriting {path}; pass --force to replace") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(text.encode("utf-8"))


def run_guide(
    out_path: Path | None, part: str | None, out: TextIO, *, full: bool = False, force: bool = False
) -> int:
    """Print the guide (or one section), or write it to `out_path`."""
    text = section(guide_text(full=True), part) if part else guide_text(full=full)
    if out_path is None:
        emit(out, text)
        return 0
    _write(out_path, text, force=force)
    logger.info("wrote the agent guide", extra={"path": str(out_path), "section": part or "all"})
    emit(out, f"wrote {out_path}\n")
    return 0
