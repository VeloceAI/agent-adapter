"""`agent-adapter init`: copy a tested starter file and its requirements into a folder."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from agent_adapter._version import __version__
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

# The packaged copies of examples/template_*.py; a test keeps them byte-identical.
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class Template:
    """One starter project: its source, the file it becomes, what to install and run."""

    source: str
    filename: str
    requirements: tuple[str, ...]
    run: str

    def text(self) -> str:
        """The starter file's content."""
        return (TEMPLATE_DIR / f"{self.source}.py.tmpl").read_text(encoding="utf-8")


def _adapter(extra: str = "") -> str:
    """`agent-adapter[extra]~=X.Y`: the version this CLI ships with, any later patch."""
    major, minor = __version__.split(".")[:2]
    return f"agent-adapter{f'[{extra}]' if extra else ''}~={major}.{minor}"


TEMPLATES: dict[str, Template] = {
    "script": Template("script", "main.py", (_adapter(),), "python main.py"),
    "fastapi": Template(
        "fastapi", "app.py", (_adapter(), "fastapi", "uvicorn"), "uvicorn app:app --reload"
    ),
    "flask": Template("flask", "app.py", (_adapter(), "flask"), "flask --app app run"),
    "celery": Template("celery", "tasks.py", (_adapter(), "celery"), "python tasks.py"),
    "langgraph": Template("langgraph", "graph.py", (_adapter("langgraph"),), "python graph.py"),
    "openai-agents": Template(
        "openai_agents", "agent.py", (_adapter("openai-agents"),), "python agent.py"
    ),
    "mcp-server": Template("mcp_server", "server.py", (_adapter("mcp"),), "python server.py"),
    "pydantic-ai": Template(
        "pydantic_ai", "agent.py", (_adapter("pydantic-ai"),), "python agent.py"
    ),
    "adk": Template("adk", "agent.py", (_adapter("adk"),), "python agent.py"),
    "agno": Template("agno", "agent.py", (_adapter("agno"),), "python agent.py"),
    "strands": Template("strands", "agent.py", (_adapter("strands"),), "python agent.py"),
    "crewai": Template("crewai", "crew.py", (_adapter("crewai"),), "python crew.py"),
}


def write_project(name: str, folder: Path, *, force: bool = False) -> list[Path]:
    """Write template `name` and a requirements.txt into `folder`; return the paths written.

    Raises:
        ConfigError: unknown template, a file exists and `force` is False, a target is a
            symlink, or a file would land outside `folder`.
        OSError: the folder or a file cannot be written.
    """
    template = TEMPLATES.get(name)
    if template is None:
        raise ConfigError(f"no template named {name!r}; choose one of {', '.join(TEMPLATES)}")
    files = {
        folder / template.filename: template.text(),
        folder / "requirements.txt": "".join(f"{line}\n" for line in template.requirements),
    }
    # Links first: resolve() follows a link, so it would report one as "outside" instead.
    links = sorted(str(path) for path in files if path.is_symlink())
    if links:  # even with --force: the link could point anywhere
        raise ConfigError(f"refusing to write through symlink {', '.join(links)}; remove it")
    for path in files:
        if path.resolve(strict=False).parent != folder.resolve(strict=False):
            raise ConfigError(f"refusing to write {path}: it is outside {folder}")
    existing = sorted(str(path) for path in files if path.exists())
    if existing and not force:
        raise ConfigError(f"not overwriting {', '.join(existing)}; pass --force to replace")
    folder.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        _write(path, text, force=force)
    logger.info("wrote template %r into %s", name, folder)
    return list(files)


def _write(path: Path, text: str, *, force: bool) -> None:
    """Write `text` without following a link; without `force` the file must not exist yet."""
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_TRUNC if force else os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:  # created since the check above
        raise ConfigError(f"not overwriting {path}; pass --force to replace") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
