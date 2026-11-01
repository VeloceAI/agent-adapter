"""Argument parsing and dispatch for the `agent-adapter` command."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from agent_adapter._version import __version__
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

# One copy-paste example per command, shown by `agent-adapter --help` and in the agent guide.
EXAMPLES: dict[str, str] = {
    "init": "agent-adapter init fastapi my-service",
    "doctor": "agent-adapter doctor",
    "guide": "agent-adapter guide --out docs/agent-adapter.md",
    "ui": "agent-adapter ui",
}


def _parser() -> argparse.ArgumentParser:
    from agent_adapter.cli.scaffold import TEMPLATES

    parser = argparse.ArgumentParser(
        prog="agent-adapter",
        description="Start a project, check this environment, or print the AI-agent guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"agent-adapter {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    init = commands.add_parser("init", help="write a starter project from a tested template")
    init.add_argument("template", choices=list(TEMPLATES), help="the kind of project")
    init.add_argument("folder", nargs="?", default=".", help="where to write (default: here)")
    init.add_argument("--force", action="store_true", help="replace files that exist")
    commands.add_parser("doctor", help="check Python, extras, integrations and the pip index")
    _add_guide(commands)
    from agent_adapter.cli.ui import add_ui_parser

    add_ui_parser(commands)
    parser.epilog = _epilog(commands)
    return parser


def _add_guide(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from agent_adapter.cli.guide import SECTIONS

    guide = commands.add_parser("guide", help="print the guide for AI coding agents (AGENTS.md)")
    guide.add_argument("--out", type=Path, metavar="PATH", help="write it to PATH instead")
    guide.add_argument("--force", action="store_true", help="replace PATH if it exists")
    guide.add_argument("--full", action="store_true", help="add the API index (every signature)")
    guide.add_argument("--section", choices=list(SECTIONS), help="only this part")


def _epilog(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> str:
    """`examples:` with one line per registered command."""
    lines = [f"  {EXAMPLES.get(name, f'agent-adapter {name}')}" for name in commands.choices]
    return "examples:\n" + "\n".join(lines)


def command_lines() -> list[str]:
    """`example: help` for every command, in `--help` order (used by the agent guide)."""
    parser = _parser()
    # argparse has no public accessor for the subcommands and their help strings.
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return [
        f"`{EXAMPLES.get(choice.dest, f'agent-adapter {choice.dest}')}`: {choice.help}"
        for choice in commands._choices_actions
    ]


def run_init(template: str, folder: Path, force: bool, out: TextIO) -> int:
    """Write the template; print what to install and run next."""
    from agent_adapter.cli.scaffold import TEMPLATES, write_project

    written = write_project(template, folder, force=force)
    chosen = TEMPLATES[template]
    out.write("".join(f"wrote {path}\n" for path in written))
    out.write(f"next:\n  cd {folder}\n  pip install -r requirements.txt\n  {chosen.run}\n")
    return 0


def run_doctor(out: TextIO) -> int:
    """Print every finding; exit 1 when any is an error."""
    from agent_adapter.cli.doctor import diagnose

    findings = diagnose()
    out.write("".join(f"{finding.line()}\n" for finding in findings))
    errors = sum(finding.level == "error" for finding in findings)
    warnings = sum(finding.level == "warn" for finding in findings)
    out.write(f"{errors} error(s), {warnings} warning(s)\n")
    return 1 if errors else 0


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run the command line; return the exit code (2 for a usage or config error)."""
    stream = out or sys.stdout
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return run_init(args.template, Path(args.folder), args.force, stream)
        if args.command == "guide":
            from agent_adapter.cli.guide import run_guide

            return run_guide(args.out, args.section, stream, full=args.full, force=args.force)
        if args.command == "ui":
            from agent_adapter.cli.ui import run_ui

            return run_ui(args, stream)
        return run_doctor(stream)
    except (ConfigError, OSError) as exc:
        logger.debug("agent-adapter %s failed", args.command, exc_info=True)
        sys.stderr.write(f"agent-adapter {args.command}: {exc}\n")
        return 2
