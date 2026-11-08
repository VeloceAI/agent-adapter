"""`agent-adapter migrate`: report framework tool definitions and suggest a diff (read-only)."""

from __future__ import annotations

import argparse
import ast
import io
import logging
import os
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from agent_adapter.cli import _migrate_rules as rules
from agent_adapter.cli._migrate_diff import suggested_diff
from agent_adapter.cli._migrate_scan import Finding, scan_module
from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 2_000_000  # larger files are skipped: generated code, not hand-written tools
SKIP_DIRS = frozenset(
    {"__pycache__", "node_modules", "site-packages", "venv", "env", "build", "dist"}
)


@dataclass(frozen=True)
class FileReport:
    """What `migrate` found in one file, its suggested diff, or why it was skipped."""

    rel: str
    findings: tuple[Finding, ...] = ()
    diff: str = ""
    skipped: str = ""


def add_migrate_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `migrate PATH [--framework NAME] [--diff]`."""
    migrate = commands.add_parser(
        "migrate", help="list framework tool definitions and suggest registry code (read-only)"
    )
    migrate.add_argument("path", type=Path, metavar="PATH", help="a Python file or a folder")
    migrate.add_argument(
        "--framework",
        choices=["auto", *rules.FRAMEWORKS],
        default="auto",
        help="only this framework's tools (default: every framework found)",
    )
    migrate.add_argument("--diff", action="store_true", help="print the suggested unified diff")


def run_migrate(args: argparse.Namespace, out: TextIO) -> int:
    """Print the report (and the diff); never writes a file. Exit 0.

    Raises:
        ConfigError: PATH does not exist or is not a file or folder.
    """
    reports = scan_path(args.path, args.framework)
    out.write(report_text(reports))
    if args.diff:
        out.write("".join(report.diff for report in reports))
    return 0


def scan_path(path: Path, framework: str = "auto") -> list[FileReport]:
    """Scan one `.py` file, or every `.py` file under a folder, without importing any of it.

    Raises:
        ConfigError: `path` does not exist or is not a file or folder.
    """
    if path.is_file():
        return [scan_file(path, path.name, framework)]
    if not path.is_dir():
        raise ConfigError(f"{path} is not a file or folder")
    return [scan_file(file, file.relative_to(path).as_posix(), framework) for file in _files(path)]


def scan_file(path: Path, rel: str, framework: str = "auto") -> FileReport:
    """The report for one file; unreadable or invalid files come back with `skipped` set."""
    if path.is_symlink():
        return FileReport(rel, skipped="a symlink, not followed")
    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return FileReport(rel, skipped=f"larger than {MAX_FILE_BYTES} bytes")
        data = path.read_bytes()
    except OSError as exc:
        logger.debug("migrate could not read a file (%s)", type(exc).__name__)
        return FileReport(rel, skipped=f"unreadable ({type(exc).__name__})")
    return scan_source(data, rel, framework)


def scan_source(data: bytes, rel: str, framework: str = "auto") -> FileReport:
    """The report for one module's bytes (its coding cookie is honoured)."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        source = data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError):  # a bad coding cookie or bytes
        return FileReport(rel, skipped="not decodable as Python source")
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        where = f" at line {exc.lineno}" if exc.lineno else ""
        return FileReport(rel, skipped=f"syntax error{where}")
    except (ValueError, RecursionError, MemoryError) as exc:  # null bytes; absurd nesting
        return FileReport(rel, skipped=f"not parseable ({type(exc).__name__})")
    try:
        scan = scan_module(tree, source, framework)
        diff = suggested_diff(rel, tree, source, scan) if scan.findings else ""
    except (RecursionError, MemoryError) as exc:  # a tree too deep to walk: skip this file only
        logger.debug("migrate skipped a file (%s)", type(exc).__name__)
        return FileReport(rel, skipped=f"too deeply nested to scan ({type(exc).__name__})")
    return FileReport(rel, tuple(scan.findings), diff)


def report_text(reports: list[FileReport]) -> str:
    """The human-readable report: one line per finding or skipped file, then a summary."""
    lines = ["agent-adapter migrate: read-only, no file was changed"]
    found = [r for r in reports if r.findings]
    for report in reports:
        if report.skipped:
            lines.append(f"{report.rel}: skipped, {report.skipped}")
        for f in report.findings:
            lines.append(f"{report.rel}:{f.line}: {f.framework} {f.kind} `{f.name}`: {f.action}")
    findings = [f for r in found for f in r.findings]
    manual = sum(not f.automatic for f in findings)
    lines.append(
        f"{len(findings)} tool definition(s) in {len(found)} file(s); {manual} need a manual step"
    )
    for framework in sorted({f.framework for f in findings}):
        bridge = rules.BRIDGES[framework]
        lines.append(f"{framework}: give it {bridge.call} from {bridge.module}")
    if findings:
        lines.append("then call tools through the executor; run with --diff to see the edits")
    return "".join(f"{line}\n" for line in lines)


def _files(root: Path) -> Iterator[Path]:
    """Every `.py` file under `root`, sorted; hidden, virtualenv and build folders skipped."""
    for folder, dirs, files in os.walk(root):  # does not follow symlinked folders
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(folder) / name
