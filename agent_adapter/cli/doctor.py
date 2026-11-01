"""`agent-adapter doctor`: Python, package, extras, integrations, shadowing and pip index."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from agent_adapter._version import __version__
from agent_adapter.cli.findings import Finding, Level
from agent_adapter.cli.pipindex import index_findings
from agent_adapter.core.errors import MissingDependency
from agent_adapter.integrations.base import Integration, _check_requires
from agent_adapter.integrations.discovery import available_integrations

logger = logging.getLogger(__name__)

DIST = "agent-adapter"
MIN_PYTHON = (3, 11)
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")
_EXTRA_MARKER = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")
Requirement = tuple[str, tuple[str, ...]]  # (distribution, the extras it asks for)
Extras = Mapping[str, list[Requirement]]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version(dist: str) -> str | None:
    """The installed version of distribution `dist`, or None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def extras_table(requires: list[str]) -> dict[str, list[Requirement]]:
    """extra -> [(distribution, its extras)] parsed from Requires-Dist lines."""
    table: dict[str, list[Requirement]] = {}
    for line in requires:
        marker, requirement = _EXTRA_MARKER.search(line), _REQUIREMENT.match(line)
        if marker is None or requirement is None:
            continue
        nested = tuple(e.strip() for e in (requirement.group(2) or "").split(",") if e.strip())
        table.setdefault(_norm(marker.group(1)), []).append((_norm(requirement.group(1)), nested))
    return table


def extra_installed(extra: str, table: Extras, seen: frozenset[str] = frozenset()) -> bool:
    """True when every distribution of `extra` (and of extras it names) is installed."""
    if extra in seen or extra not in table:
        return extra in seen
    for dist, nested in table[extra]:
        if dist == DIST:
            if not all(extra_installed(_norm(n), table, seen | {extra}) for n in nested):
                return False
        elif _version(dist) is None:
            return False
    return True


def check_python(version: tuple[int, int] | None = None) -> Finding:
    """Python must be 3.11 or later."""
    found = version or (sys.version_info.major, sys.version_info.minor)
    text = f"{found[0]}.{found[1]} ({sys.executable})"
    if found < MIN_PYTHON:
        return Finding("error", "python", f"{text}: Agent Adapter needs Python 3.11 or later")
    level: Level = "ok" if sys.prefix != sys.base_prefix else "warn"
    note = "" if level == "ok" else "; not in a virtual environment"
    return Finding(level, "python", text + note)


def check_package(installed: str | None) -> Finding:
    """The installed metadata must match the imported code."""
    where = Path(__file__).resolve().parents[1]
    if installed is None:
        return Finding("warn", "agent-adapter", f"{__version__} from {where} is not installed")
    if installed != __version__:
        return Finding(
            "warn",
            "agent-adapter",
            f"metadata says {installed} but the code is {__version__} ({where}): reinstall it "
            "(pip install -e . in a checkout) so entry points and extras are current",
        )
    return Finding("ok", "agent-adapter", f"{__version__} ({where})")


def check_extras(table: Extras) -> Finding:
    """The extras whose distributions are all installed."""
    ready = sorted(extra for extra in table if extra_installed(extra, table))
    return Finding("ok", "extras", ", ".join(ready) or "none (the core needs no extra)")


def _dist_versions(integration: Integration, table: Extras) -> str:
    dists = [d for d, _ in table.get(_norm(integration.extra), []) if d != DIST]
    return ", ".join(f"{d} {_version(d) or 'missing'}" for d in dists)


@contextmanager
def _captured(name: str) -> Iterator[list[logging.LogRecord]]:
    """Collect `name`'s WARNING+ records instead of printing them twice."""
    records: list[logging.LogRecord] = []
    target = logging.getLogger(name)

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler, propagate = _Keep(logging.WARNING), target.propagate
    target.addHandler(handler)
    target.propagate = False
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.propagate = propagate


def check_integrations(table: Extras) -> list[Finding]:
    """One finding per discovered integration, plus one per plugin that failed to load."""
    with _captured("agent_adapter.integrations.discovery") as skipped:
        found = available_integrations(refresh=True)
    findings = [Finding("warn", "integrations", r.getMessage()) for r in skipped]
    for name, integration in sorted(found.items()):
        try:
            _check_requires(integration)
        except MissingDependency as missing:
            hint = f'pip install "agent-adapter[{integration.extra}]"' if integration.extra else ""
            findings.append(Finding("info", f"integration {name}", hint or str(missing)))
            continue
        versions = _dist_versions(integration, table)
        findings.append(Finding("ok", f"integration {name}", f"ready {versions}".rstrip()))
    return findings


def check_shadowing(cwd: Path, found: Mapping[str, Integration]) -> list[Finding]:
    """A local `mcp.py` or `agents/` folder hides the real SDK from scripts run here."""
    names = {root.split(".")[0] for i in found.values() for root in i.requires}
    package = Path(__file__).resolve().parents[1]
    findings = []
    for name in sorted(names | {"agent_adapter"}):
        for path in (cwd / f"{name}.py", cwd / name / "__init__.py"):
            if path.is_file() and path.resolve().parent != package:
                findings.append(
                    Finding("warn", "shadowing", f"{path} hides the installed {name}: rename it")
                )
    return findings


def diagnose(environ: Mapping[str, str] | None = None, cwd: Path | None = None) -> list[Finding]:
    """Every doctor check, in display order."""
    env = os.environ if environ is None else environ
    try:
        requires = importlib.metadata.distribution(DIST).requires or []
        installed: str | None = importlib.metadata.version(DIST)
    except importlib.metadata.PackageNotFoundError:
        requires, installed = [], None
    table = extras_table(requires)
    findings = [check_python(), check_package(installed), check_extras(table)]
    findings += check_integrations(table)
    findings += check_shadowing(cwd or Path.cwd(), available_integrations())
    findings += index_findings(env)
    return findings
