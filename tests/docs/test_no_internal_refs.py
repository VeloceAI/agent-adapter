"""Published docs and the README carry no internal design, process or machine references."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
# Third-party files copied verbatim (checked by tests/docs/test_vendored_mermaid.py).
VENDOR = Path("assets") / "javascripts" / "vendor"
CONFIG = ROOT / "mkdocs.yml"
PACKAGE = ROOT / "agent_adapter"
# Text files mkdocs copies or renders into the site; `docs/superpowers/` is excluded from it.
TEXT_SUFFIXES = {".md", ".txt", ".css", ".js", ".svg", ".html"}
# mkdocs.yml keys that end up in the built HTML (the rest is build configuration).
RENDERED_KEYS = re.compile(r"^(site_name|site_url|site_description|copyright):.*$", re.M)
SNIPPET = re.compile(r'--8<--\s+"([^"]+)"')
FORBIDDEN = {
    "internal docs folder": re.compile(r"superpowers", re.I),
    "interface contract": re.compile(r"interface-contract", re.I),
    "design spec": re.compile(r"design spec", re.I),
    "section number": re.compile("§"),
    "work-package id": re.compile(r"\bWP\d"),
    "local release path": re.compile(r"D:\\aa", re.I),
    "local user path": re.compile(r"C:\\Users", re.I),
    "internal company name": re.compile(r"Taxmann", re.I),
    "internal edit log": re.compile(r"SHARED-EDITS", re.I),
}


def _site_sources() -> list[Path]:
    """Every docs file mkdocs publishes, plus the README."""
    files = [ROOT / "README.md"]
    for path in sorted(DOCS.rglob("*")):
        rel = path.relative_to(DOCS)
        skipped = "superpowers" in rel.parts or rel.parts[: len(VENDOR.parts)] == VENDOR.parts
        if path.is_file() and path.suffix in TEXT_SUFFIXES and not skipped:
            files.append(path)
    return files


def _snippet_targets(files: list[Path]) -> list[Path]:
    """Files pulled into pages with `--8<--` (paths are relative to the repo root)."""
    found: set[Path] = set()
    pending = list(files)
    while pending:
        text = pending.pop().read_text(encoding="utf-8")
        for ref in SNIPPET.findall(text):
            target = ROOT / ref.split(":", 1)[0]
            if target.is_file() and target not in found:
                found.add(target)
                pending.append(target)
    return sorted(found)


def _published() -> list[Path]:
    sources = _site_sources()
    return sources + [p for p in _snippet_targets(sources) if p not in sources]


def _hits(text: str) -> list[str]:
    """One `line: label: text` entry per forbidden pattern found."""
    return [
        f"{number}: {label}: {line.strip()[:100]}"
        for number, line in enumerate(text.splitlines(), start=1)
        for label, pattern in FORBIDDEN.items()
        if pattern.search(line)
    ]


def test_no_internal_refs_scans_pages_snippets_and_readme() -> None:
    published = {p.relative_to(ROOT).as_posix() for p in _published()}
    assert {"README.md", "CHANGELOG.md", "docs/index.md", "docs/llms.txt"} <= published
    assert any(name.startswith("examples/") for name in published)
    assert not any("superpowers" in name for name in published)


@pytest.mark.parametrize("path", _published(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_internal_refs_in_published_file(path: Path) -> None:
    assert _hits(path.read_text(encoding="utf-8")) == []


def _public_docstrings(path: Path) -> str:
    """Docstrings of a module and its public classes and functions (what mkdocstrings shows)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes: list[ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = [tree]
    nodes += [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    docs = (ast.get_docstring(node, clean=False) for node in nodes)
    return "\n".join(doc for doc in docs if doc)


def _is_public(path: Path) -> bool:
    parts = path.relative_to(PACKAGE).parts
    return not any(part.startswith("_") and part != "__init__.py" for part in parts)


PUBLIC_MODULES = sorted(path for path in PACKAGE.rglob("*.py") if _is_public(path))


@pytest.mark.parametrize("path", PUBLIC_MODULES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_internal_refs_in_api_reference_docstrings(path: Path) -> None:
    assert _hits(_public_docstrings(path)) == []


def test_no_internal_refs_in_rendered_site_settings() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    rendered = "\n".join(match.group(0) for match in RENDERED_KEYS.finditer(config))
    assert "site_url: https://" in rendered
    assert _hits(rendered) == []


def test_no_internal_refs_detector_catches_each_pattern() -> None:
    samples = [
        "see docs/superpowers/specs",
        "the 2026-10-29-interface-contract.md file",
        "every feature in the Design Spec",
        "contract §14.4",
        "done in WP18",
        r"built to D:\aa\release",
        r"C:\Users\someone\Desktop",
        "the Taxmann pipeline",
        "logged in SHARED-EDITS.md",
    ]
    assert [len(_hits(sample)) for sample in samples] == [1] * len(samples)
    assert _hits("run_plan waves, the run owner, a contract test, RFC 9457") == []
