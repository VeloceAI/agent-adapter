"""Public files use generic domains: no tax terms and no internal company name."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CACHES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".jsonl", ".json", ".yaml", ".yml", ".tmpl", ".toml"}
TEXT_SUFFIXES |= {".svg", ".css", ".html", ".js", ""}  # "": Dockerfile
ROOT_FILES = ("README.md", "CHANGELOG.md", "AGENTS.md", "pyproject.toml", "mkdocs.yml")
FOLDERS = ("examples", "docs", "agent_adapter")
INTERNAL = Path("docs") / "superpowers"  # internal design notes, excluded from the site
# Third-party files copied verbatim (checked by tests/docs/test_vendored_mermaid.py).
VENDOR = Path("docs") / "assets" / "javascripts" / "vendor"
# Each pattern with a sample it must catch and a lookalike it must leave alone.
FORBIDDEN: dict[str, tuple[re.Pattern[str], str, str]] = {
    "GST": (re.compile(r"(\b|_)gst", re.I), "gst_rate(gstin), get_gst", "against"),
    # The ID is upper case or an identifier; a bare lower-case "pan" is the graph gesture.
    "PAN": (
        re.compile(r"\bPAN\b|_pan(\b|_)|\bpan_|\bpan\s*:\s*str"),
        "PAN, pan: str, user_pan, pan_number",
        "panel, pandas, span, drag to pan, pan and zoom",
    ),
    "tax": (re.compile(r"(\b|_)tax(es)?(\b|_)", re.I), "get_tax, tax-tools", "syntax, taxi"),
    "INR": (re.compile(r"(\b|_)inr(\b|_)", re.I), "83.2 INR, usd_to_inr", "inrange"),
    "file_return": (re.compile(r"file_return", re.I), "file_return(year)", "file_result"),
    "80C": (re.compile(r"(\b|_)80c\b", re.I), "section 80C, sec_80c", "180C, 80cm"),
    "income tax": (re.compile(r"income.tax", re.I), "income-tax return", "income statement"),
    "statute": (re.compile(r"(\b|_)statut", re.I), "statute index, get_statutes", "status"),
    "Taxmann": (re.compile(r"taxmann", re.I), "Taxmann", "taxi man"),
    "duty": (re.compile(r"(\b|_)dut(y|ies)(\b|_)", re.I), "duty_rate, customs_duties", "dutyfree?"),
}


def _texts(folder: Path) -> list[Path]:
    """Text files under `folder`, without caches and the internal docs."""
    found = []
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(ROOT)
        internal = any(rel.parts[: len(skip.parts)] == skip.parts for skip in (INTERNAL, VENDOR))
        cached = bool(CACHES & set(rel.parts))
        if path.is_file() and path.suffix in TEXT_SUFFIXES and not internal and not cached:
            found.append(path)
    return found


def _public_files() -> list[Path]:
    """Every file this guard covers; the built `site/` is regenerated, so it is not read."""
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for folder in FOLDERS:
        files += _texts(ROOT / folder)
    return files


def _hits(text: str) -> list[str]:
    return [
        f"{name} (line {text.count(chr(10), 0, match.start()) + 1}: {match.group(0)!r})"
        for name, (pattern, _, _) in FORBIDDEN.items()
        for match in pattern.finditer(text)
    ]


@pytest.mark.parametrize("path", _public_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_public_file_uses_generic_domains(path: Path) -> None:
    hits = _hits(path.read_text(encoding="utf-8"))
    assert not hits, f"replace with a generic domain: {hits}"


def test_the_guard_covers_every_public_source() -> None:
    files = {p.relative_to(ROOT).as_posix() for p in _public_files()}
    for expected in (
        "README.md",
        "CHANGELOG.md",
        "mkdocs.yml",
        "docs/changelog.md",
        "docs/index.md",
        "examples/quickstart.py",
        "agent_adapter/__init__.py",
        "agent_adapter/cli/templates/script.py.tmpl",
    ):
        assert expected in files, expected
    assert not any(f.startswith(("docs/superpowers/", "site/")) for f in files)
    assert not any(CACHES & set(f.split("/")) for f in files)


@pytest.mark.parametrize("name", FORBIDDEN)
def test_each_pattern_catches_its_sample_and_spares_its_lookalike(name: str) -> None:
    pattern, samples, lookalikes = FORBIDDEN[name]
    for sample in samples.split(", "):
        assert pattern.search(sample), sample
    for lookalike in lookalikes.split(", "):
        assert not pattern.search(lookalike), lookalike
