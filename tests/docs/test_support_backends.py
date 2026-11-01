"""The backends matrix is generated from BACKENDS, the examples and tests on disk and noxfile.py.

Regenerate the table with `python -m tests.docs.test_support_backends` and paste the output
between the markers in docs/support-backends.md.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from tests.docs.backends_matrix import BACKENDS, NODE, Backend, Check
from tests.docs.test_support_matrix import _constant, _nox, _pyproject

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGE = DOCS / "support-backends.md"
START, END = "<!-- support-backends:start -->", "<!-- support-backends:end -->"
CONTRACT_MARK = re.compile(r"^pytestmark = pytest\.mark\.contract$|@pytest\.mark\.contract", re.M)


@dataclass
class Session:
    """What a nox session runs and installs."""

    name: str
    paths: list[str]  # empty: the whole suite
    marker: str | None  # "contract", "not contract", or None
    packages: set[str]


def _name(requirement: str) -> str:
    return re.split(r"[\[<>=!~ ;]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _extra_packages(extras: dict[str, list[str]], names: list[str]) -> set[str]:
    found: set[str] = set()
    for extra in names:
        for requirement in extras.get(extra, []):
            if match := re.fullmatch(r"agent-adapter\[([\w,-]+)\]", requirement):
                found |= _extra_packages(extras, match.group(1).split(","))
            else:
                found.add(_name(requirement))
    return found


def _installs(strings: list[str], extras: dict[str, list[str]]) -> set[str]:
    found: set[str] = set()
    for value in strings:
        if match := re.fullmatch(r"\.\[([\w,-]+)\]", value):
            found |= _extra_packages(extras, match.group(1).split(","))
        elif re.fullmatch(r"[A-Za-z][\w.-]*(\[[\w,-]+\])?([<>=!~][^ ]*)?", value):
            found.add(_name(value))
    return found


def sessions() -> list[Session]:
    """Every pytest-running nox session, with the contract sessions expanded from EXTRAS."""
    source = (ROOT / "noxfile.py").read_text(encoding="utf-8")
    tree = _nox()
    extras = _pyproject()["project"]["optional-dependencies"]
    found: list[Session] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        strings = [
            s.value
            for s in ast.walk(node)
            if isinstance(s, ast.Constant) and isinstance(s.value, str)
        ]
        if "pytest" not in strings:
            continue
        name = node.name
        for decorator in node.decorator_list:
            for keyword in getattr(decorator, "keywords", []):
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    name = str(keyword.value.value)
        paths = [s for s in strings if s.startswith("tests/")]
        marker = next((strings[i + 1] for i, s in enumerate(strings[:-1]) if s == "-m"), None)
        packages = _installs(strings, extras)
        if 'shutil.which("node")' in (ast.get_source_segment(source, node) or ""):
            packages.add(NODE)
        found.append(Session(name, paths, marker, packages))
    own_packages, all_markers = _constant(tree, "SESSION_PACKAGES"), _constant(tree, "ALL_MARKERS")
    for target, extra in _constant(tree, "EXTRAS").items():
        packages = _installs([f".[dev,{extra}]" if extra else ".[dev]"], extras)
        packages |= _installs(own_packages.get(target, []), extras)
        marker = None if target in all_markers else "contract"
        paths = [f"tests/integrations/test_{target}_contract.py", f"tests/integrations/{target}"]
        found.append(Session(f"contract-{target}", paths, marker, packages))
    return found


def _covers(session: Session, test_file: str) -> bool:
    inside = not session.paths or any(
        test_file == p or test_file.startswith(p + "/") for p in session.paths
    )
    marked = bool(CONTRACT_MARK.search((ROOT / test_file).read_text(encoding="utf-8")))
    wanted = {"contract": marked, "not contract": not marked, None: True}[session.marker]
    return inside and wanted


def runs_in(check: Check, all_sessions: list[Session]) -> list[str]:
    """The nox sessions that collect this test and install everything it needs."""
    test_file = check.test.split("::", 1)[0]
    return sorted(
        s.name for s in all_sessions if _covers(s, test_file) and set(check.needs) <= s.packages
    )


def _tested(backend: Backend, all_sessions: list[Session]) -> str:
    if not backend.checks:
        return f"documented recipe: {backend.recipe}"
    parts = []
    for check in backend.checks:
        names = runs_in(check, all_sessions)
        if names:
            where = "in " + ", ".join(f"`{n}`" for n in names)
        else:
            where = "skipped: no nox session installs " + ", ".join(f"`{n}`" for n in check.needs)
        note = f"; {check.note}" if check.note else ""
        parts.append(f"`{check.test}` {where}{note}")
    return "<br>".join(parts)


def render() -> str:
    """The table, exactly as it appears between the page's markers."""
    all_sessions = sessions()
    rows = [
        "| Backend | Attach with | Extra | Guide | Example | Tested by, in nox session |",
        "|---|---|---|---|---|---|",
    ]
    for b in BACKENDS:
        title = (DOCS / b.guide).read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
        examples = "<br>".join(f"`{e}`" for e in b.examples) or "none"
        rows.append(
            f"| {b.name} | {b.attach} | {b.extra} | [{title}]({b.guide}) | {examples} | "
            f"{_tested(b, all_sessions)} |"
        )
    return "\n".join(rows)


def _block() -> str:
    text = PAGE.read_text(encoding="utf-8")
    return text.split(START, 1)[1].split(END, 1)[0].strip()


def test_support_backends_matches_the_list_and_noxfile() -> None:
    assert _block() == render(), "regenerate: python -m tests.docs.test_support_backends"


def test_every_listed_guide_example_and_test_exists() -> None:
    for backend in BACKENDS:
        assert (DOCS / backend.guide).is_file(), backend.guide
        assert backend.checks or backend.recipe, f"{backend.name}: a test or a recipe reason"
        for example in backend.examples:
            assert (ROOT / example).is_file(), example
        for check in backend.checks:
            test_file, test = check.test.split("::", 1)
            text = (ROOT / test_file).read_text(encoding="utf-8")
            assert re.search(rf"^(async )?def {test}\(", text, re.M), check.test


def test_every_example_is_named_on_its_guide() -> None:
    for backend in BACKENDS:
        guide = (DOCS / backend.guide).read_text(encoding="utf-8")
        for example in backend.examples:
            assert example.rsplit("/", 1)[-1] in guide, f"{example} not on {backend.guide}"


def test_framework_free_checks_run_in_core_and_framework_ones_in_docs_examples() -> None:
    all_sessions = sessions()
    for backend in BACKENDS:
        for check in backend.checks:
            names = runs_in(check, all_sessions)
            if check.test.startswith("tests/docs/") and not check.needs:
                assert "core" in names and "docs-examples" in names, check.test


def test_the_matrix_pages_link_each_other() -> None:
    assert "support-backends.md" in (DOCS / "support-matrix.md").read_text(encoding="utf-8")
    assert "support-matrix.md" in PAGE.read_text(encoding="utf-8")


if __name__ == "__main__":
    print(render())
