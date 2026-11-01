"""Newcomer and architecture pages, and their examples, run tool calls through the executor."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGES = [
    ROOT / "README.md",
    DOCS / "index.md",
    *sorted((DOCS / "get-started").glob("*.md")),
    *sorted((DOCS / "how-to").glob("*.md")),
    *sorted((DOCS / "tutorials").glob("*.md")),
    *sorted((DOCS / "backends").glob("*.md")),
    *sorted((DOCS / "architecture").glob("*.md")),
    *sorted((DOCS / "cookbook").glob("*.md")),
    *sorted((DOCS / "guides").glob("*.md")),
    *sorted((DOCS / "migration").glob("*.md")),
]
# `spec.invoke` is taught only as a test tool, on the testing page.
TESTING_ONLY = {"examples/howto_testing.py"}
# testing-tools teaches spec.invoke; langchain-tools shows LangChain's `tool.invoke` as "before".
PAGE_EXEMPT = {"testing-tools.md", "langchain-tools.md"}
SNIPPET = re.compile(r'--8<-- "(examples/[\w/]+\.py)')
CALL = re.compile(r"\.invoke\((?!\.\.\.\))")  # a warning may name `spec.invoke(...)`


def _examples() -> set[str]:
    found: set[str] = set()
    for page in PAGES:
        found.update(SNIPPET.findall(page.read_text(encoding="utf-8")))
    return found


def test_newcomer_examples_never_call_spec_invoke() -> None:
    examples = _examples() - TESTING_ONLY
    assert "examples/quickstart.py" in examples
    offenders = [e for e in sorted(examples) if CALL.search((ROOT / e).read_text("utf-8"))]
    assert offenders == []


def test_newcomer_pages_never_show_spec_invoke_as_a_call() -> None:
    offenders = [
        page.relative_to(ROOT).as_posix()
        for page in PAGES
        if page.name not in PAGE_EXEMPT and CALL.search(page.read_text(encoding="utf-8"))
    ]
    assert offenders == []
