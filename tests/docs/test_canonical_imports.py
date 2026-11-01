"""Examples and docs import each public name from its one canonical path."""

import ast
import importlib
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import agent_adapter

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TOP_LEVEL = frozenset(agent_adapter.__all__)
# Design history, not published: mkdocs.yml excludes it.
UNPUBLISHED = DOCS / "superpowers"
FENCE = re.compile(r"^[ \t]*```[ \t]*(?:python|py)\b[^\n]*\n(.*?)^[ \t]*```", re.M | re.S)
SNIPPET_LINE = re.compile(r"^[ \t]*--8<--.*$", re.M)
IMPORT_LINE = re.compile(r"^[ \t]*(?:from|import)[ \t]+agent_adapter\b", re.M)


def _pages() -> list[Path]:
    pages = [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]
    return [page for page in pages if UNPUBLISHED not in page.parents]


def _blocks(page: Path) -> Iterator[tuple[int, str]]:
    """Yield (first line, code) for each fenced python block, without snippet includes."""
    text = page.read_text(encoding="utf-8")
    for match in FENCE.finditer(text):
        code = textwrap.dedent(SNIPPET_LINE.sub("", match.group(1)))
        if code.strip():
            yield text.count("\n", 0, match.start(1)) + 1, code


def _parse(code: str) -> ast.Module:
    """Parse a block; for a fragment that isn't valid Python, parse its import statements alone."""
    try:
        return ast.parse(code)
    except SyntaxError:
        pass
    lines = code.splitlines()
    statements: list[str] = []
    for start in (m.start() for m in IMPORT_LINE.finditer(code)):
        index = code.count("\n", 0, start)
        statement = lines[index].strip()
        while "(" in statement and ")" not in statement and index + 1 < len(lines):
            index += 1
            statement += " " + lines[index].strip()
        statements.append(statement)
    return ast.parse("\n".join(statements))


def _is_private(module: str) -> bool:
    return any(part.startswith("_") for part in module.split("."))


def _is_package(name: str) -> bool:
    """True for a public subpackage or module of agent_adapter, such as `errors`."""
    home = ROOT / "agent_adapter"
    return (home / name).is_dir() or (home / f"{name}.py").is_file()


def _home_problem(module: str, name: str) -> str | None:
    """Why `name` must not come from `module`: not in its `__all__`, or a shallower home exists."""
    folder = ROOT / Path(*module.split("."))
    if (folder / name).is_dir() or (folder / f"{name}.py").is_file():
        return None  # a submodule, such as `from agent_adapter.cache import providers`
    home = importlib.import_module(module)
    value = getattr(home, name, None)
    parts = module.split(".")
    for depth in range(2, len(parts)):
        package = ".".join(parts[:depth])
        parent = importlib.import_module(package)
        if name in getattr(parent, "__all__", ()) and getattr(parent, name) is value:
            return f"from {module} import {name}: use `from {package} import {name}`"
    if name not in getattr(home, "__all__", ()):
        return f"from {module} import {name}: not in {module}.__all__"
    return None


def _problems(tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "agent_adapter.core" or alias.name.startswith(
                    "agent_adapter.core."
                ):
                    yield f"import {alias.name}: agent_adapter.core is internal"
                elif alias.name.startswith("agent_adapter.") and _is_private(alias.name):
                    yield f"import {alias.name}: private module"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module = node.module
            if module != "agent_adapter" and not module.startswith("agent_adapter."):
                continue
            if module == "agent_adapter.core" or module.startswith("agent_adapter.core."):
                yield f"from {module}: agent_adapter.core is internal"
                continue
            if _is_private(module):
                yield f"from {module}: private module"
                continue
            for alias in node.names:
                name = alias.name
                if name.startswith("_") and name != "__version__":
                    yield f"from {module} import {name}: private name"
                elif module == "agent_adapter.integrations":
                    yield (
                        f"from {module} import {name}: use "
                        f"`from agent_adapter.integrations.{name} import <names>`"
                    )
                elif module != "agent_adapter" and name in TOP_LEVEL:
                    yield f"from {module} import {name}: use `from agent_adapter import {name}`"
                elif module == "agent_adapter" and name not in TOP_LEVEL and not _is_package(name):
                    yield f"from agent_adapter import {name}: not a top-level name"
                elif module != "agent_adapter" and (problem := _home_problem(module, name)):
                    yield problem


def _example_problems() -> list[str]:
    found: list[str] = []
    for path in sorted((ROOT / "examples").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(ROOT).as_posix()
        found += [f"{rel}: {problem}" for problem in _problems(tree)]
    return found


def _page_problems() -> list[str]:
    found: list[str] = []
    for page in _pages():
        rel = page.relative_to(ROOT).as_posix()
        for line, code in _blocks(page):
            found += [f"{rel}:{line}: {problem}" for problem in _problems(_parse(code))]
    return found


def test_examples_use_canonical_imports() -> None:
    assert _example_problems() == []


def test_doc_code_blocks_use_canonical_imports() -> None:
    assert _page_problems() == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from agent_adapter.core.types import Event", "agent_adapter.core is internal"),
        ("import agent_adapter.core.errors", "agent_adapter.core is internal"),
        ("from agent_adapter._version import x", "private module"),
        ("from agent_adapter.approval._store import X", "private module"),
        ("import agent_adapter.cache._lru", "private module"),
        ("from agent_adapter.exec import Executor", "use `from agent_adapter import Executor`"),
        ("from agent_adapter import _version", "private name"),
        ("from agent_adapter import ToolNotFound", "not a top-level name"),
        (
            "from agent_adapter.integrations import openai as oa",
            "agent_adapter.integrations.openai",
        ),
        ("from agent_adapter.exec.retry import Backoff", "use `from agent_adapter.exec import"),
        ("from agent_adapter.server.auth import is_invoker", "use `from agent_adapter.server"),
        ("from agent_adapter.cost.prices import Table", "not in agent_adapter.cost.prices.__all__"),
        ("from agent_adapter.hooks import nothing_here", "not in agent_adapter.hooks.__all__"),
    ],
)
def test_guard_flags_each_forbidden_form(source: str, expected: str) -> None:
    problems = list(_problems(ast.parse(source)))
    assert len(problems) == 1
    assert expected in problems[0]


@pytest.mark.parametrize(
    "source",
    [
        "from agent_adapter import Executor, __version__",
        "from agent_adapter.errors import ToolNotFound, ErrorInfo",
        "from agent_adapter.hooks import ShortCircuit",
        "from agent_adapter import errors, hooks",
        "from agent_adapter.integrations.openai import chat_tools",
        "from agent_adapter.cache import providers",
        "from agent_adapter.cache.providers import AnthropicPromptCache",
        "from agent_adapter.models import ModelCaps, LLMPort",
        "from . import local",
        "import agent_adapter",
    ],
)
def test_guard_accepts_canonical_forms(source: str) -> None:
    assert list(_problems(ast.parse(source))) == []


def test_guard_parses_imports_out_of_a_fragment() -> None:
    fragment = "result = = run()\nfrom agent_adapter.exec import (\n    Executor,\n)\n"
    problems = list(_problems(_parse(fragment)))
    assert problems == [
        "from agent_adapter.exec import Executor: use `from agent_adapter import Executor`"
    ]


def test_guard_finds_python_blocks_and_drops_snippet_includes(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        '# T\n\n```python\n--8<-- "examples/x.py:a"\n```\n\n'
        '```py title="x"\nfrom agent_adapter.core import y\n```\n\n```bash\nls\n```\n',
        encoding="utf-8",
    )
    assert [(line, code.strip()) for line, code in _blocks(page)] == [
        (8, "from agent_adapter.core import y")
    ]
