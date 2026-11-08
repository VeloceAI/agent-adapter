"""Every recipe in the AI-agent guide is short, plain-style, valid against the API, and runs.

A recipe whose framework is not installed is compiled and checked against the real
signatures, then skipped; the `docs-examples` nox session installs those frameworks.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest

import agent_adapter
from agent_adapter.exec import reset_default_executor
from scripts.agent_guide import recipes
from scripts.agent_guide.recipes import MAX_LINES, PRELUDE, Recipe

# Prelude names whose methods a snippet may call: checked against these classes.
PRELUDE_TYPES = {"executor": agent_adapter.Executor, "registry": agent_adapter.Registry}
COMPOUND = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While)
COMPOUND += (ast.With, ast.Try, ast.ExceptHandler)


def _resolve(node: ast.expr, names: dict[str, Any]) -> Any:
    """The object a `Name` / dotted `Attribute` refers to, or None when unknown."""
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve(node.value, names)
        if owner is None:
            return None
        assert hasattr(owner, node.attr), f"{ast.unparse(node)} does not exist"
        return getattr(owner, node.attr)
    return None


def _imported(tree: ast.Module) -> dict[str, Any]:
    """`aa`, prelude objects and every name a snippet imports from agent_adapter."""
    names: dict[str, Any] = {"aa": agent_adapter, **PRELUDE_TYPES}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agent_adapter"):
            module = importlib.import_module(node.module or "")
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module} has no {alias.name}"
                names[alias.asname or alias.name] = getattr(module, alias.name)
    return names


def api_problems(code: str) -> None:
    """Assert every agent_adapter name exists and every call to one binds its signature."""
    tree = ast.parse(code)
    names = _imported(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            _resolve(node, names)
        if not isinstance(node, ast.Call):
            continue
        target = _resolve(node.func, names)
        if target is None or not callable(target):
            continue
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            continue
        args: list[Any] = [None] * len(node.args)
        owner = node.func.value if isinstance(node.func, ast.Attribute) else None
        if isinstance(owner, ast.Name) and owner.id in PRELUDE_TYPES:
            args.insert(0, None)  # self
        try:
            inspect.signature(target).bind(*args, **{k.arg: None for k in node.keywords if k.arg})
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"{ast.unparse(node)}: {exc}") from exc


def style_problems(code: str) -> list[str]:
    """One-line compound statements and `;`-joined statements (E701, E702, E704)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(code)):
        body = getattr(node, "body", None)
        if isinstance(node, COMPOUND) and body and body[0].lineno == node.lineno:
            found.append(f"line {node.lineno}: body on the header line")
        for block in ("body", "orelse", "finalbody"):
            statements = getattr(node, block, None)
            if not isinstance(statements, list):
                continue
            lines = [s.lineno for s in statements if isinstance(s, ast.stmt)]
            found += [
                f"line {n}: two statements"
                for n in sorted({n for n in lines if lines.count(n) > 1})
            ]
    return found


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a dotted name whose parent package is missing
        return False


def _missing(recipe: Recipe) -> list[str]:
    return [name for name in recipe.requires if not _installed(name)]


def _run(recipe: Recipe) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "agent_guide_recipe"}
    exec(compile(PRELUDE, "<prelude>", "exec"), namespace)
    if recipe.setup:
        exec(compile(recipe.setup, "<setup>", "exec"), namespace)
    code = compile(recipe.code, "<recipe>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if inspect.iscoroutine(result):
        asyncio.run(result)
    return namespace


def _recipe(prefix: str) -> Recipe:
    return next(r for r in recipes.recipes() if r.want.startswith(prefix))


def test_agent_guide_openapi_recipe_gates_writes(tmp_path, monkeypatch) -> None:
    recipe = _recipe("import tools from an OpenAPI")
    missing = _missing(recipe)
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    monkeypatch.chdir(tmp_path)
    registry = _run(recipe)["registry"]
    assert registry.get("cancelBooking").confirm is True
    assert registry.get("getForecast").confirm is False


def test_agent_guide_secret_recipe_fails_fast_without_the_key(monkeypatch) -> None:
    monkeypatch.delenv("ORDERS_API_KEY", raising=False)
    with pytest.raises(KeyError, match="ORDERS_API_KEY"):
        _run(_recipe("keep a secret"))


@pytest.mark.parametrize("recipe", recipes.recipes(), ids=lambda r: r.want[:40])
def test_agent_guide_recipe_is_valid_and_runs(
    recipe: Recipe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert 1 <= sum(1 for line in recipe.lines if line.strip()) <= MAX_LINES
    assert all("\n" not in line for line in recipe.lines)
    compile(recipe.code, "<recipe>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    assert style_problems(recipe.code) == []
    api_problems(recipe.code)
    missing = _missing(recipe)
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed: compiled and API-checked only")
    monkeypatch.chdir(tmp_path)  # snippets that write files (SQLite, runs, cases) write here
    for name, value in recipe.env:
        monkeypatch.setenv(name, value)
    try:
        _run(recipe)
    finally:
        reset_default_executor()


def test_agent_guide_prelude_is_plain_style() -> None:
    assert style_problems(PRELUDE) == []


def test_agent_guide_checkers_catch_bad_snippets() -> None:
    with pytest.raises(AssertionError, match="no NotAThing"):
        api_problems("from agent_adapter.handlers import NotAThing")
    with pytest.raises(AssertionError, match="does not exist"):
        api_problems("aa.NotAThing()")
    with pytest.raises(AssertionError, match="unexpected keyword"):
        api_problems("aa.Executor(registry, approvals=None)")
    with pytest.raises(AssertionError, match="run_sync"):
        api_problems('executor.run_sync("t", {}, bogus=1)')
    api_problems('executor.run_sync("get_weather", {"city": "Oslo"}, timeout=30)')
    assert style_problems("def f(): return 1\nx = 1; y = 2\ntry: f()\nexcept Exception: pass")
    assert style_problems("def f():\n    return 1\n") == []
