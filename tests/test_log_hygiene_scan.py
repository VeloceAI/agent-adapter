"""Scan agent_adapter/ so no WARNING+ log call carries exception text or a traceback."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "agent_adapter"
LOUD_METHODS = frozenset({"warning", "warn", "error", "exception", "critical", "fatal"})
LOUD_LEVELS = frozenset({"WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"})
EXCEPTION_PARAM_NAMES = frozenset({"exc", "error", "err"})
# Structured fields that name a class, a location or an id, never free text or data.
SAFE_ATTRIBUTES = frozenset(
    {
        "__class__",
        "kind",
        "name",  # ImportError: the missing module
        "path",  # NotSealable: where in the value, e.g. "$.items[0]"
        "type_name",
        "problems",  # SchemaNotPortable: adapter-written schema findings
        "model",
        "status_code",
        "request_id",
        "approval_id",
        "call_id",
        "tool",
        "stage",
        "run_id",
    }
)


def _is_logger(node: ast.expr) -> bool:
    name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
    return "log" in name.lower()


def _is_loud(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or not _is_logger(func.value):
        return False
    if func.attr in LOUD_METHODS:
        return True
    if func.attr == "log" and call.args:
        level = call.args[0]
        name = level.attr if isinstance(level, ast.Attribute) else getattr(level, "id", "")
        return name in LOUD_LEVELS
    return False


def _exception_params(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = func.args
    names = set()
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        hint = ast.unparse(arg.annotation) if arg.annotation else ""
        if arg.arg in EXCEPTION_PARAM_NAMES or "Exception" in hint or "Error" in hint:
            names.add(arg.arg)
    return names


def _leaks(node: ast.AST, names: set[str]) -> bool:
    """True when `node` reads an exception other than through `type(...)` or a safe attribute."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "type":
        return False
    if isinstance(node, ast.Attribute) and node.attr in SAFE_ATTRIBUTES:
        return False
    if isinstance(node, ast.Name):
        return node.id in names
    return any(_leaks(child, names) for child in ast.iter_child_nodes(node))


def _offence(call: ast.Call, names: set[str]) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "exc_info" and not (
            isinstance(keyword.value, ast.Constant) and not keyword.value.value
        ):
            return "passes exc_info"
    if getattr(call.func, "attr", "") == "exception":
        return "logger.exception adds a traceback"
    if any(_leaks(arg, names) for arg in call.args):
        return "passes exception text"
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.names: set[str] = set()
        self.found: list[str] = []

    def _scoped(self, node: ast.AST, extra: set[str]) -> None:
        saved = self.names
        self.names = saved | extra
        self.generic_visit(node)
        self.names = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, _exception_params(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, _exception_params(node))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._scoped(node, {node.name} if node.name else set())

    def visit_Call(self, node: ast.Call) -> None:
        if _is_loud(node):
            reason = _offence(node, self.names)
            if reason:
                where = self.path.relative_to(PACKAGE.parent).as_posix()
                self.found.append(f"{where}:{node.lineno}: {reason}")
        self.generic_visit(node)


def scan(path: Path) -> list[str]:
    """Offending log calls in one source file, as `path:line: why`."""
    scanner = _Scanner(path)
    scanner.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return scanner.found


def _sources() -> Iterator[Path]:
    yield from sorted(PACKAGE.rglob("*.py"))


def test_log_hygiene_no_warning_or_error_call_logs_exception_text():
    found = [line for path in _sources() for line in scan(path)]
    assert found == []


SNIPPETS = [
    ("except E as exc:\n    logger.warning('x %s', exc)", True),
    ("except E as exc:\n    logger.error('x %s', str(exc))", True),
    ("except E as exc:\n    log.warning('x %s', exc.reason)", True),
    ("except E as exc:\n    logger.error(f'x {exc}')", True),
    ("except E as exc:\n    logger.log(logging.ERROR, 'x %s', exc)", True),
    ("except E:\n    logger.warning('x', exc_info=True)", True),
    ("except E:\n    logger.exception('x')", True),
    ("except E as exc:\n    logger.warning('x %s', type(exc).__name__)", False),
    ("except E as exc:\n    logger.warning('x %s', qualified_name(type(exc)))", False),
    ("except E as exc:\n    logger.debug('x %s', exc, exc_info=exc)", False),
    ("except E as exc:\n    logger.info('x %s', exc)", False),
    ("except E as exc:\n    log_failure(logger, logging.WARNING, exc, 'x')", False),
    ("except E as exc:\n    warnings.warn(str(exc))", False),
]


@pytest.mark.parametrize(("body", "flagged"), SNIPPETS)
def test_scan_flags_exception_text_only_at_warning_and_above(body, flagged):
    source = PACKAGE / "_scan_probe.py"
    code = "def f():\n    try:\n        pass\n    " + body.replace("\n", "\n    ") + "\n"
    scanner = _Scanner(source)
    scanner.visit(ast.parse(code))
    assert bool(scanner.found) is flagged


def test_scan_treats_exception_parameters_as_exceptions():
    code = "def f(self, failure: BaseException):\n    logger.error('x %s', failure)\n"
    scanner = _Scanner(PACKAGE / "_scan_probe.py")
    scanner.visit(ast.parse(code))
    assert scanner.found == ["agent_adapter/_scan_probe.py:2: passes exception text"]
