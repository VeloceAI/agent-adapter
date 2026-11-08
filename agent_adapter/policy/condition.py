"""`Condition`: a tiny, safe expression on tool arguments, such as `amount > 100`; never eval."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

from agent_adapter.hooks.builtin.logging import is_secret_name

MAX_CHARS = 500
MAX_NODES = 100
_WORDS = {"true": True, "false": False, "null": None, "none": None}
_ORDER: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_EQUAL: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Constant,
    ast.List,
    ast.Tuple,
    *_ORDER,
    *_EQUAL,
    ast.In,
    ast.NotIn,
)


class ConditionError(ValueError):
    """A condition cannot be parsed, or cannot be decided for these arguments."""


class Condition:
    """A parsed condition: comparisons (`== != < <= > >= in not in`), `and`, `or`, `not`.

    Operands are argument names (dotted for nested fields), numbers, strings, `true`,
    `false`, `null` and lists of those. Secret-named arguments cannot be read.

    Raises:
        ConditionError: anything outside that grammar, or a secret-named argument.
    """

    def __init__(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ConditionError("a condition must be a non-empty string")
        if len(text) > MAX_CHARS:
            raise ConditionError(f"a condition is at most {MAX_CHARS} characters")
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            raise ConditionError(f"not a valid condition: {exc.msg}") from None
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_NODES:
            raise ConditionError(f"a condition has at most {MAX_NODES} parts")
        for node in nodes:
            if not isinstance(node, _ALLOWED):
                raise ConditionError(f"{type(node).__name__} is not allowed in a condition")
            _check_node(node)
        if not _decides(tree.body):
            raise ConditionError("a condition is a comparison, or several joined by and, or, not")
        self.text = text.strip()
        self._tree = tree.body
        self.fields = frozenset(_fields(tree.body))

    def __repr__(self) -> str:
        return f"Condition({self.text!r})"

    def shown(self) -> str:
        """The text with every string literal as '...', for explanations and logs."""
        hidden = _Hide().visit(ast.parse(self.text, mode="eval"))
        return ast.unparse(hidden)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Condition) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)

    def evaluate(self, args: Mapping[str, Any]) -> bool:
        """True when the arguments meet the condition.

        Raises:
            ConditionError: a missing argument, or values that cannot be compared.
        """
        try:
            value = _eval(self._tree, args)
        except ConditionError:
            raise
        except (ArithmeticError, TypeError, ValueError, RecursionError) as exc:
            # Model-chosen values (a huge int, an odd type) must never escape as raw errors.
            raise ConditionError(f"cannot decide ({type(exc).__name__})") from None
        if not isinstance(value, bool):
            raise ConditionError("the condition does not give true or false")
        return value


def _check_node(node: ast.AST) -> None:
    if isinstance(node, ast.Constant) and not (
        node.value is None or isinstance(node.value, bool | int | float | str)
    ):
        raise ConditionError("only numbers, strings, true, false and null are allowed")
    if isinstance(node, ast.Name | ast.Attribute):
        path = _path(node)
        if any(part.startswith("_") for part in path):
            raise ConditionError("argument names cannot start with '_'")
        if any(is_secret_name(part) for part in path):
            raise ConditionError(f"a condition cannot read the secret argument {'.'.join(path)!r}")
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and not (isinstance(node.operand, ast.Constant) and _number(node.operand.value))
    ):
        raise ConditionError("'-' applies to numbers only")
    if isinstance(node, ast.List | ast.Tuple) and not all(
        isinstance(item, ast.Constant | ast.UnaryOp) for item in node.elts
    ):
        raise ConditionError("a list holds numbers, strings, true, false or null only")


def _decides(node: ast.AST) -> bool:
    """A root that gives true or false: a comparison, `and`/`or`, or `not`."""
    return isinstance(node, ast.Compare | ast.BoolOp) or (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
    )


class _Hide(ast.NodeTransformer):
    """Replaces string literals, which may hold personal data, with '...'."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.Constant("...") if isinstance(node.value, str) else node


def _path(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_path(node.value), node.attr)
    raise ConditionError("a dotted name must start with an argument name")


def _fields(node: ast.AST) -> list[str]:
    found: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id.lower() not in _WORDS:
            found.append(child.id)
    return found


def _eval(node: ast.AST, args: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        values = (_truth(_eval(item, args)) for item in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, args)
        return not _truth(value) if isinstance(node.op, ast.Not) else -value
    if isinstance(node, ast.Compare):
        left = _eval(node.left, args)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _eval(right_node, args)
            if not _compare(op, left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List | ast.Tuple):
        return [_eval(item, args) for item in node.elts]
    if isinstance(node, ast.Name) and node.id.lower() in _WORDS:
        return _WORDS[node.id.lower()]
    return _lookup(_path(node), args)


def _lookup(path: tuple[str, ...], args: Mapping[str, Any]) -> Any:
    value: Any = args
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ConditionError(f"argument {'.'.join(path)!r} is missing")
        value = value[part]
    return value


def _truth(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConditionError("'and', 'or' and 'not' need true or false on each side")
    return value


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    if type(op) in _EQUAL:
        if (_number(left) and isinstance(right, str)) or (isinstance(left, str) and _number(right)):
            raise ConditionError("'==' compares a number with a string")
        return _EQUAL[type(op)](left, right)
    if type(op) in _ORDER:
        both_numbers = _number(left) and _number(right)
        if not (both_numbers or (isinstance(left, str) and isinstance(right, str))):
            raise ConditionError("'<' and '>' compare two numbers or two strings")
        if any(isinstance(side, float) and math.isnan(side) for side in (left, right)):
            raise ConditionError("cannot order NaN")
        return _ORDER[type(op)](left, right)
    if not isinstance(right, list | tuple | str):
        raise ConditionError("'in' needs a list or a string on its right")
    if isinstance(right, str) and not isinstance(left, str):
        raise ConditionError("'in' a string needs a string on its left")
    found = left in right
    return found if isinstance(op, ast.In) else not found
