"""Find framework tool definitions in one module's source, with `ast` only (never imported)."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from agent_adapter.cli import _migrate_rules as rules

# The registry's tool-name rule; a test keeps it equal to the one in core.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


@dataclass(frozen=True)
class Finding:
    """One tool definition found, and what to do about it."""

    line: int
    framework: str
    kind: str
    name: str
    action: str
    automatic: bool  # the suggested diff covers it


@dataclass(frozen=True)
class Edit:
    """Replace source from `start` to `end` (line, UTF-8 byte column) with `text`."""

    start: tuple[int, int]
    end: tuple[int, int]
    text: str
    framework: str
    alias: str = ""  # the imported name whose reference this edit removes
    imports: tuple[str, ...] = ()  # names it needs from `agent_adapter`
    finding: int = -1  # index of the finding it belongs to, while scanning


@dataclass
class Scan:
    """Every finding of one module and the edits the diff applies."""

    findings: list[Finding] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)


_LINE_END = re.compile(r"\r\n|\r|\n")


def split_lines(source: str) -> list[str]:
    """Lines with their ends, split the way `ast` numbers them (not `str.splitlines`)."""
    lines, start = [], 0
    for match in _LINE_END.finditer(source):
        lines.append(source[start : match.end()])
        start = match.end()
    if start < len(source):
        lines.append(source[start:])
    return lines


class _Module:
    """One parsed module: its source, its import aliases and its top-level functions."""

    def __init__(self, tree: ast.Module, source: str) -> None:
        self.tree = tree
        self.source = source
        self.lines = split_lines(source)
        self.encoded = [line.encode("utf-8") for line in self.lines]
        self.aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
                    else:
                        top = alias.name.split(".")[0]
                        self.aliases[top] = top
        self.functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        self.pydantic_agents = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.Call)
            and self.resolve(node.value.func) == rules.PYDANTIC_AI_AGENT
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }

    def resolve(self, node: ast.expr) -> str:
        """The dotted import path an expression names, or "" when it is not imported."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):  # iterative: chains can be thousands long
            parts.append(node.attr)
            node = node.value
        base = self.aliases.get(node.id, "") if isinstance(node, ast.Name) else ""
        return ".".join([base, *reversed(parts)]) if base else ""

    def segment(self, node: ast.expr) -> str:
        """The source text of `node`, cut from the already split lines."""
        first, last = node.lineno - 1, (node.end_lineno or node.lineno) - 1
        start, end = node.col_offset, node.end_col_offset or 0
        if first == last:
            return self.encoded[first][start:end].decode("utf-8")
        middle = self.encoded[first + 1 : last]
        chunks = [self.encoded[first][start:], *middle, self.encoded[last][:end]]
        return b"".join(chunks).decode("utf-8")

    def ending(self, line: int) -> str:
        """The line end this module uses on `line` (1-based), or LF when it has none."""
        text = self.lines[line - 1] if line <= len(self.lines) else ""
        return text[len(text.rstrip("\r\n")) :] or "\n"

    def at_sign(self, decorator: ast.expr) -> tuple[int, int] | None:
        """The (line, byte column) of the `@` in front of `decorator`, if only spaces between."""
        line = self.encoded[decorator.lineno - 1]
        before = line[: decorator.col_offset].rstrip()
        if not before.endswith(b"@") or before[:-1].strip():
            return None
        return decorator.lineno, len(before) - 1


def scan_module(tree: ast.Module, source: str, framework: str = "auto") -> Scan:
    """Every tool definition in `tree` (only `framework`'s unless "auto"), and the edits."""
    module = _Module(tree, source)
    scan = Scan()
    decorated: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _decorators(module, node, scan):
                decorated.add(node.name)
        elif isinstance(node, ast.ClassDef):
            _subclass(module, node, scan)
        elif isinstance(node, ast.Call):
            _wrapper(module, node, scan)
        elif isinstance(node, ast.Dict):
            _schema(node, scan)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _agent_tools(module, node, scan, decorated)
    _drop_overlaps(scan)
    if framework != "auto":
        _keep_only(scan, framework)
    scan.findings.sort(key=lambda finding: finding.line)
    return scan


def _drop_overlaps(scan: Scan) -> None:
    """Edits that overlap (nested wrappers) are dropped; their findings become manual."""
    dropped: set[int] = set()
    while True:
        kept = [edit for edit in scan.edits if edit.finding not in dropped]
        clash = _first_clash(kept)
        if clash is None:
            break
        dropped.add(clash)
    scan.edits[:] = kept
    action = "manual: nested in another rewritten call"
    for index in dropped:
        scan.findings[index] = replace(scan.findings[index], action=action, automatic=False)


def _first_clash(edits: list[Edit]) -> int | None:
    """The finding of the first edit that overlaps an earlier one, in source order."""
    end = (0, 0)
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        if edit.start < end:
            return edit.finding
        end = max(end, edit.end)
    return None


def _keep_only(scan: Scan, framework: str) -> None:
    """Drop the findings and edits of other frameworks."""
    scan.findings[:] = [f for f in scan.findings if f.framework == framework]
    scan.edits[:] = [e for e in scan.edits if e.framework == framework]


def _decorators(module: _Module, node: ast.FunctionDef | ast.AsyncFunctionDef, scan: Scan) -> bool:
    found = False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        framework = rules.DECORATORS.get(module.resolve(target), "")
        if not framework and _pydantic_ai_decorator(module, target):
            framework = "pydantic-ai"
        if not framework:
            continue
        found = True
        options = _decorator_options(decorator)
        at = module.at_sign(decorator)
        label = "@" + module.segment(target)
        if at is None or options is None:
            action = "manual: its options have no direct match"
            scan.findings.append(Finding(node.lineno, framework, label, node.name, action, False))
            continue
        end = (decorator.end_lineno or decorator.lineno, decorator.end_col_offset or 0)
        text = f"@registry.tool{options}"
        key = len(scan.findings)
        scan.edits.append(Edit(at, end, text, framework, _root(target), finding=key))
        action = "use @registry.tool"
        if _run_context(module, node, scan, framework, key):
            action += ", with `ctx: ToolContext` for its RunContext"
        scan.findings.append(Finding(node.lineno, framework, label, node.name, action, True))
    return found


def _run_context(
    module: _Module,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    scan: Scan,
    framework: str,
    key: int,
) -> bool:
    """Retype a first `ctx: RunContext[...]` parameter as `ToolContext`."""
    params = [*node.args.posonlyargs, *node.args.args]
    annotation = params[0].annotation if params else None
    if annotation is None:
        return False
    base = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    if module.resolve(base) != rules.PYDANTIC_AI_RUN_CONTEXT:
        return False
    start = (annotation.lineno, annotation.col_offset)
    end = (annotation.end_lineno or annotation.lineno, annotation.end_col_offset or 0)
    alias = _root(base)
    edit = Edit(start, end, "ToolContext", framework, alias, ("ToolContext",), finding=key)
    scan.edits.append(edit)
    return True


def _pydantic_ai_decorator(module: _Module, target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and target.attr in rules.PYDANTIC_AI_DECORATORS
        and isinstance(target.value, ast.Name)
        and target.value.id in module.pydantic_agents
    )


# Decorator keywords with a `@registry.tool` equivalent.
_KEYWORDS = {
    "name": "name",
    "name_override": "name",
    "description": "description",
    "description_override": "description",
}


def _decorator_options(decorator: ast.expr) -> str | None:
    """`(name="x")` for `@tool("x")`, "" for a bare decorator, None when options don't map."""
    if not isinstance(decorator, ast.Call):
        return ""
    options: dict[str, str] = {}
    if decorator.args:
        name = _string(decorator.args[0])
        if len(decorator.args) > 1 or name is None:
            return None
        options["name"] = name
    for keyword in decorator.keywords:
        option = _KEYWORDS.get(keyword.arg or "")
        value = _string(keyword.value)
        if option is None or value is None:
            return None
        options[option] = value
    if not NAME_RE.match(options.get("name", "_")):
        return None  # registry tool names are identifiers
    return _call_options(options, leading="")


def _wrapper_options(node: ast.Call, func_keyword: str) -> str | None:
    """`, name="x"` for a wrapper call's name and description; None when others are set."""
    options: dict[str, str] = {}
    for keyword in node.keywords:
        if keyword.arg == func_keyword:
            continue
        option = _KEYWORDS.get(keyword.arg or "")
        value = _string(keyword.value)
        if option is None or value is None:
            return None  # e.g. ADK's require_confirmation=: never drop a safety option
        options[option] = value
    if not NAME_RE.match(options.get("name", "_")):
        return None  # registry tool names are identifiers
    return _call_options(options, leading=", ")


def _call_options(options: dict[str, str], *, leading: str) -> str:
    """`(name="x")` (or `, name="x"` with `leading=", "`) as Python source; "" for none."""
    if not options:
        return ""
    text = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in options.items()
    )
    return f"{leading}{text}" if leading else f"({text})"


def _wrapper(module: _Module, node: ast.Call, scan: Scan) -> None:
    rule = rules.WRAPPERS.get(module.resolve(node.func))
    if rule is None:
        return
    framework, keyword = rule
    func = node.args[0] if node.args else _keyword(node, keyword)
    label = module.segment(node.func)
    options = _wrapper_options(node, keyword)
    if func is None or len(node.args) > 1 or options is None:
        action = "manual: its options have no direct match; use registry.wrap(...)"
        tool = module.segment(func) if func is not None else "?"
        scan.findings.append(Finding(node.lineno, framework, label, tool, action, False))
        return
    name = _string(_keyword(node, "name"))
    start = (node.lineno, node.col_offset)
    end = (node.end_lineno or node.lineno, node.end_col_offset or 0)
    text = f"registry.wrap({module.segment(func)}{options})"
    key = len(scan.findings)
    scan.edits.append(Edit(start, end, text, framework, _root(node.func), finding=key))
    tool = name or module.segment(func)
    scan.findings.append(Finding(node.lineno, framework, label, tool, "use registry.wrap", True))


def _agent_tools(module: _Module, node: ast.Call, scan: Scan, decorated: set[str]) -> None:
    """Plain functions in an agent's `tools=[...]` get `@registry.tool` on their `def`."""
    framework = rules.AGENTS.get(module.resolve(node.func))
    tools = _keyword(node, "tools")
    if framework is None or not isinstance(tools, ast.List | ast.Tuple):
        return
    for item in tools.elts:
        if not isinstance(item, ast.Name) or item.id in decorated:
            continue
        func = module.functions.get(item.id)
        if func is None:
            continue
        decorated.add(item.id)
        first = min([func.lineno, *(d.lineno for d in func.decorator_list)])
        indent = " " * func.col_offset
        text = f"{indent}@registry.tool{module.ending(first)}"
        key = len(scan.findings)
        scan.edits.append(Edit((first, 0), (first, 0), text, framework, finding=key))
        action = "add @registry.tool"
        scan.findings.append(Finding(func.lineno, framework, "tools=[...]", item.id, action, True))


def _subclass(module: _Module, node: ast.ClassDef, scan: Scan) -> None:
    for base in node.bases:
        framework = rules.SUBCLASSES.get(module.resolve(base))
        if framework is None:
            continue
        name = _class_name(node) or node.name
        action = "manual: move its run method into a @registry.tool function"
        label = f"{module.segment(base)} subclass"
        scan.findings.append(Finding(node.lineno, framework, label, name, action, False))
        return


def _schema(node: ast.Dict, scan: Scan) -> None:
    """Hand-written OpenAI or Anthropic tool schemas (dict literals)."""
    items = {key: value for key, value in _items(node)}
    name = _string(items.get("name"))
    if _string(items.get("type")) == "function":
        inner = items.get("function")
        if isinstance(inner, ast.Dict):
            name = _string(dict(_items(inner)).get("name"))
        elif "parameters" not in items:
            return
        framework = "openai"
    elif "input_schema" in items:
        framework = "anthropic"
    else:
        return
    if name is None:
        return
    action = "manual: write it as a @registry.tool function; render with the bridge"
    scan.findings.append(Finding(node.lineno, framework, "tool schema", name, action, False))


def _items(node: ast.Dict) -> Iterator[tuple[str, ast.expr]]:
    for key, value in zip(node.keys, node.values, strict=True):
        text = _string(key)
        if text is not None:
            yield text, value


def _class_name(node: ast.ClassDef) -> str | None:
    """The `name = "..."` a tool class declares in its body."""
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "name":
            return _string(value)
    return None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in node.keywords if k.arg == name), None)


def _string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _root(target: ast.expr) -> str:
    """The name at the root of `target` (`FunctionTool` in `FunctionTool.from_defaults`)."""
    while isinstance(target, ast.Attribute):
        target = target.value
    return target.id if isinstance(target, ast.Name) else ""
