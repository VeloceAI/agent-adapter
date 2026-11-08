"""Turn a module's scan into suggested source text and a unified diff (nothing is written).

The diff is built from the edits themselves, region by region, so it costs time in
proportion to the file, never a line-by-line comparison of the old and new text.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass

from agent_adapter.cli import _migrate_rules as rules
from agent_adapter.cli._migrate_scan import Edit, Scan, split_lines

EXECUTOR_LINE = "executor = Executor(registry)  # add approval=, hooks= and limits= here"
CONTEXT = 3  # unchanged lines around each hunk, as `diff -u`
NO_NEWLINE = "\\ No newline at end of file\n"


@dataclass(frozen=True)
class _Region:
    """Old lines `first` to `last` (0-based, `last` exclusive) become `new`."""

    first: int
    last: int
    new: list[str]


def suggested(tree: ast.Module, source: str, scan: Scan) -> str:
    """`source` with the scan's edits, the unused framework imports and a registry header."""
    lines, regions = _plan(tree, source, scan)
    out: list[str] = []
    cursor = 0
    for region in regions:
        out += [*lines[cursor : region.first], *region.new]
        cursor = region.last
    out += lines[cursor:]
    return "".join(out)


def suggested_diff(rel: str, tree: ast.Module, source: str, scan: Scan) -> str:
    """A `git apply`-style unified diff from `a/rel` to `b/rel`; "" when nothing changes."""
    lines, regions = _plan(tree, source, scan)
    if not regions:
        return ""
    out = [f"--- a/{rel}\n", f"+++ b/{rel}\n"]
    shift = 0  # new line index minus old line index, before the current hunk
    for group in _hunks(regions):
        start = max(0, group[0].first - CONTEXT)
        stop = min(len(lines), group[-1].last + CONTEXT)
        delta = sum(len(r.new) - (r.last - r.first) for r in group)
        old, new = _range(start, stop - start), _range(start + shift, stop - start + delta)
        out.append(f"@@ -{old} +{new} @@\n")
        cursor = start
        for region in group:
            out += [f" {line}" for line in lines[cursor : region.first]]
            out += [_removed(line) for line in lines[region.first : region.last]]
            out += [f"+{line}" for line in region.new]
            cursor = region.last
        out += [f" {line}" for line in lines[cursor:stop]]
        shift += delta
    return "".join(out)


def _plan(tree: ast.Module, source: str, scan: Scan) -> tuple[list[str], list[_Region]]:
    """The source lines and the changed regions, in order."""
    lines = split_lines(source)
    if not scan.edits:
        return lines, []
    eol = _line_end(lines)
    extra: list[Edit] = []
    if not lines[-1].endswith(("\n", "\r")):  # the suggestion ends with a line end
        end = (len(lines), len(lines[-1].encode("utf-8")))
        extra.append(Edit(end, end, eol, ""))
    frameworks = {edit.framework for edit in scan.edits}
    needs = {name for edit in scan.edits for name in edit.imports}
    converted = Counter(edit.alias for edit in scan.edits if edit.alias)
    extra += _header(tree, frameworks, needs, eol)
    extra += _import_edits(tree, lines, converted, eol)
    return lines, _regions(lines, [*extra, *scan.edits])


def _regions(lines: list[str], edits: list[Edit]) -> list[_Region]:
    """Group the edits by the whole old lines they touch; touching groups merge."""
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line.encode("utf-8")))
    spans: list[tuple[int, int, list[Edit]]] = []
    for edit in sorted(edits, key=lambda e: e.start):  # stable: ties keep list order
        first, last = _span(edit, len(lines))
        if spans and first <= spans[-1][1]:
            prev_first, prev_last, group = spans[-1]
            spans[-1] = (prev_first, max(prev_last, last), [*group, edit])
        else:
            spans.append((first, last, [edit]))
    data = "".join(lines).encode("utf-8")
    regions = []
    for first, last, group in spans:
        text = _splice(data, starts, starts[first], starts[last], group)
        regions.append(_Region(first, last, split_lines(text)))
    return regions


def _span(edit: Edit, count: int) -> tuple[int, int]:
    """The old lines (0-based, end exclusive) an edit rewrites; empty for a line insertion."""
    (start_line, start_col), (end_line, end_col) = edit.start, edit.end
    at_line_start = end_col == 0 and (end_line > start_line or start_col == 0)
    last = end_line - 1 if at_line_start else end_line
    return min(start_line - 1, count), min(last, count)


def _splice(data: bytes, starts: list[int], low: int, high: int, edits: list[Edit]) -> str:
    """The text of bytes `low` to `high` with `edits` applied (they do not overlap)."""

    def offset(position: tuple[int, int]) -> int:
        line, column = position
        return min(starts[min(line, len(starts)) - 1] + column, len(data))

    out: list[bytes] = []
    cursor = low
    for edit in edits:
        start = offset(edit.start)
        out += [data[cursor:start], edit.text.encode("utf-8")]
        cursor = max(cursor, offset(edit.end))
    out.append(data[cursor:high])
    return b"".join(out).decode("utf-8")


def _hunks(regions: list[_Region]) -> list[list[_Region]]:
    """Regions close enough to share one hunk's context."""
    groups: list[list[_Region]] = []
    for region in regions:
        if groups and region.first - groups[-1][-1].last <= 2 * CONTEXT:
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def _range(start: int, length: int) -> str:
    """A hunk range as `diff -u` writes it (0-based `start`)."""
    if length == 1:
        return str(start + 1)
    return f"{start + 1 if length else start},{length}"


def _removed(line: str) -> str:
    return f"-{line}" if line.endswith(("\n", "\r")) else f"-{line}\n{NO_NEWLINE}"


def _line_end(lines: list[str]) -> str:
    """The module's line end, from its first line (LF for a one-line file without one)."""
    first = lines[0]
    return first[len(first.rstrip("\r\n")) :] or "\n"


def _header(tree: ast.Module, frameworks: set[str], needs: set[str], eol: str) -> list[Edit]:
    """`from agent_adapter import ...`, the bridge imports and `registry = Registry()`."""
    bound = _top_level_names(tree)
    names = sorted(name for name in {"Executor", "Registry", *needs} if name not in bound)
    lines = [f"from agent_adapter import {', '.join(names)}"] if names else []
    for framework in sorted(frameworks):
        bridge = rules.BRIDGES[framework]
        lines.append(f"from {bridge.module} import {bridge.function}")
    setup = []
    if "registry" not in bound:
        setup.append("registry = Registry()")
    if "executor" not in bound:
        setup.append(EXECUTOR_LINE)
    text = "".join(line + eol for line in lines)
    if setup:
        text += eol + "".join(line + eol for line in setup)
    at = (_after_imports(tree), 0)
    return [Edit(at, at, text, "")]


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _after_imports(tree: ast.Module) -> int:
    """The line after the leading imports (or the docstring); 1 for an empty prelude."""
    line = 1
    for index, node in enumerate(tree.body):
        is_doc = index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_doc or isinstance(node, ast.Import | ast.ImportFrom)):
            break
        line = (node.end_lineno or node.lineno) + 1
    return line


def _import_edits(
    tree: ast.Module, lines: list[str], converted: Counter[str], eol: str
) -> list[Edit]:
    """Drop imported names whose every reference the edits replace."""
    uses = Counter(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
    edits = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module or node.level:
            continue
        keep = [
            a
            for a in node.names
            if not 0 < uses[a.asname or a.name] <= converted[a.asname or a.name]
        ]
        if len(keep) == len(node.names) or not _whole_lines(node, lines):
            continue
        end = node.end_lineno or node.lineno
        text = ""
        if keep:
            names = ", ".join(f"{a.name} as {a.asname}" if a.asname else a.name for a in keep)
            text = f"from {node.module} import {names}{eol}"
        edits.append(Edit((node.lineno, 0), (end + 1, 0), text, ""))
    return edits


def _whole_lines(node: ast.stmt, lines: list[str]) -> bool:
    """Whether `node` alone fills its lines (no `;` neighbour, no trailing code)."""
    last = lines[(node.end_lineno or node.lineno) - 1].encode("utf-8")
    rest = last[node.end_col_offset or 0 :].strip()
    return node.col_offset == 0 and (not rest or rest.startswith(b"#"))
