"""`migrate` on odd inputs: deep chains, huge files, nested wrappers, safety options, line ends."""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import pytest

from agent_adapter.cli import migrate
from agent_adapter.cli._migrate_diff import suggested, suggested_diff
from agent_adapter.cli._migrate_scan import scan_module, split_lines

FIXTURES = Path(__file__).resolve().parent / "migrate_fixtures"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)


def _apply(source: str, diff: str) -> str:
    """Apply a unified diff to `source` the way `patch` does (checks every old line)."""
    old = split_lines(source)
    lines = split_lines(diff)[2:]
    out: list[str] = []
    cursor, index = 0, 0
    while index < len(lines):
        match = HUNK.match(lines[index])
        assert match, lines[index]
        start = int(match.group(1)) - (0 if match.group(2) == "0" else 1)
        out += old[cursor:start]
        cursor = start
        index += 1
        while index < len(lines) and not lines[index].startswith("@@"):
            tag, text = lines[index][0], lines[index][1:]
            if index + 1 < len(lines) and lines[index + 1].startswith("\\"):
                text = text.rstrip("\n")  # the old line had no line end
                index += 1
            if tag in " -":
                assert old[cursor] == text, (old[cursor], text)
                cursor += 1
            if tag in " +":
                out.append(text)
            index += 1
    return "".join(out + old[cursor:])


def _both(source: str, framework: str = "auto") -> tuple[str, str]:
    tree = ast.parse(source)
    scan = scan_module(tree, source, framework)
    return suggested(tree, source, scan), suggested_diff("m.py", tree, source, scan)


@pytest.mark.parametrize("fixture", sorted(p.name for p in FIXTURES.glob("[!b]*.fixture")))
def test_the_diff_applies_and_gives_the_suggested_text(fixture):
    source = (FIXTURES / fixture).read_text(encoding="utf-8")
    new, diff = _both(source)
    if new != source:
        assert _apply(source, diff) == new
        ast.parse(new)


def test_a_long_attribute_chain_is_scanned_without_recursion():
    chain = "langchain_core" + ".a" * 2000
    source = f"import langchain_core\n@{chain}\ndef f() -> int:\n    return 1\n"
    report = migrate.scan_source(source.encode(), "deep.py")
    assert report.skipped == "" and report.findings == ()


def test_a_tree_too_deep_to_walk_skips_only_that_file(tmp_path, monkeypatch):
    (tmp_path / "a_deep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b_ok.py").write_text(
        "from agents import function_tool\n@function_tool\ndef f() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    real = migrate.scan_module

    def deep(tree, source, framework="auto"):
        if source.startswith("x = 1"):
            raise RecursionError
        return real(tree, source, framework)

    monkeypatch.setattr(migrate, "scan_module", deep)
    reports = migrate.scan_path(tmp_path)
    assert reports[0].skipped == "too deeply nested to scan (RecursionError)"
    assert [f.name for f in reports[1].findings] == ["f"]


def test_a_large_file_is_scanned_in_linear_time():
    tools = "".join(f"@tool\ndef t{i}(x: int) -> int:\n    return x\n\n\n" for i in range(4000))
    source = "from langchain_core.tools import tool\n\n\n" + tools
    began = time.perf_counter()
    report = migrate.scan_source(source.encode(), "big.py")
    assert time.perf_counter() - began < 10  # quadratic took 14 s here, 30,000 tools never ended
    assert len(report.findings) == 4000 and report.diff.count("+@registry.tool\n") == 4000


def test_a_nested_wrapper_is_marked_manual_when_its_edit_is_dropped():
    source = (
        "from google.adk.tools import FunctionTool\n"
        "from langchain_core.tools import StructuredTool\n"
        "def f() -> int:\n    return 1\n"
        "t = FunctionTool(func=StructuredTool.from_function(f))\n"
    )
    new, diff = _both(source)
    by_kind = {f.kind: f for f in scan_module(ast.parse(source), source).findings}
    assert by_kind["FunctionTool"].automatic
    assert not by_kind["StructuredTool.from_function"].automatic
    assert by_kind["StructuredTool.from_function"].action.startswith("manual: nested")
    assert "t = registry.wrap(StructuredTool.from_function(f))" in new
    assert "from langchain_core.tools import StructuredTool\n" in new  # still used
    assert _apply(source, diff) == new


@pytest.mark.parametrize(
    "source",
    [
        "from agno.tools import tool\n@tool(requires_confirmation=True)\ndef f() -> int:\n"
        "    return 1\n",
        "from agents import function_tool\n@function_tool(needs_approval=True)\ndef f() -> int:\n"
        "    return 1\n",
        "from pydantic_ai import Agent\nagent = Agent('m')\n@agent.tool(requires_approval=True)\n"
        "def f(ctx) -> int:\n    return 1\n",
        "from strands import tool\n@tool(context=True)\ndef f() -> int:\n    return 1\n",
    ],
    ids=["agno", "openai-agents", "pydantic-ai", "strands"],
)
def test_approval_and_context_options_stay_manual(source):
    new, diff = _both(source)
    (finding,) = scan_module(ast.parse(source), source).findings
    assert not finding.automatic and new == source and diff == ""


def test_crlf_files_get_crlf_in_every_added_line():
    source = (
        "from langchain_core.tools import tool\r\n\r\n@tool\r\ndef f() -> int:\r\n    return 1\r\n"
    )
    new, diff = _both(source)
    assert "\n" not in new.replace("\r\n", "")
    assert "registry = Registry()\r\n" in new
    assert _apply(source, diff) == new


def test_a_file_without_a_final_line_end_gets_a_correct_diff():
    source = "from agents import function_tool\n@function_tool\ndef f() -> int:\n    return 1"
    new, diff = _both(source)
    assert new.endswith("    return 1\n")
    assert "\\ No newline at end of file\n" in diff
    assert _apply(source, diff) == new


def test_hunks_far_apart_stay_separate_and_count_right():
    middle = "".join(f"x{i} = {i}\n" for i in range(20))
    source = (
        "from agents import function_tool\n"
        "@function_tool\ndef a() -> int:\n    return 1\n"
        + middle
        + "@function_tool\ndef b() -> int:\n    return 2\n"
    )
    new, diff = _both(source)
    assert len(HUNK.findall(diff)) == 2
    assert _apply(source, diff) == new


def test_migrate_name_rule_matches_the_registry() -> None:
    from agent_adapter.cli import _migrate_scan
    from agent_adapter.core import _signature

    assert _migrate_scan.NAME_RE.pattern == _signature.NAME_RE.pattern
