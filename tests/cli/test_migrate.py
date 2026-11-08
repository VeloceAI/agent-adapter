"""`agent-adapter migrate`: finds framework tools with `ast`, suggests a diff, writes nothing."""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import os
import shutil
from pathlib import Path

import pytest

from agent_adapter.cli import migrate
from agent_adapter.cli._migrate_diff import suggested
from agent_adapter.cli._migrate_scan import scan_module, split_lines
from agent_adapter.errors import ConfigError

FIXTURES = Path(__file__).resolve().parent / "migrate_fixtures"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-adapter")
    migrate.add_migrate_parser(parser.add_subparsers(dest="command"))
    return parser


def _cli(*argv: str) -> str:
    out = io.StringIO()
    assert migrate.run_migrate(_parser().parse_args(["migrate", *argv]), out) == 0
    return out.getvalue()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """The fixture sources as a project: `.py` files, one broken file in a sub-folder."""
    root = tmp_path / "project"
    (root / "sub").mkdir(parents=True)
    for fixture in FIXTURES.glob("*.py.fixture"):
        target = root / ("sub" if fixture.name.startswith("broken") else "") / fixture.stem
        shutil.copyfile(fixture, target)
    return root


def _new(name: str, framework: str = "auto") -> str:
    source = (FIXTURES / f"{name}.py.fixture").read_text(encoding="utf-8")
    tree = ast.parse(source)
    new = suggested(tree, source, scan_module(tree, source, framework))
    ast.parse(new)  # the suggestion is valid Python
    return new


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_migrate_never_writes_a_file(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = _snapshot(tmp_path)
    real_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        assert not set(mode) & set("wax+"), f"migrate opened {file} with mode {mode!r}"
        return real_open(file, mode, *args, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("migrate tried to write")

    monkeypatch.setattr(builtins, "open", read_only_open)
    monkeypatch.setattr(Path, "write_text", refuse)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    monkeypatch.setattr(os, "replace", refuse)
    out = _cli(str(project), "--diff")
    assert out.startswith("agent-adapter migrate: read-only, no file was changed\n")
    assert "+@registry.tool" in out
    assert _snapshot(tmp_path) == before


def test_migrate_reports_every_framework_and_the_bridges(project):
    lines = _cli(str(project)).splitlines()
    assert "adk_agent.py:7: adk tools=[...] `seats_left`: add @registry.tool" in lines
    assert "adk_agent.py:21: adk FunctionTool `book_seat`: use registry.wrap" in lines
    assert any(
        line.startswith("crewai_crew.py:6: crewai BaseTool subclass `opening_hours`: manual")
        for line in lines
    )
    assert "langchain_app.py:10: langchain @tool `ticket_status`: use @registry.tool" in lines
    assert (
        "openai_agents_app.py:11: openai-agents @function_tool `track`: use @registry.tool" in lines
    )
    assert any("pydantic-ai @agent.tool `order_total`" in line for line in lines)
    assert any("anthropic tool schema `return_item`: manual" in line for line in lines)
    assert any("openai tool schema `cancel_order`" in line for line in lines)
    assert any("llamaindex FunctionTool.from_defaults `gift_wrap_order`" in line for line in lines)
    assert "sub/broken.py: skipped, syntax error at line 5" in lines
    assert "15 tool definition(s) in 6 file(s); 6 need a manual step" in lines
    assert (
        "langchain: give it structured_tools(registry, executor=executor) from "
        "agent_adapter.integrations.langchain" in lines
    )
    assert "openai: give it chat_tools(registry) from agent_adapter.integrations.openai" in lines


def test_migrate_ignores_schema_lookalikes():
    source = 'X = {"type": "function", "label": "no name"}\nY = {"name": "n"}\n'
    tree = ast.parse(source)
    assert scan_module(tree, source).findings == []


def test_langchain_diff_swaps_decorators_and_keeps_the_import_still_in_use():
    new = _new("langchain_app")
    assert "@registry.tool\ndef ticket_status" in new
    assert '@registry.tool(name="refund_ticket")\ndef refund' in new
    assert "@tool(return_direct=True)\ndef escalate" in new  # no match: left for a person
    assert "from langchain_core.tools import tool\n" in new  # still used by escalate
    assert "from agent_adapter.integrations.langchain import structured_tools\n" in new
    assert "registry = Registry()\nexecutor = Executor(registry)" in new


def test_openai_agents_diff_drops_only_the_replaced_import_name():
    new = _new("openai_agents_app")
    assert new.startswith(
        "from agents import Agent\nfrom agent_adapter import Executor, Registry\n"
    )
    assert '@registry.tool(name="track_parcel")\ndef track' in new


def test_adk_diff_wraps_function_tools_and_decorates_plain_functions():
    new = _new("adk_agent")
    assert "FunctionTool" not in new
    assert "registry.wrap(book_seat)," in new
    assert "@registry.tool\ndef seats_left" in new


def test_pydantic_ai_diff_retypes_the_run_context():
    new = _new("pydantic_ai_agent")
    assert "def order_total(ctx: ToolContext, order_id: str)" in new
    assert "from pydantic_ai import Agent\n" in new
    assert "from agent_adapter import Executor, Registry, ToolContext\n" in new


def test_llamaindex_diff_keeps_the_tool_name():
    new = _new("raw_schemas")
    assert 'wrap_tool = registry.wrap(gift_wrap, name="gift_wrap_order")' in new
    assert "from llama_index" not in new
    assert "find_order" in new  # schemas are reported, never rewritten


def test_framework_filter_keeps_only_that_framework(project):
    out = _cli(str(project), "--framework", "anthropic")
    assert "return_item" in out and "langchain" not in out
    assert "1 tool definition(s) in 1 file(s); 1 need a manual step" in out
    assert "openai_agents" not in _new("openai_agents_app", "langchain")


def test_a_wrapper_with_a_safety_option_is_left_manual():
    source = (
        "from google.adk.tools import FunctionTool\n"
        "def pay(order_id: str) -> str:\n    return order_id\n"
        "t = FunctionTool(func=pay, require_confirmation=True)\n"
    )
    tree = ast.parse(source)
    scan = scan_module(tree, source)
    assert [(f.automatic, f.action.split(":")[0]) for f in scan.findings] == [(False, "manual")]
    assert suggested(tree, source, scan) == source


def test_an_invalid_tool_name_is_left_manual():
    source = 'from crewai.tools import tool\n@tool("Ticket price")\ndef f() -> int:\n    return 1\n'
    tree = ast.parse(source)
    assert not scan_module(tree, source).findings[0].automatic


def test_an_existing_registry_is_reused():
    source = (
        "from agent_adapter import Executor, Registry\n"
        "from langchain_core.tools import tool\n"
        "registry = Registry()\nexecutor = Executor(registry)\n"
        "@tool\ndef f() -> int:\n    return 1"
    )
    tree = ast.parse(source)
    new = suggested(tree, source, scan_module(tree, source))
    assert new.count("registry = Registry()") == 1 and new.count("Executor(registry)") == 1
    assert "langchain_core" not in new.replace("integrations.langchain", "")
    assert new.endswith("@registry.tool\ndef f() -> int:\n    return 1\n")


def test_aliases_and_module_imports_resolve():
    source = (
        "import agents\nfrom langchain_core.tools import tool as lc_tool\n"
        "@agents.function_tool\ndef a() -> int:\n    return 1\n"
        "@lc_tool\ndef b() -> int:\n    return 2\n"
    )
    tree = ast.parse(source)
    found = {(f.framework, f.name) for f in scan_module(tree, source).findings}
    assert found == {("openai-agents", "a"), ("langchain", "b")}


EDGES = """
import crewai.tools as ct
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from langchain_core.tools import tool
from strands import tool as strands_tool
x: int


@(tool)
def spaced() -> int:
    return 1


@tool(7)
def numbered() -> int:
    return 1


@tool("a", "b")
def two_names() -> int:
    return 1


@tool(description=DESCRIPTION)
def computed() -> int:
    return 1


@strands_tool(description="Says hi.")
def hello() -> str:
    return "hi"


class Plain(ct.BaseTool):
    name = "plain_tool"


class Nameless(ct.BaseTool):
    description: str = "no name"
    size: int


lone = FunctionTool()
renamed = FunctionTool(func=hello, name="not valid")
agent = LlmAgent(tools=[hello, missing, other.attr, spaced])
"""


def test_edge_shapes_are_reported_and_only_safe_ones_rewritten():
    tree = ast.parse(EDGES)
    found = {(f.name, f.automatic) for f in scan_module(tree, EDGES).findings}
    assert found == {
        ("spaced", False),
        ("numbered", False),
        ("two_names", False),
        ("computed", False),
        ("hello", True),
        ("plain_tool", False),
        ("Nameless", False),
        ("?", False),
        ("hello", False),
    }
    new = suggested(tree, EDGES, scan_module(tree, EDGES))
    assert '@registry.tool(description="Says hi.")\ndef hello' in new
    assert "from langchain_core.tools import tool\n" in new  # still used by manual ones
    assert "from strands import tool as strands_tool" not in new


def test_a_file_without_tools_has_no_diff(tmp_path):
    (tmp_path / "plain.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    assert "0 tool definition(s) in 0 file(s); 0 need a manual step" in _cli(
        str(tmp_path), "--diff"
    )


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"def broken(:\n", "syntax error at line 1"),
        (b"x = 1\x00\n", "syntax error"),  # null bytes
        (b"\xff\xfe\x00junk", "not decodable as Python source"),
        (b"# -*- coding: no-such-codec -*-\nx = 1\n", "not decodable as Python source"),
    ],
)
def test_bad_files_are_skipped_one_by_one(data, reason):
    report = migrate.scan_source(data, "bad.py")
    assert report.skipped.startswith(reason) and not report.findings


def test_absurd_nesting_is_skipped(monkeypatch):
    def deep(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr(migrate.ast, "parse", deep)
    assert migrate.scan_source(b"x = 1\n", "deep.py").skipped == "not parseable (RecursionError)"


def test_a_coding_cookie_is_honoured():
    data = (
        "# -*- coding: latin-1 -*-\nfrom agents import function_tool\n"
        "@function_tool\ndef caf\xe9() -> int:\n    return 1\n"
    )
    report = migrate.scan_source(data.encode("latin-1"), "cafe.py")
    assert [f.name for f in report.findings] == ["caf\xe9"]


def test_crlf_sources_keep_their_line_ends():
    source = (
        "from langchain_core.tools import tool\r\n\r\n@tool\r\ndef f() -> int:\r\n    return 1\r\n"
    )
    tree = ast.parse(source)
    new = suggested(tree, source, scan_module(tree, source))
    assert "@registry.tool\r\ndef f()" in new
    assert split_lines("a\r\nb\rc\nd") == ["a\r\n", "b\r", "c\n", "d"]


def test_large_symlinked_and_hidden_files_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate, "MAX_FILE_BYTES", 10)
    (tmp_path / "big.py").write_text("x = 1  # a long line\n", encoding="utf-8")
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "lib.py").write_text("from agents import function_tool\n", encoding="utf-8")
    out = _cli(str(tmp_path))
    assert "big.py: skipped, larger than 10 bytes" in out and "lib.py" not in out
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "big.py" or real(self))
    assert migrate.scan_file(tmp_path / "big.py", "big.py").skipped == "a symlink, not followed"


def test_an_unreadable_file_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")

    def fail(self):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", fail)
    assert migrate.scan_file(path, "a.py").skipped == "unreadable (PermissionError)"


def test_one_file_path_and_a_missing_path(tmp_path):
    path = tmp_path / "app.py"
    shutil.copyfile(FIXTURES / "openai_agents_app.py.fixture", path)
    out = _cli(str(path), "--diff")
    assert "app.py:5: openai-agents" in out and "--- a/app.py\n+++ b/app.py\n" in out
    with pytest.raises(ConfigError, match="not a file or folder"):
        migrate.scan_path(tmp_path / "missing")


def test_unknown_framework_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as caught:
        _parser().parse_args(["migrate", ".", "--framework", "nope"])
    assert caught.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err
