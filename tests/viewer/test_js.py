"""The page's JavaScript: `node --test` suites, Mermaid parity with Python, the example run."""

from __future__ import annotations

import asyncio
import json
import runpy
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_adapter import Flow
from agent_adapter.flow.diagram import flow_mermaid
from agent_adapter.viewer import layout
from tests.viewer.helpers import refreshed
from tests.viewer.test_graph import parallel, research

HERE = Path(__file__).resolve().parent
JS = HERE / "js"
EXAMPLE = HERE.parents[1] / "examples" / "viewer_fan_out.py"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def node(*args: str) -> str:
    assert NODE is not None
    done = subprocess.run(
        [NODE, *args], capture_output=True, text=True, encoding="utf-8", timeout=120, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@needs_node
def test_every_js_suite_passes() -> None:
    suites = sorted(str(path) for path in JS.glob("*.test.mjs"))
    assert suites, "no JS test files"
    out = node("--test", *suites)
    assert "# fail 0" in out or "fail 0" in out


@needs_node
@pytest.mark.parametrize("make", [research, parallel], ids=["research", "parallel"])
def test_the_page_builds_the_same_mermaid_as_python(make: Any, tmp_path: Path) -> None:
    flow: Flow[Any] = make()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(layout(flow.graph())), encoding="utf-8")
    assert node(str(JS / "graph_cli.mjs"), "mermaid", str(path)) == flow_mermaid(flow.validate())


@needs_node
def test_the_graph_reads_the_fan_out_example_run(tmp_path: Path) -> None:
    module: dict[str, Any] = runpy.run_path(str(EXAMPLE))
    runs = asyncio.run(module["main"](tmp_path))
    folder = refreshed(runs)
    (summary,) = [s for s in folder.summaries() if s["flows"]]
    detail = folder.detail(summary["key"], paged=False)
    path = tmp_path / "detail.json"
    path.write_text(json.dumps(detail, default=str), encoding="utf-8")
    found = json.loads(node(str(JS / "graph_cli.mjs"), "snapshot", str(path), "help_search"))
    nodes = found["nodes"]
    assert {name: info["status"] for name, info in nodes.items()} == dict.fromkeys(
        ("plan", "search", "merge", "answer", "finish"), "ok"
    )
    assert [nodes[n]["visits"] for n in ("plan", "search", "merge", "answer", "finish")] == [
        1,
        2,
        2,
        2,
        1,
    ]
    assert nodes["search"]["items"] == {"done": 3, "total": 3}  # Map progress 3/3
    assert nodes["search"]["calls"] == 6 and nodes["answer"]["calls"] >= 2
    assert nodes["answer"]["cost"] and nodes["plan"]["changed"] == ["plan_steps"]
    arrow = "→"
    assert found["edges"] == {
        f"plan{arrow}search": 1,
        f"search{arrow}merge": 2,
        f"merge{arrow}answer": 2,
        f"answer{arrow}search": 1,  # the loop, taken once
        f"answer{arrow}finish": 1,
    }
    assert found["steps"] == 16
    assert found["mermaid"] == module["flow"].to_mermaid()


def test_no_static_file_names_an_external_url() -> None:
    static = HERE.parents[1] / "agent_adapter" / "viewer" / "static"
    for path in (p for p in static.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in text and "https://" not in text, path.name
