"""The viewer's side of sub-flows: nested layout and the page's Mermaid matching Python's."""

from __future__ import annotations

import json
from pathlib import Path

from agent_adapter.flow import Flow, SubFlow
from agent_adapter.flow.diagram import flow_mermaid
from agent_adapter.viewer import layout
from agent_adapter.viewer.layout import MAX_DEPTH
from tests.flow.subflow_helpers import Order, child_flow, parent_flow
from tests.viewer.test_js import JS, needs_node, node


def nested() -> Flow[Order]:
    inner = child_flow("inner", gate=True)
    middle = Flow("middle", state=Order).node("deep", SubFlow(inner)).end("deep")
    return parent_flow(middle)


def test_a_subflow_graph_is_laid_out_too() -> None:
    placed = layout(nested().graph())
    sub = next(node for node in placed["nodes"] if node["id"] == "sub")
    deep = sub["subflow"]["nodes"][0]
    assert {"layer", "order", "x"} <= set(deep)
    inner = deep["subflow"]
    assert [n["layer"] for n in inner["nodes"]] == [0, 1] and inner["width"] == 1.0


def test_sub_flows_nested_past_the_cap_are_dropped() -> None:
    graph: dict = {"flow": "leaf", "start": "a", "nodes": [{"id": "a"}], "edges": []}
    for depth in range(MAX_DEPTH + 2):
        graph = {
            "flow": f"f{depth}",
            "start": "a",
            "nodes": [{"id": "a", "subflow": graph}],
            "edges": [],
        }
    placed, levels = layout(graph), 0
    while "subflow" in placed["nodes"][0]:
        placed, levels = placed["nodes"][0]["subflow"], levels + 1
    assert levels == MAX_DEPTH


def test_a_malformed_subflow_entry_is_dropped() -> None:
    graph = {"flow": "f", "start": "a", "nodes": [{"id": "a", "subflow": "nope"}], "edges": []}
    assert "subflow" not in layout(graph)["nodes"][0]


@needs_node
def test_the_page_builds_the_same_subgraph_mermaid_as_python(tmp_path: Path) -> None:
    flow = nested()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(layout(flow.graph())), encoding="utf-8")
    assert node(str(JS / "graph_cli.mjs"), "mermaid", str(path)) == flow_mermaid(flow.validate())
