"""Flow.graph() and Flow.to_mermaid(): Map fan-out, Join fan-in, routes, loops, gates; layout."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from agent_adapter import Executor, Flow, FlowRunner, Registry
from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow import Call, Join, Map, Route
from agent_adapter.flow.diagram import back_edges
from agent_adapter.tracing import EventLog
from agent_adapter.viewer import layout


class State(BaseModel):
    items: list[str] = Field(default_factory=list)
    found: list[str] = Field(default_factory=list)
    left: str = ""
    right: str = ""
    done: bool = False


def noop(state: State) -> dict[str, Any]:
    return {}


def research() -> Flow[State]:
    return (
        Flow("research", state=State)
        .node("plan", noop)
        .node("search", Map("items", tool="lookup", into="found"), concurrency=4)
        .node("merge", Join(["search"]))
        .node("check", noop, pause=True)
        .node("finish", noop, pause=lambda s: {"n": len(s.found)})
        .edge("plan", "search")
        .edge("search", "merge")
        .edge("merge", "check")
        .route(
            "check",
            Route(lambda s: "done" if s.done else "again", {"again": "plan", "done": "finish"}),
            max_loops=2,
            on_exhausted="finish",
        )
        .end("finish")
    )


def parallel() -> Flow[State]:
    return (
        Flow("branches", state=State)
        .node("start", noop)
        .node("left", Call("lookup", into="left"))
        .node("right", Call("lookup", into="right"))
        .node("join", Join(["left", "right"]))
        .edge("start", ["left", "right"])
        .edge("left", "join")
        .edge("right", "join")
        .end("join")
    )


def by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in graph["nodes"]}


def test_graph_marks_fan_out_fan_in_and_pause_gates() -> None:
    graph = research().graph()
    nodes = by_id(graph)
    assert graph["flow"] == "research" and graph["start"] == "plan"
    assert nodes["search"]["fan_out"] and nodes["search"]["over"] == "items"
    assert nodes["search"]["tool"] == "lookup" and nodes["search"]["concurrency"] == 4
    assert nodes["merge"]["fan_in"] and nodes["merge"]["sources"] == ["search"]
    assert nodes["check"]["pause"] == "always"
    assert nodes["finish"]["pause"] == "rule" and nodes["finish"]["end"]
    assert nodes["plan"]["pause"] is None and nodes["plan"]["start"]


def test_graph_marks_route_branches_and_the_capped_loop() -> None:
    edges = {(e["source"], e["target"]): e for e in research().graph()["edges"]}
    back = edges[("check", "plan")]
    assert back["kind"] == "route" and back["label"] == "again"
    assert back["loop"] and back["max_loops"] == 2 and not back["on_exhausted"]
    done = edges[("check", "finish")]
    assert done["on_exhausted"] and not done["loop"]
    assert edges[("plan", "search")]["kind"] == "edge"
    assert not any(e["loop"] for key, e in edges.items() if key != ("check", "plan"))


def test_graph_marks_parallel_static_branches_and_call_tools() -> None:
    graph = parallel().graph()
    edges = {(e["source"], e["target"]): e for e in graph["edges"]}
    assert edges[("start", "left")]["parallel"] and edges[("start", "right")]["parallel"]
    assert not edges[("left", "join")]["parallel"]
    assert by_id(graph)["left"]["tool"] == "lookup"
    assert by_id(graph)["join"]["sources"] == ["left", "right"]


def test_route_labels_naming_the_same_target_share_one_edge() -> None:
    flow = (
        Flow("labels", state=State)
        .node("a", noop)
        .node("b", noop)
        .route("a", Route(lambda s: "x", {"x": "b", "y": "b"}))
        .end("b")
    )
    (edge,) = flow.graph()["edges"]
    assert edge["label"] == "x, y"


def test_map_with_tool_from_or_fn_names_its_source() -> None:
    def per_item(item: str, state: State, ctx: Any) -> str:
        return item

    flow = (
        Flow("maps", state=State)
        .node("by_item", Map("items", tool_from="tool", into="found"))
        .node("by_fn", Map("items", fn=per_item, into="found"))
        .edge("by_item", "by_fn")
        .end("by_fn")
    )
    nodes = by_id(flow.graph())
    assert nodes["by_item"]["tool"] == "item.tool"
    assert nodes["by_fn"]["tool"] == "per_item"


def test_mermaid_shows_shapes_loops_and_gates() -> None:
    text = research().to_mermaid()
    assert text.startswith("flowchart TD\n  aa_start((start)) --> n_plan\n")
    assert 'n_search[["search<br/>fan-out over items<br/>tool: lookup"]]' in text
    assert 'n_merge{{"merge<br/>fan-in: search"}}' in text
    assert 'n_check["check<br/>pause gate"]:::pause' in text
    assert 'n_check -.->|"again (max 2)"| n_plan' in text
    assert 'n_check -->|"done"| n_finish' in text
    assert "n_finish --> aa_end((end))" in text
    assert "classDef pause" in text


def test_mermaid_draws_parallel_branches_thick_and_escapes_labels() -> None:
    text = parallel().to_mermaid()
    assert "n_start ==> n_left" in text
    assert 'n_left(["left<br/>tool: lookup"])' in text
    flow = Flow("quotes", state=State).node("a", Call('say "<hi>" & bye')).end("a")
    assert 'tool: say #quot;#lt;hi#gt;#quot; #amp; bye"' in flow.to_mermaid()


def test_graph_and_mermaid_validate_first() -> None:
    broken = Flow("broken", state=State).node("a", noop)
    with pytest.raises(FlowDefinitionError):
        broken.graph()
    with pytest.raises(FlowDefinitionError):
        broken.to_mermaid()


def test_back_edges_follow_the_depth_first_path() -> None:
    assert back_edges(research().validate()) == {("check", "plan")}
    assert back_edges(parallel().validate()) == set()


def test_layout_layers_fan_out_and_fan_in() -> None:
    placed = layout(parallel().graph())
    nodes = by_id(placed)
    assert [nodes[n]["layer"] for n in ("start", "left", "right", "join")] == [0, 1, 1, 2]
    assert {nodes["left"]["order"], nodes["right"]["order"]} == {0, 1}
    assert placed["layers"] == 3 and placed["width"] == 2


def test_layout_ignores_loop_edges_for_depth() -> None:
    nodes = by_id(layout(research().graph()))
    assert [nodes[n]["layer"] for n in ("plan", "search", "merge", "check", "finish")] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_layout_of_an_empty_graph() -> None:
    assert layout({"nodes": [], "edges": []})["layers"] == 0


async def test_only_graph_takers_get_the_graph_and_it_is_built_once() -> None:
    registry = Registry()

    @registry.tool
    def lookup(item: str = "x") -> str:
        """Look an item up."""
        return item

    class Taker:
        def __init__(self) -> None:
            self.graphs: list[tuple[str, str, dict[str, Any]]] = []

        def emit(self, event: Any) -> None:
            return None

        def take_flow_graph(self, run_id: str, flow: str, graph: dict[str, Any]) -> None:
            self.graphs.append((run_id, flow, graph))

    log, taker = EventLog(), Taker()
    executor = Executor(registry, sink=log)
    executor.add_sink(taker)
    runner = FlowRunner(executor)
    for _ in range(2):
        assert (await runner.run(parallel(), {"items": []})).status == "done"
    (started, _) = log.of_type("flow.started")
    assert started.data == {"flow": "branches"}  # generic sinks never see the graph
    assert [flow for _, flow, _ in taker.graphs] == ["branches", "branches"]
    assert taker.graphs[0][2] == parallel().graph()
    assert taker.graphs[0][2] is taker.graphs[1][2]  # cached per graph digest
