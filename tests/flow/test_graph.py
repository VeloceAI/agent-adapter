from typing import Annotated

import pytest
from pydantic import BaseModel

from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow import Call, Flow, FlowGraph, Join, Map, Merge, Route


class S(BaseModel):
    items: list[dict] = []
    out: Annotated[list[dict], Merge.append] = []
    n: int = 0
    done: bool = False


def step(state):
    return None


def problems(flow: Flow) -> list[str]:
    with pytest.raises(FlowDefinitionError) as info:
        flow.validate()
    assert info.value.flow == flow.name
    return info.value.problems


def test_valid_flow_builds_a_graph():
    graph = (
        Flow("ok", state=S)
        .node("a", step)
        .node("fan", Map("items", tool="t", into="out"), concurrency=2)
        .node("j", Join(["fan"]))
        .node("b", step)
        .node("end_", step)
        .edge("a", "fan")
        .edge("fan", "j")
        .edge("j", "b")
        .route(
            "b",
            Route(lambda s: "loop", targets={"loop": "a", "stop": "end_"}),
            max_loops=2,
            on_exhausted="end_",
        )
        .end("end_")
        .validate()
    )
    assert isinstance(graph, FlowGraph)
    assert graph.start == "a" and graph.ends == frozenset({"end_"})
    assert list(graph.nodes) == ["a", "fan", "j", "b", "end_"]
    assert graph.nodes["fan"].kind == "map" and graph.nodes["fan"].concurrency == 2
    assert graph.nodes["j"].kind == "join" and graph.nodes["a"].kind == "fn"
    assert graph.edges["a"] == ("fan",) and graph.edges["b"] == ()
    assert graph.routes["b"].targets == {"loop": "a", "stop": "end_"}
    assert graph.successors("b") == ("a", "end_")
    assert graph.predecessors("a") == ("b",)
    assert graph.predecessors("j") == ("fan",)
    assert graph.rules["out"].name == "append" and graph.rules["n"].name == "replace"
    assert graph.state_model is S and graph.name == "ok"


def test_sequence_targets_map_each_name_to_itself():
    graph = (
        Flow("seq", state=S)
        .node("a", step)
        .node("b", step)
        .node("c", step)
        .route("a", Route(lambda s: "b", targets=["b", "c"]))
        .end("b", "c")
        .validate()
    )
    assert graph.routes["a"].targets == {"b": "b", "c": "c"}
    assert graph.routes["a"].max_loops is None


def test_names_must_be_identifiers():
    flow = Flow("bad-name", state=S).node("bad node", step).end("bad node")
    assert problems(flow) == [
        "flow name 'bad-name' is not an identifier (^[A-Za-z_][A-Za-z0-9_]{0,63}$)",
        "node name 'bad node' is not an identifier (^[A-Za-z_][A-Za-z0-9_]{0,63}$)",
    ]


def test_every_problem_is_listed_together():
    flow = (
        Flow("many", state=S)
        .node("a", step)
        .node("a", step)
        .node("b", step)
        .node("orphan", step)
        .node("j", Join(["orphan"]))
        .edge("a", "ghost")
        .edge("b", "j")
        .route("b", Route(lambda s: "a", targets=["a", "nowhere"]), on_exhausted="a")
        .end("j", "missing")
    )
    found = problems(flow)
    assert "duplicate node 'a'" in found
    assert "edge a -> ghost: unknown node 'ghost'" in found
    assert "route from b: unknown target 'nowhere'" in found
    assert "route from b: on_exhausted needs max_loops" in found
    assert "end: unknown node 'missing'" in found
    assert "node 'b' has static edges and a route; use one" in found
    assert "node 'orphan' has no exit and is not an end node" in found
    assert "node 'orphan' is not reachable from 'a'" in found
    assert "join 'j': source 'orphan' is not a predecessor" in found


def test_join_sources_must_be_predecessors_and_known():
    flow = (
        Flow("joins", state=S)
        .node("a", step)
        .node("b", step)
        .node("j", Join(["b", "zzz"]))
        .node("k", Join([]))
        .edge("a", ["b", "j"])
        .edge("b", "k")
        .edge("j", "k")
        .end("k")
    )
    assert problems(flow) == [
        "join 'j': source 'b' is not a predecessor",
        "join 'j': unknown source 'zzz'",
        "join 'k' has no sources",
    ]


def test_cycles_need_a_route_with_max_loops():
    flow = (
        Flow("loops", state=S)
        .node("a", step)
        .node("b", step)
        .node("c", step)
        .edge("a", "b")
        .route("b", Route(lambda s: "a", targets=["a", "c"]))
        .end("c")
    )
    assert problems(flow) == ["cycle a -> b -> a does not pass through a route with max_loops"]


def test_static_self_loop_is_a_cycle():
    flow = Flow("self", state=S).node("a", step).edge("a", "a")
    assert "cycle a -> a does not pass through a route with max_loops" in problems(flow)


def test_route_options_are_checked():
    flow = (
        Flow("routes", state=S)
        .node("a", step)
        .node("b", step)
        .node("c", step)
        .route("a", Route(lambda s: "b", targets=["b"]), max_loops=-1, on_exhausted="c")
        .route("a", Route(lambda s: "b", targets=["b"]))
        .route("b", Route(lambda s: "c", targets="c"))  # a bare string is not a target list
        .route("zz", Route(lambda s: "c", targets=["c"]))
        .end("c")
    )
    found = problems(flow)
    assert "route from a: max_loops must be an int >= 0, got -1" in found
    assert "route from a: on_exhausted 'c' is not one of its targets" in found
    assert "node 'a' has more than one route" in found
    assert "route from b: targets must be node names or a label -> node mapping" in found
    assert "route from zz: unknown node 'zz'" in found


def test_end_nodes_have_no_exit():
    flow = Flow("ends", state=S).node("a", step).node("b", step).edge("a", "b").end("a", "b")
    assert problems(flow) == ["end node 'a' has an exit"]


def test_state_fields_are_checked_for_map_and_call():
    flow = (
        Flow("fields", state=S)
        .node("a", Map("nope", tool="t"))
        .node("b", Map("items", fn=lambda item, state, ctx: item, into="zzz", on_error="explode"))
        .node("c", Call("t", into="missing"))
        .node("out", Map("items", tool="t"))
        .edge("a", "b")
        .edge("b", "c")
        .edge("c", "out")
        .end("out")
    )
    assert problems(flow) == [
        "node 'a': Map.over 'nope' is not a state field",
        "node 'a': Map writes 'a', which is not a state field",
        "node 'b': Map writes 'zzz', which is not a state field",
        "node 'b': Map.on_error must be 'fail' or 'skip', got 'explode'",
        "node 'c': Call.into 'missing' is not a state field",
    ]


def test_reserved_compiler_fields_are_rejected():
    class Reserved(BaseModel):
        aa_loop_counts: dict = {}
        aa_visits: dict = {}

    flow = Flow("reserved", state=Reserved).node("a", step).end("a")
    assert problems(flow) == [
        "state field 'aa_loop_counts' is reserved",
        "state field 'aa_visits' is reserved",
    ]


def test_merge_rule_problems_are_listed():
    class TwoRules(BaseModel):
        log: Annotated[list[str], Merge.append, Merge.sum] = []

    flow = Flow("rules", state=TwoRules).node("a", step).end("a")
    assert problems(flow) == ["field 'log' has more than one merge rule"]


def test_node_options_are_checked():
    flow = (
        Flow("opts", state=S)
        .node("a", step, concurrency=0, timeout=-1, retries=-2, pause="yes", prompt=3)
        .node("b", 42)
        .edge("a", "b")
        .end("b")
    )
    assert problems(flow) == [
        "node 'a': concurrency must be an int >= 1, got 0",
        "node 'a': timeout must be a positive number, got -1",
        "node 'a': retries must be an int >= 0, got -2",
        "node 'a': pause must be a bool or a callable, got 'yes'",
        "node 'a': prompt must be a string, got 3",
        "node 'b': action must be a function, Call, Map, Join or SubFlow, got int",
    ]


def test_empty_flow_and_unknown_start():
    assert problems(Flow("empty", state=S)) == ["flow has no nodes"]
    assert "start: unknown node 'zz'" in problems(Flow("st", state=S).node("a", step).start("zz"))
