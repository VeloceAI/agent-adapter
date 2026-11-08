"""SubFlow validation (child, cycles, field maps, types), the digest, the graph and mermaid."""

from typing import Any

import pytest
from pydantic import BaseModel

from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow import Flow, SubFlow
from agent_adapter.flow._digest import graph_digest
from tests.flow.subflow_helpers import Order, Pack, Shop, child_flow, parent_flow


def problems(flow: Flow[Any]) -> list[str]:
    with pytest.raises(FlowDefinitionError) as caught:
        flow.validate()
    return caught.value.problems


def one(action: Any, state: type[BaseModel] = Shop) -> Flow[Any]:
    return Flow("host", state=state).node("sub", action).end("sub")


def test_a_valid_subflow_node_holds_its_child_graph():
    graph = parent_flow(child_flow()).validate()
    node = graph.nodes["sub"]
    assert node.kind == "subflow" and node.child is not None and node.child.name == "child"


def test_a_flow_that_contains_itself_is_a_cycle():
    loop = Flow("loop", state=Order)
    loop.node("again", SubFlow(loop)).end("again")
    assert problems(loop) == ["node 'again': sub-flow 'loop': sub-flow cycle loop -> loop"]


def test_an_indirect_subflow_cycle_is_reported_once_per_path():
    a, b = Flow("a", state=Order), Flow("b", state=Order)
    a.node("x", SubFlow(b)).end("x")
    b.node("y", SubFlow(a)).end("y")
    assert problems(a) == [
        "node 'x': sub-flow 'b': node 'y': sub-flow 'a': sub-flow cycle a -> b -> a"
    ]


def test_the_same_child_twice_is_not_a_cycle():
    child = child_flow()
    flow = (
        Flow("twice", state=Order)
        .node("one", SubFlow(child))
        .node("two", SubFlow(child))
        .edge("one", "two")
        .end("two")
    )
    assert set(flow.validate().nodes) == {"one", "two"}


def test_a_subflow_needs_a_flow_and_a_valid_child():
    assert problems(one(SubFlow("nope"))) == ["node 'sub': SubFlow needs a Flow, got str"]
    broken = Flow("broken", state=Order)
    assert problems(one(SubFlow(broken), Order)) == [
        "node 'sub': sub-flow 'broken': flow has no nodes"
    ]


def labeler() -> Flow[Pack]:
    return Flow("labeler", state=Pack).node("label", lambda s: None).end("label")


@pytest.mark.parametrize(
    ("options", "found"),
    [
        ({"input": {"nope": "item"}}, "SubFlow input: 'nope' is not a field of Pack"),
        ({"input": {"item": "nope"}}, "SubFlow input: 'nope' is not a field of Shop"),
        ({"into": {"nope": "label"}}, "SubFlow into: 'nope' is not a field of Shop"),
        ({"into": {"label": "nope"}}, "SubFlow into: 'nope' is not a field of Pack"),
        ({"input": ["item"]}, "SubFlow input must be a field mapping, a function or None"),
        ({"into": "nope"}, "SubFlow into 'nope' is not a field of Shop"),
        ({"into": "label"}, "SubFlow into 'label' cannot hold a Pack"),
        ({"input": {"label": "qty"}}, "SubFlow field 'qty' (int) and child field 'label' (str)"),
    ],
)
def test_field_maps_and_types_are_checked(options, found):
    [problem] = problems(one(SubFlow(labeler(), **options)))
    assert problem.startswith(f"node 'sub': {found}")


def test_a_whole_state_field_may_be_optional_and_functions_skip_field_checks():
    flow = one(SubFlow(labeler(), input=lambda s: {}, into="packed"))
    assert flow.validate().nodes["sub"].child is not None


def test_an_additive_field_mapped_in_and_back_would_count_twice():
    [problem] = problems(one(SubFlow(child_flow(), input={"log": "log"}), Order))
    assert "field 'log' goes in and comes back under merge rule 'append'" in problem


def test_same_named_fields_of_different_types_need_a_map():
    class Other(BaseModel):
        qty: str = ""

    child = Flow("other", state=Other).node("x", lambda s: None).end("x")
    [problem] = problems(one(SubFlow(child), Order))
    assert "field 'qty' (int) and child field 'qty' (str)" in problem


def test_the_digest_covers_the_child_graph_and_the_field_maps():
    base = graph_digest(parent_flow(child_flow()).validate())
    assert base == graph_digest(parent_flow(child_flow()).validate())
    gated = graph_digest(parent_flow(child_flow(gate=True)).validate())
    mapped = graph_digest(parent_flow(child_flow(), into={"note": "note"}).validate())
    assert len({base, gated, mapped}) == 3


def test_the_graph_json_nests_the_child_graph():
    data = parent_flow(child_flow()).graph()
    sub = next(node for node in data["nodes"] if node["id"] == "sub")
    assert sub["kind"] == "subflow" and sub["subflow"]["flow"] == "child"
    assert [n["id"] for n in sub["subflow"]["nodes"]] == ["count", "check"]
    start = next(node for node in data["nodes"] if node["id"] == "start")
    assert "subflow" not in start and "cache" not in start  # other nodes keep their shape


def test_mermaid_draws_a_subflow_as_a_subgraph():
    inner = child_flow("inner", gate=True)
    middle = Flow("middle", state=Order).node("deep", SubFlow(inner)).end("deep")
    text = parent_flow(middle).to_mermaid()
    assert text.splitlines()[2:12] == [
        '  n_start["start"]',
        '  subgraph n_sub["sub<br/>sub-flow: middle"]',
        '    subgraph s1__deep["deep<br/>sub-flow: inner"]',
        '      s1_0__count["count"]',
        '      s1_0__check["check<br/>pause gate"]:::pause',
        "      s1_0__count --> s1_0__check",
        "    end",
        "  end",
        '  n_after["after"]',
        "  n_start --> n_sub",
    ]
