import pytest
from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow import Call, Flow, Join, Map, node_model, node_prompt


class S(BaseModel):
    items: list[dict] = []
    text: str = ""


def step(state):
    return None


def test_builder_methods_chain_and_default_start_is_the_first_node():
    flow = Flow("chain", state=S, description="demo")
    assert flow.node("a", step) is flow
    assert flow.node("b", step).edge("a", "b").end("b", "b") is flow
    graph = flow.validate()
    assert graph.start == "a" and graph.ends == frozenset({"b"})
    assert flow.description == "demo" and flow.state_model is S
    assert repr(flow) == "Flow('chain', nodes=['a', 'b'])"


def test_explicit_start_and_parallel_edges():
    graph = (
        Flow("par", state=S)
        .node("a", step)
        .node("b", step)
        .node("c", step)
        .node("d", Join(["b", "c"]))
        .start("a")
        .edge("a", ["b", "c"])
        .edge("a", "b")  # repeated edges are kept once
        .edge("b", "d")
        .edge("c", "d")
        .end("d")
        .validate()
    )
    assert graph.edges["a"] == ("b", "c")


def test_state_must_be_a_pydantic_model():
    with pytest.raises(FlowDefinitionError, match="state must be a pydantic BaseModel subclass"):
        Flow("bad", state=dict)  # type: ignore[type-var]


def test_node_kinds_follow_the_action():
    graph = (
        Flow("kinds", state=S)
        .node("f", step)
        .node("c", Call("t"))
        .node("m", Map("items", tool="t", into="items"))
        .node("j", Join(["m"]))
        .edge("f", "c")
        .edge("c", "m")
        .edge("m", "j")
        .end("j")
        .validate()
    )
    assert [node.kind for node in graph.nodes.values()] == ["fn", "call", "map", "join"]


def test_prompts_and_with_prompts_return_a_copy():
    flow = (
        Flow("prompts", state=S)
        .node("a", step, prompt="Plan the research.")
        .node("b", step)
        .edge("a", "b")
        .end("b")
    )
    assert flow.prompts() == {"a": "Plan the research."}
    copy = flow.with_prompts({"a": "Plan briefly.", "b": "Answer."})
    assert copy.prompts() == {"a": "Plan briefly.", "b": "Answer."}
    assert flow.prompts() == {"a": "Plan the research."}
    assert copy.validate().nodes["a"].prompt == "Plan briefly."
    copy.node("c", step)  # the copy's tables are its own
    assert "c" not in flow.validate().nodes
    assert repr(copy) == "Flow('prompts', nodes=['a', 'b', 'c'])"


def test_with_prompts_rejects_unknown_nodes():
    flow = Flow("prompts", state=S).node("a", step).end("a")
    with pytest.raises(FlowDefinitionError, match="unknown node 'zz'"):
        flow.with_prompts({"zz": "x"})


def test_map_needs_exactly_one_source_of_work():
    with pytest.raises(FlowDefinitionError, match="exactly one of tool=, tool_from= or fn="):
        Map("items")
    with pytest.raises(FlowDefinitionError):
        Map("items", tool="t", fn=lambda item, state, ctx: item)


def test_node_prompt_and_model_read_the_context():
    ctx = ToolContext()
    assert node_prompt(ctx) is None and node_model(ctx) is None
    model = object()
    ctx = ToolContext(extras={"aa.node_prompt": "Be brief.", "aa.model": model})
    assert node_prompt(ctx) == "Be brief."
    assert node_model(ctx) is model
    assert node_prompt(ToolContext(extras={"aa.node_prompt": 3})) is None
