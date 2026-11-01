from typing import Annotated

import pytest
from pydantic import Field

from agent_adapter.core.errors import FlowError
from agent_adapter.flow import Call, Flow, FlowRunner, Map, Merge
from tests.flow.helpers import Hit, State, registry, runner


async def test_call_node_runs_a_tool_by_name_or_spec():
    flow = (
        Flow("calls", state=State)
        .node("by_name", Call("bump", args=lambda s: {"n": s.n}, into="n"))
        .node("by_spec", Call(registry.get("bump"), args=lambda s: {"n": s.n}, into="n"))
        .node("no_write", Call("bump", args={"n": 1}))
        .edge("by_name", "by_spec")
        .edge("by_spec", "no_write")
        .end("no_write")
    )
    run, _, _ = runner()
    result = await run.run(flow, {"n": 5})
    assert result.state.n == 7


async def test_call_node_failures_are_flow_errors():
    flow = Flow("call_fail", state=State).node("a", Call("boom", into="note")).end("a")
    run, _, _ = runner()
    with pytest.raises(FlowError, match="node 'a' failed: tool-failed"):
        await run.run(flow, {})
    unknown = Flow("call_unknown", state=State).node("a", Call("nope")).end("a")
    with pytest.raises(FlowError, match="tool-not-found"):
        await run.run(unknown, {})


async def test_call_without_a_registry_looks_in_the_default_registry():
    from agent_adapter.exec import Executor

    flow = Flow("noreg", state=State).node("a", Call("bump_not_in_default")).end("a")
    with pytest.raises(FlowError, match="tool-not-found"):
        await FlowRunner(Executor()).run(flow, {})


async def test_map_with_tool_from_flattens_list_results_into_a_keyed_field():
    items = [
        {"tool": "search", "args": {"query": "a"}},
        {"tool": "search", "query": "b"},
        {"tool": "search", "args": {"query": "a"}},
    ]
    flow = (
        Flow("fanout", state=State)
        .node("retrieve", Map("items", tool_from="tool", into="hits"), concurrency=2)
        .end("retrieve")
    )
    run, _, _ = runner()
    result = await run.run(flow, {"items": items})
    assert [h.source_id for h in result.state.hits] == ["src-a", "shared", "src-b"]


async def test_map_with_a_tool_and_args_fn_and_default_into():
    class Bumped(State):
        bumped: Annotated[list[int], Merge.append] = Field(default_factory=list)

    for tool in ("bump", registry.get("bump")):
        flow = (
            Flow("map_tool", state=Bumped)
            .node("bumped", Map("items", tool=tool, args_fn=lambda item: {"n": item}))
            .end("bumped")
        )
        run, _, _ = runner()
        result = await run.run(flow, {"items": [1, 2, 3]})
        assert result.state.bumped == [2, 3, 4]


async def test_map_items_may_be_models():
    flow = (
        Flow("models", state=State)
        .node(
            "retrieve",
            Map("items", tool="search", args_fn=lambda h: {"query": h.source_id}, into="hits"),
        )
        .end("retrieve")
    )

    run, _, _ = runner()
    result = await run.run(flow, {"items": [Hit(source_id="m")]})
    assert [h.source_id for h in result.state.hits] == ["src-m", "shared"]


async def test_map_with_fn_gets_item_state_and_context():
    seen = []

    def per_item(item, state, ctx):
        seen.append(ctx.call_id)
        return f"{state.query}:{item}"

    flow = Flow("map_fn", state=State).node("log", Map("items", fn=per_item)).end("log")
    run, _, _ = runner()
    result = await run.run(flow, {"query": "q", "items": ["x", "y"]})
    assert result.state.log == ["q:x", "q:y"]
    assert sorted(seen) == ["map_fn:log:1:0", "map_fn:log:1:1"]


async def test_map_over_an_empty_list_writes_nothing():
    flow = Flow("empty", state=State).node("log", Map("items", fn=lambda i, s, c: i)).end("log")
    run, _, _ = runner()
    assert (await run.run(flow, {})).state.log == []


async def test_map_item_failure_fails_or_is_skipped():
    def maybe(item, state, ctx):
        if item == "bad":
            raise ValueError("bad item")
        return item

    fail = Flow("map_fail", state=State).node("log", Map("items", fn=maybe)).end("log")
    skip = (
        Flow("map_skip", state=State)
        .node("log", Map("items", fn=maybe, on_error="skip"))
        .end("log")
    )
    run, _, _ = runner()
    with pytest.raises(FlowError, match="item 1 failed: ValueError"):
        await run.run(fail, {"items": ["ok", "bad"]})
    result = await run.run(skip, {"items": ["ok", "bad", "fine"]})
    assert result.state.log == ["ok", "fine"]


async def test_map_tool_failures_fail_or_are_skipped():
    items = [{"tool": "search", "query": "a"}, {"tool": "boom"}]
    fail = Flow("tools_fail", state=State).node("hits", Map("items", tool_from="tool")).end("hits")
    skip = (
        Flow("tools_skip", state=State)
        .node("hits", Map("items", tool_from="tool", on_error="skip"))
        .end("hits")
    )
    run, _, _ = runner()
    with pytest.raises(FlowError, match="item 1 failed: tool-failed"):
        await run.run(fail, {"items": items})
    assert len((await run.run(skip, {"items": items})).state.hits) == 2


async def test_map_rejects_bad_items():
    run, _, _ = runner()
    no_tool = Flow("no_tool", state=State).node("hits", Map("items", tool_from="tool")).end("hits")
    with pytest.raises(FlowError, match="item 0 has no tool name under 'tool'"):
        await run.run(no_tool, {"items": [{"query": "a"}]})
    bad_args = Flow("bad_args", state=State).node("hits", Map("items", tool="search")).end("hits")
    with pytest.raises(FlowError, match="item 0 arguments are str, not a mapping"):
        await run.run(bad_args, {"items": ["a"]})
    not_list = (
        Flow("not_list", state=State).node("log", Map("query", fn=lambda i, s, c: i)).end("log")
    )
    with pytest.raises(FlowError, match=r"Map\.over 'query' is str, not a list"):
        await run.run(not_list, {"query": "q"})
