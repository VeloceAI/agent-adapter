"""SubFlow nodes: state in and out, merge rules, nested events, failures and hooks."""

import pytest
from pydantic import BaseModel

from agent_adapter.core.errors import FlowError
from agent_adapter.exec import Executor
from agent_adapter.flow import Flow, FlowRunner, SubFlow
from agent_adapter.flow.visit import NodeEnv, run_node
from agent_adapter.hooks.base import Hook
from tests.fakes.sinks import RecordingSink
from tests.flow.helpers import registry, runner
from tests.flow.subflow_helpers import Order, Pack, Shop, child_flow, parent_flow


async def test_same_named_fields_go_in_and_changed_ones_come_back():
    run, _, _ = runner()
    done = await run.run(parent_flow(child_flow()), {"item": "mug"})
    assert done.status == "done"
    assert done.state.qty == 3 and done.state.note == "checked 3"
    # additive fields start empty in the child, so its additions land once
    assert done.state.log == ["start", "count mug", "check", "after checked 3"]
    assert done.state.total == 1


async def test_explicit_field_maps_choose_what_goes_in_and_out():
    child = (
        Flow("labeler", state=Pack)
        .node("label", lambda s: {"label": f"{s.boxes} x {s.item}"})
        .end("label")
    )
    flow = (
        Flow("shop", state=Shop)
        .node(
            "sub", SubFlow(child, input={"item": "item", "boxes": "qty"}, into={"label": "label"})
        )
        .end("sub")
    )
    run, _, _ = runner()
    done = await run.run(flow, {"item": "cup", "qty": 4})
    assert done.state.label == "4 x cup" and done.state.packed is None


async def test_functions_map_in_and_out_and_a_field_takes_the_whole_child_state():
    child = Flow("labeler", state=Pack).node("label", lambda s: {"label": s.item * 2}).end("label")
    by_fn = SubFlow(
        child, input=lambda s: {"item": s.item.upper()}, into=lambda c: {"label": c.label}
    )
    whole = SubFlow(child, input=lambda s: Pack(item=s.item), into="packed")
    flow = Flow("shop", state=Shop).node("a", by_fn).node("b", whole).edge("a", "b").end("b")
    run, _, _ = runner()
    done = await run.run(flow, {"item": "ab"})
    assert done.state.label == "ABAB"
    assert done.state.packed == Pack(item="ab", label="abab")


async def test_unchanged_fields_are_not_written_back():
    child = Flow("noop", state=Order).node("idle", lambda s: None).end("idle")
    flow = (
        Flow("both", state=Order)
        .node("start", lambda s: None)
        .node("sub", SubFlow(child))
        .node("other", lambda s: {"note": "from other"})  # a sibling writing the same field
        .edge("start", ["sub", "other"])
        .end("sub", "other")
    )
    run, _, _ = runner()
    done = await run.run(flow, {"note": "x"})
    assert done.state.note == "from other"  # no conflicting write from the idle child


async def test_child_events_nest_under_the_subflow_node_visit():
    sink = RecordingSink()
    run, _, _ = runner(sink=sink)
    await run.run(parent_flow(child_flow()), {})
    child_events = [e for e in sink.events if e.data.get("flow") == "child"]
    started = child_events[0]
    assert started.type == "flow.started" and started.call_id == "parent:sub:1/child"
    assert started.parent_id == "parent:sub:1"
    assert started.data["parent_flow"] == "parent" and started.data["parent_node"] == "sub"
    nodes = [e for e in child_events if e.type == "node.started"]
    assert [e.call_id for e in nodes] == [
        "parent:sub:1/child:count:1",
        "parent:sub:1/child:check:1",
    ]
    assert {e.parent_id for e in nodes} == {"parent:sub:1"}
    assert nodes[0].data["scope"] == "parent:sub:1/"
    finished = child_events[-1]
    assert finished.type == "flow.finished" and finished.data["status"] == "done"
    assert finished.data["steps"] == 2


async def test_two_subflow_nodes_reuse_one_child_flow_with_their_own_ids():
    child = child_flow()
    flow = (
        Flow("twice", state=Order)
        .node("first", SubFlow(child))
        .node("second", SubFlow(child))
        .edge("first", "second")
        .end("second")
    )
    sink = RecordingSink()
    run, _, _ = runner(sink=sink)
    done = await run.run(flow, {})
    assert done.state.qty == 2 and done.state.total == 2
    ids = [e.call_id for e in sink.events if e.type == "node.started" and e.data["flow"] == "child"]
    assert ids[0].startswith("twice:first:1/") and ids[2].startswith("twice:second:1/")


async def test_a_failing_child_node_fails_the_subflow_node():
    child = Flow("bad", state=Order).node("boom", lambda s: 1 / 0).end("boom")
    sink = RecordingSink()
    run, _, _ = runner(sink=sink)
    with pytest.raises(FlowError) as caught:
        await run.run(parent_flow(child), {})
    assert caught.value.node == "sub" and caught.value.flow == "parent"
    finished = [e for e in sink.events if e.type == "flow.finished" and e.data["flow"] == "bad"]
    assert finished[0].data["status"] == "failed"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"input": lambda s: 1 / 0}, "SubFlow input failed: ZeroDivisionError"),
        ({"input": lambda s: 42}, "SubFlow input returned int, not a mapping"),
        ({"input": lambda s: {"qty": "lots"}}, "sub-flow input is invalid: qty"),
        ({"into": lambda c: 1 / 0}, "SubFlow into failed: ZeroDivisionError"),
        ({"into": lambda c: ["no"]}, "SubFlow into returned list, not a mapping"),
    ],
)
async def test_mapping_failures_name_the_subflow_node(options, message):
    run, _, _ = runner()
    with pytest.raises(FlowError, match=message) as caught:
        await run.run(parent_flow(child_flow(), **options), {})
    assert caught.value.node == "sub"


async def test_the_child_respects_the_runners_max_steps():
    class Loop(BaseModel):
        n: int = 0

    from agent_adapter.core.errors import FlowLoopExceeded
    from agent_adapter.flow import Route

    child = (
        Flow("spin", state=Loop)
        .node("tick", lambda s: {"n": s.n + 1})
        .route("tick", Route(lambda s: "tick", ["tick"]), max_loops=50)
    )
    flow = Flow("outer", state=Loop).node("sub", SubFlow(child)).end("sub")
    run, _, _ = runner(max_steps=3)
    with pytest.raises(FlowError) as caught:
        await run.run(flow, {})
    assert isinstance(caught.value.__cause__, FlowLoopExceeded)


async def test_node_hooks_see_the_child_nodes_too():
    seen = []

    class Watch(Hook):
        async def before_node(self, inv):
            seen.append(f"{inv.flow}.{inv.node}")

    run, _, _ = runner(hooks=[Watch()])
    await run.run(parent_flow(child_flow()), {})
    assert seen == ["parent.start", "parent.sub", "child.count", "child.check", "parent.after"]


async def test_a_subflow_node_outside_the_local_runner_fails_clearly():
    graph = parent_flow(child_flow()).validate()
    env = NodeEnv("parent", Executor(registry), ())
    ctx = Executor(registry).context()
    with pytest.raises(FlowError, match="SubFlow nodes run on FlowRunner only"):
        await run_node(env, graph.nodes["sub"], Order(), ctx, visit=1)


async def test_a_runner_built_once_runs_subflows_in_many_runs():
    run = FlowRunner(Executor(registry))
    flow = parent_flow(child_flow())
    first, second = await run.run(flow, {"item": "a"}), await run.run(flow, {"item": "b"})
    assert first.state.log[1] == "count a" and second.state.log[1] == "count b"
