import asyncio

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    FlowDefinitionError,
    FlowError,
    FlowLoopExceeded,
    ToolUsageError,
)
from agent_adapter.flow import Flow, FlowRunner, Join, Route
from agent_adapter.runtime.current import current_context
from tests.flow.helpers import State, runner


def write(**update):
    return lambda state: dict(update)


async def test_linear_flow_runs_to_done_and_counts_steps_and_visits():
    flow = (
        Flow("linear", state=State)
        .node("a", write(log=["a"]))
        .node("b", lambda state: {"log": ["b"], "n": len(state.log)})
        .edge("a", "b")
        .end("b")
    )
    run, _, _ = runner()
    result = await run.run(flow, {"query": "q"})
    assert result.status == "done"
    assert result.state.log == ["a", "b"] and result.state.n == 1
    assert result.steps == 2 and result.visits == {"a": 1, "b": 1}
    assert result.checkpoint is None and result.pending is None and result.denied_at is None


async def test_state_instance_is_accepted_and_bad_state_is_rejected():
    flow = Flow("one", state=State).node("a", write(n=1)).end("a")
    run, _, _ = runner()
    assert (await run.run(flow, State(query="x"))).state.query == "x"
    with pytest.raises(FlowError, match="initial state of flow 'one' is invalid"):
        await run.run(flow, {"n": "not a number"})
    with pytest.raises(ToolUsageError, match="state must be a State or a mapping"):
        await run.run(flow, 42)


async def test_invalid_flow_raises_before_running():
    flow = Flow("bad", state=State).node("a", write(n=1))
    run, _, _ = runner()
    with pytest.raises(FlowDefinitionError, match="no exit"):
        await run.run(flow, {})


async def test_static_branches_run_in_one_superstep_and_merge_in_node_name_order():
    order = []

    async def slow_branch(state):
        await asyncio.sleep(0.01)
        order.append("b")
        return {"log": ["b"], "total": 2}

    async def fast_branch(state):
        order.append("c")
        return {"log": ["c"], "total": 3}

    flow = (
        Flow("branches", state=State)
        .node("a", write(log=["a"]))
        .node("b", slow_branch)
        .node("c", fast_branch)
        .node("d", Join(["b", "c"], fn=lambda s: {"note": ",".join(s.log)}))
        .edge("a", ["b", "c"])
        .edge("b", "d")
        .edge("c", "d")
        .end("d")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert order == ["c", "b"]  # concurrent: the fast branch finished first
    assert result.state.log == ["a", "b", "c"]  # merged in node-name order
    assert result.state.total == 5 and result.state.note == "a,b,c"
    assert result.steps == 3


async def test_join_waits_until_nothing_else_is_pending():
    flow = (
        Flow("defer", state=State)
        .node("start", write(log=["start"]))
        .node("short", write(log=["short"]))
        .node("long1", write(log=["long1"]))
        .node("long2", write(log=["long2"]))
        .node("join", Join(["short", "long2"], fn=lambda s: {"note": "|".join(s.log)}))
        .edge("start", ["short", "long1"])
        .edge("short", "join")
        .edge("long1", "long2")
        .edge("long2", "join")
        .end("join")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert result.state.note == "start|long1|short|long2"  # node-name order within a step
    assert result.visits["join"] == 1 and result.steps == 4


async def test_replace_conflict_in_one_superstep_fails_the_flow():
    flow = (
        Flow("conflict", state=State)
        .node("a", write())
        .node("b", write(note="x"))
        .node("c", write(note="y"))
        .node("d", Join(["b", "c"]))
        .edge("a", ["b", "c"])
        .edge("b", "d")
        .edge("c", "d")
        .end("d")
    )
    run, _, _ = runner()
    with pytest.raises(
        FlowError, match=r"superstep 2 \(nodes \['b', 'c'\]\): conflicting writes to note"
    ):
        await run.run(flow, {})


async def test_route_with_label_mapping_and_several_labels():
    flow = (
        Flow("routes", state=State)
        .node("pick", write())
        .node("x", write(log=["x"]))
        .node("y", write(log=["y"]))
        .route("pick", Route(lambda s: ["left", "right"], targets={"left": "x", "right": "y"}))
        .end("x", "y")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert result.state.log == ["x", "y"] and result.visits == {"pick": 1, "x": 1, "y": 1}


async def test_async_route_and_empty_route_end_the_branch():
    async def none(state):
        return []

    flow = (
        Flow("async_route", state=State)
        .node("a", write(n=1))
        .node("b", write(n=2))
        .route("a", Route(none, targets=["b"]))
        .end("b")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert result.status == "done" and result.state.n == 1 and "b" not in result.visits


async def test_route_unknown_label_and_failing_route_are_flow_errors():
    def broken(state):
        raise KeyError("secret")

    for fn, message in [
        (lambda s: "nowhere", "returned unknown label 'nowhere'"),
        (lambda s: [1], "returned unknown label 1"),
        (broken, "route from 'a' failed: KeyError"),
    ]:
        flow = (
            Flow("bad_route", state=State)
            .node("a", write())
            .node("b", write())
            .route("a", Route(fn, targets=["b"]))
            .end("b")
        )
        run, _, _ = runner()
        with pytest.raises(FlowError, match=message) as info:
            await run.run(flow, {})
        assert info.value.node == "a" and "secret" not in str(info.value)


def loop_flow(max_loops, on_exhausted="done"):
    return (
        Flow("loop", state=State)
        .node("work", lambda s: {"n": s.n + 1, "log": [f"work{s.n + 1}"]})
        .node("done", write(note="finished"))
        .route(
            "work",
            Route(
                lambda s: "again" if s.n < 10 else "stop", targets={"again": "work", "stop": "done"}
            ),
            max_loops=max_loops,
            on_exhausted=on_exhausted,
        )
        .end("done")
    )


async def test_loop_cap_goes_to_on_exhausted():
    run, _, _ = runner()
    result = await run.run(loop_flow(3), {})
    assert result.state.n == 4 and result.state.note == "finished"
    assert result.visits == {"work": 4, "done": 1} and result.steps == 5


async def test_loop_that_finishes_under_the_cap():
    run, _, _ = runner()
    result = await run.run(loop_flow(20), {"n": 8})
    assert result.state.n == 10 and result.visits == {"work": 2, "done": 1}


async def test_loop_cap_without_on_exhausted_raises():
    run, _, _ = runner()
    with pytest.raises(FlowLoopExceeded, match="node 'work' exceeded its loop limit 2") as info:
        await run.run(loop_flow(2, on_exhausted=None), {})
    assert info.value.flow == "loop" and info.value.limit == 2


async def test_max_steps_stops_a_long_run():
    run, _, _ = runner(max_steps=3)
    with pytest.raises(FlowLoopExceeded, match="loop limit 3"):
        await run.run(loop_flow(50), {})


async def test_runner_options_are_checked():
    _, executor, _ = runner()
    for bad in ({"max_steps": 0}, {"max_steps": True}, {"concurrency": 0}, {"hooks": [object()]}):
        with pytest.raises(ToolUsageError):
            FlowRunner(executor, **bad)


async def test_node_failure_is_a_flow_error_naming_the_node():
    def broken(state):
        raise ValueError("secret detail")

    flow = Flow("fails", state=State).node("a", broken).end("a")
    run, _, _ = runner()
    with pytest.raises(FlowError, match="flow 'fails' node 'a' failed: ValueError") as info:
        await run.run(flow, {})
    assert info.value.node == "a" and isinstance(info.value.__cause__, ValueError)
    assert "secret" not in str(info.value)


async def test_a_failing_branch_cancels_its_siblings():
    cancelled = []

    async def waits(state):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def fails(state):
        await asyncio.sleep(0)
        raise RuntimeError("x")

    flow = (
        Flow("cancel", state=State)
        .node("a", write())
        .node("b", waits)
        .node("c", fails)
        .edge("a", ["b", "c"])
        .end("b", "c")
    )
    run, _, _ = runner()
    with pytest.raises(FlowError, match="node 'c' failed"):
        await run.run(flow, {})
    assert cancelled == [True]


async def test_node_must_return_a_mapping_or_none():
    flow = Flow("returns", state=State).node("a", lambda s: ["not", "a", "mapping"]).end("a")
    run, _, _ = runner()
    with pytest.raises(FlowError, match="returned list, expected a mapping or None"):
        await run.run(flow, {})


async def test_unknown_update_keys_fail_the_superstep():
    flow = Flow("unknown", state=State).node("a", write(zzz=1)).end("a")
    run, _, _ = runner()
    with pytest.raises(FlowError, match=r"unknown state fields \['zzz'\]"):
        await run.run(flow, {})


async def test_nodes_see_their_context_and_the_current_context():
    seen = {}

    def sync_node(state, ctx):
        seen["sync"] = (ctx.call_id, current_context().call_id, ctx.run_id)

    async def async_node(state, ctx: ToolContext):
        seen["async"] = (ctx.call_id, current_context().call_id, ctx.run_id)

    flow = (
        Flow("ctx", state=State).node("s", sync_node).node("a", async_node).edge("s", "a").end("a")
    )
    run, executor, _ = runner()
    ctx = executor.context(run_id="run_fixed")
    await run.run(flow, {}, ctx)
    assert seen["sync"] == ("ctx:s:1", "ctx:s:1", "run_fixed")
    assert seen["async"] == ("ctx:a:1", "ctx:a:1", "run_fixed")


async def test_a_context_without_run_id_gets_one():
    seen = []
    flow = Flow("rid", state=State).node("a", lambda s, ctx: seen.append(ctx.run_id)).end("a")
    run, _, _ = runner()
    await run.run(flow, {}, ToolContext())
    await run.run(flow, {})
    assert all(rid.startswith("run_") for rid in seen) and seen[0] != seen[1]


async def test_callable_objects_and_star_args_are_called_with_the_context():
    class Node:
        async def __call__(self, state, ctx):
            return {"note": ctx.call_id}

    def star(*args):
        return {"n": len(args)}

    flow = (
        Flow("callables", state=State)
        .node("obj", Node())
        .node("star", star)
        .edge("obj", "star")
        .end("star")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert result.state.note == "callables:obj:1" and result.state.n == 2


async def test_sync_node_returning_an_awaitable_is_awaited():
    async def later():
        return {"n": 7}

    flow = Flow("awaitable", state=State).node("a", lambda s: later()).end("a")
    run, _, _ = runner()
    assert (await run.run(flow, {})).state.n == 7


async def test_concurrency_caps_parallel_branches():
    active = []
    peak = []

    async def branch(state):
        active.append(1)
        peak.append(len(active))
        await asyncio.sleep(0.005)
        active.pop()

    flow = Flow("cap", state=State).node("a", write())
    names = [f"b{i}" for i in range(4)]
    for name in names:
        flow.node(name, branch)
    flow.edge("a", names).end(*names)
    run, _, _ = runner(concurrency=2)
    await run.run(flow, {})
    assert max(peak) == 2
