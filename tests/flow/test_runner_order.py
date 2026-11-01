from agent_adapter.flow import Flow, Route
from tests.flow.helpers import State, runner


def write(**update):
    return lambda state: dict(update)


async def test_a_route_sees_its_own_writes_but_not_its_siblings():
    seen = []

    def pick(state):
        seen.append((state.n, state.note))
        return "yes" if state.note == "set by a" else "no"

    flow = (
        Flow("route_view", state=State)
        .node("start", write())
        .node("a", write(note="set by a"))
        .node("b", write(n=7))
        .node("yes", write(log=["yes"]))
        .node("no", write(log=["no"]))
        .edge("start", ["a", "b"])
        .edge("a", "yes")
        .route("b", Route(pick, targets=["yes", "no"]))
        .end("yes", "no")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert seen == [(7, "")]  # b's own write, not a's
    assert result.state.log == ["no", "yes"] and result.state.note == "set by a"


async def test_cross_node_writes_merge_in_node_name_order():
    flow = (
        Flow("collide", state=State)
        .node("start", write())
        .node("zeta", write(hits=[{"source_id": "x", "text": "zeta"}], log=["zeta"]))
        .node("alpha", write(hits=[{"source_id": "x", "text": "alpha"}], log=["alpha"]))
        .edge("start", ["zeta", "alpha"])
        .end("zeta", "alpha")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert [h.text for h in result.state.hits] == ["zeta"]  # alpha first, zeta wins the key
    assert result.state.log == ["alpha", "zeta"]


async def test_equal_replace_writes_from_two_nodes_are_fine():
    flow = (
        Flow("same", state=State)
        .node("start", write())
        .node("a", write(note="same"))
        .node("b", write(note="same"))
        .edge("start", ["a", "b"])
        .end("a", "b")
    )
    run, _, _ = runner()
    assert (await run.run(flow, {})).state.note == "same"


async def test_picking_a_node_running_in_the_same_superstep_is_not_a_loop():
    flow = (
        Flow("diamond", state=State)
        .node("a", write())
        .node("b", write(log=["b"]))
        .node("c", write(log=["c"]))
        .node("d", write(log=["d"]))
        .edge("a", ["b", "c"])
        .route("b", Route(lambda s: "c", targets=["c"]), max_loops=0)
        .edge("c", "d")
        .end("d")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})  # max_loops=0 would raise if this counted as a loop
    assert result.visits == {"a": 1, "b": 1, "c": 2, "d": 2}


async def test_a_route_back_to_itself_counts_a_loop():
    flow = (
        Flow("self_loop", state=State)
        .node("a", lambda s: {"n": s.n + 1})
        .node("z", write())
        .route("a", Route(lambda s: "a", targets=["a", "z"]), max_loops=1, on_exhausted="z")
        .end("z")
    )
    run, _, _ = runner()
    result = await run.run(flow, {})
    assert result.state.n == 2 and result.visits == {"a": 2, "z": 1}
