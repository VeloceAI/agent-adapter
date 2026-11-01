"""`call_route` and `pick_targets`: the two halves of a route, shared with the ADK driver."""

import pytest

from agent_adapter.core.errors import FlowError, FlowLoopExceeded
from agent_adapter.flow._step import call_route, pick_targets
from agent_adapter.flow.graph import RouteSpec
from tests.flow.parity_flows import Board


def spec(fn, **options):
    base = {"source": "a", "fn": fn, "targets": {"x": "b", "self": "a"}}
    return RouteSpec(**{"max_loops": None, "on_exhausted": None, **base, **options})


async def test_call_route_takes_sync_async_one_or_many_labels():
    async def both(state):
        return ("x", "self")

    assert await call_route("f", spec(lambda s: "x"), Board()) == ["x"]
    assert await call_route("f", spec(both), Board()) == ["x", "self"]


async def test_call_route_wraps_a_failure():
    def broken(state):
        raise KeyError("k")

    with pytest.raises(FlowError, match="route from 'a' failed: KeyError"):
        await call_route("f", spec(broken), Board())


def test_pick_targets_counts_loops_and_refuses_unknown_labels():
    loops: dict[str, int] = {}
    route = spec(None, max_loops=1, on_exhausted="b")
    assert pick_targets("f", route, ["self"], {}, loops) == ["a"]
    assert pick_targets("f", route, ["self"], {}, loops) == ["b"] and loops == {"a": 2}
    with pytest.raises(FlowError, match="unknown label"):
        pick_targets("f", route, ["nope"], {}, {})
    with pytest.raises(FlowLoopExceeded):
        pick_targets("f", spec(None, max_loops=0), ["x"], {"b": 1}, {})
