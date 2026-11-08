"""layout(): layers, barycentre crossing reduction, x positions, bends for long edges, speed."""

from __future__ import annotations

import importlib
import random
import time
from typing import Any

from agent_adapter.viewer import layout

_layout = importlib.import_module("agent_adapter.viewer.layout")


def graph(
    edges: list[tuple[str, str]], *, loops: tuple[tuple[str, str], ...] = ()
) -> dict[str, Any]:
    names = list(dict.fromkeys(name for pair in (*edges, *loops) for name in pair))
    return {
        "nodes": [{"id": name} for name in names],
        "edges": [{"source": s, "target": t} for s, t in edges]
        + [{"source": s, "target": t, "loop": True} for s, t in loops],
    }


def by_id(placed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in placed["nodes"]}


def crossings(placed: dict[str, Any]) -> int:
    """Crossings between real nodes of adjacent layers (bends ignored)."""
    nodes = by_id(placed)
    segments = [
        (nodes[e["source"]]["layer"], nodes[e["source"]]["x"], nodes[e["target"]]["x"])
        for e in placed["edges"]
        if not e.get("loop") and nodes[e["target"]]["layer"] - nodes[e["source"]]["layer"] == 1
    ]
    return sum(
        1
        for i, (layer1, a1, b1) in enumerate(segments)
        for layer2, a2, b2 in segments[i + 1 :]
        if layer1 == layer2 and (a1 - a2) * (b1 - b2) < 0
    )


def test_layout_untangles_a_crossing_the_declaration_order_makes() -> None:
    edges = [("s", "a"), ("s", "b"), ("a", "d"), ("b", "c"), ("c", "e"), ("d", "f")]
    placed = layout(graph(edges))
    assert crossings(placed) == 0
    nodes = by_id(placed)
    assert nodes["d"]["x"] < nodes["c"]["x"]  # d hangs under a, which comes first
    assert [nodes[n]["layer"] for n in "sabcdef"] == [0, 1, 1, 2, 2, 3, 3]


def test_layout_centres_a_parent_over_its_children() -> None:
    nodes = by_id(layout(graph([("top", "l"), ("top", "m"), ("top", "r")])))
    assert nodes["top"]["x"] == nodes["m"]["x"] == 1.0
    assert (nodes["l"]["x"], nodes["r"]["x"]) == (0.0, 2.0)
    assert [nodes[n]["order"] for n in ("l", "m", "r")] == [0, 1, 2]


def test_a_long_edge_bends_around_the_nodes_it_skips() -> None:
    placed = layout(graph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]))
    nodes = by_id(placed)
    long_edge = next(e for e in placed["edges"] if (e["source"], e["target"]) == ("a", "d"))
    assert len(long_edge["via"]) == 2  # one bend per layer it crosses
    assert all(abs(x - nodes["b"]["x"]) >= 0.6 for x in long_edge["via"])
    assert all(e["via"] == [] for e in placed["edges"] if e is not long_edge)


def test_loop_edges_and_unknown_or_self_edges_do_not_shape_the_layout() -> None:
    placed = layout(graph([("a", "b"), ("b", "c")], loops=(("c", "a"),)))
    placed_edges = {(e["source"], e["target"]): e for e in placed["edges"]}
    assert placed_edges[("c", "a")]["via"] == []
    assert [by_id(placed)[n]["layer"] for n in "abc"] == [0, 1, 2]
    odd = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "ghost"},
            {"source": "b", "target": "b"},
        ],
    }
    assert [n["layer"] for n in layout(odd)["nodes"]] == [0, 1]
    assert layout(odd)["width"] == 1.0


def test_a_route_and_an_edge_to_the_same_target_share_their_bends() -> None:
    base = graph([("a", "b"), ("b", "c"), ("a", "c")])
    base["edges"].append({"source": "a", "target": "c", "kind": "route"})
    placed = layout(base)
    vias = [e["via"] for e in placed["edges"] if (e["source"], e["target"]) == ("a", "c")]
    assert len(vias) == 2 and vias[0] == vias[1] and len(vias[0]) == 1


def test_layout_keeps_extra_graph_keys_and_is_deterministic() -> None:
    source = {**graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]), "flow": "f"}
    first, second = layout(source), layout(source)
    assert first == second and first["flow"] == "f"
    assert first["layers"] == 3 and first["width"] == 2.0
    assert "x" not in source["nodes"][0]  # the input is not changed


def test_the_order_with_the_fewest_crossings_wins() -> None:
    rows = [["a", "b"], ["c", "d"]]
    down = {"a": ["d"], "b": ["c"], "c": [], "d": []}
    up = {"a": [], "b": [], "c": ["b"], "d": ["a"]}
    assert _layout._crossings(rows, down) == 1
    assert _layout._crossings(_layout._order(rows, down, up), down) == 0


def flow_shaped(stages: int, fan: int) -> dict[str, Any]:
    """A plan -> fan-out -> join pipeline per stage, with skip edges and loops."""
    rng = random.Random(11)
    edges: list[tuple[str, str]] = []
    loops: list[tuple[str, str]] = []
    previous = "start"
    for stage in range(stages):
        branches = [f"s{stage}b{i}" for i in range(fan)]
        join = f"s{stage}join"
        edges += [(previous, b) for b in branches] + [(b, join) for b in branches]
        if stage >= 2 and rng.random() < 0.5:
            edges.append((f"s{stage - 2}join", join))
        if stage >= 1 and rng.random() < 0.3:
            loops.append((join, f"s{stage - 1}join"))
        previous = join
    return graph(edges, loops=tuple(loops))


def test_layout_of_200_nodes_takes_under_50_ms() -> None:
    big = flow_shaped(stages=20, fan=9)
    assert len(big["nodes"]) == 201
    best = min(_timed(big) for _ in range(3))
    assert best < 0.05, f"layout took {best * 1000:.1f} ms"
    placed = layout(big)
    assert crossings(placed) == 0


def test_a_dense_random_graph_still_lays_out_fast() -> None:
    rng = random.Random(7)
    edges = [(f"n{i}", f"n{min(199, i + rng.randint(1, 6))}") for i in range(199) for _ in range(2)]
    dense = graph(edges)
    # ~0.05 s plain, ~2x under coverage tracing; a quadratic regression takes seconds.
    assert min(_timed(dense) for _ in range(3)) < 0.5


def _timed(source: dict[str, Any]) -> float:
    start = time.perf_counter()
    layout(source)
    return time.perf_counter() - start


def test_a_cycle_not_marked_as_a_loop_is_drawn_as_one() -> None:
    placed = layout(graph([("s", "a"), ("a", "b"), ("b", "a"), ("b", "d")]))
    nodes = by_id(placed)
    assert [nodes[n]["layer"] for n in "sabd"] == [0, 1, 2, 3]
    loops = [(e["source"], e["target"]) for e in placed["edges"] if e.get("loop")]
    assert loops == [("b", "a")]
    assert all(e["via"] == [] for e in placed["edges"])


def test_duplicate_node_ids_share_one_place() -> None:
    twice = {
        "nodes": [{"id": "a"}, {"id": "a"}, {"id": "b"}],
        "edges": [{"source": "a", "target": "b"}],
    }
    placed = layout(twice)
    assert [n["layer"] for n in placed["nodes"]] == [0, 0, 1]
