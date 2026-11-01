"""A layered layout for a flow graph: layers, barycentre crossing reduction, x positions."""

from __future__ import annotations

import bisect
import itertools
from collections import deque
from collections.abc import Mapping
from typing import Any

SWEEPS = 6  # down + up barycentre passes; the order with the fewest crossings wins
PLACE_ROUNDS = 4  # coordinate passes pulling each item towards its neighbours
_GAP = {(True, True): 1.0, (True, False): 0.6, (False, True): 0.6, (False, False): 0.35}
_DUMMY_WEIGHT = 4.0  # long edges stay straight before short ones do
WORK = 6_000  # item passes per layout; big graphs (many long edges) get fewer passes

Rows = list[list[str]]
Links = dict[str, list[str]]


def layout(graph: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of `graph`: nodes get `layer`, `order` and `x`; edges get `via` (bend x per layer).

    Loop edges are ignored for depth and order; an edge that closes a cycle in a graph not
    marked so (a hand-edited or foreign file) is marked `loop` here. `x` is in node slots (one
    node and its gap); `width` is the span of `x` plus one slot.
    """
    nodes = [dict(node) for node in graph.get("nodes", [])]
    edges = [dict(edge) for edge in graph.get("edges", [])]
    names = list(dict.fromkeys(str(node["id"]) for node in nodes))
    known = set(names)
    pairs: dict[tuple[str, str], None] = {}
    for edge in edges:
        source, target = str(edge.get("source")), str(edge.get("target"))
        if not edge.get("loop") and source in known and target in known and source != target:
            pairs[(source, target)] = None
    back = _back_pairs(names, list(pairs))
    for edge in edges:
        if (str(edge.get("source")), str(edge.get("target"))) in back:
            edge["loop"] = True
    pairs = {pair: None for pair in pairs if pair not in back}
    layers = _layers(names, list(pairs))
    rows, down, up, chains = _proper(names, layers, list(pairs))
    rows = _order(rows, down, up)
    x = _place(rows, down, up, known)
    low = min(x.values(), default=0.0)
    for name in x:
        x[name] = round(x[name] - low, 3)
    for node in nodes:
        name = str(node["id"])
        node["layer"] = layers[name]
        node["order"] = [n for n in rows[layers[name]] if n in known].index(name)
        node["x"] = x[name]
    for edge in edges:
        chain = chains.get((str(edge.get("source")), str(edge.get("target"))), [])
        edge["via"] = [] if edge.get("loop") else [x[item] for item in chain]
    return {
        **graph,
        "nodes": nodes,
        "edges": edges,
        "layers": len(rows),
        "width": round(max(x.values(), default=-1.0) + 1.0, 3) if x else 0,
    }


def _back_pairs(names: list[str], pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Edges that lead back to a node on the depth-first path: dropping them leaves a DAG."""
    successors: Links = {name: [] for name in names}
    for source, target in pairs:
        successors[source].append(target)
    state: dict[str, int] = {}  # 1 = on the path, 2 = done
    back: set[tuple[str, str]] = set()
    for root in names:
        if root in state:
            continue
        state[root] = 1
        stack = [(root, iter(successors[root]))]
        while stack:
            node, todo = stack[-1]
            following = next(todo, None)
            if following is None:
                state[node] = 2
                stack.pop()
            elif state.get(following) == 1:
                back.add((node, following))
            elif following not in state:
                state[following] = 1
                stack.append((following, iter(successors[following])))
    return back


def _layers(names: list[str], pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Longest-path depth over the forward (acyclic) edges, in topological order."""
    successors: Links = {name: [] for name in names}
    incoming = dict.fromkeys(names, 0)
    for source, target in pairs:
        successors[source].append(target)
        incoming[target] += 1
    depth = dict.fromkeys(names, 0)
    ready = deque(name for name in names if incoming[name] == 0)
    while ready:
        name = ready.popleft()
        for target in successors[name]:
            depth[target] = max(depth[target], depth[name] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    return depth


def _proper(
    names: list[str], layers: dict[str, int], pairs: list[tuple[str, str]]
) -> tuple[Rows, Links, Links, dict[tuple[str, str], list[str]]]:
    """Rows with a dummy item per layer an edge crosses, so every link spans one layer."""
    count = max(layers.values(), default=-1) + 1
    rows: Rows = [[] for _ in range(count)]
    for name in names:
        rows[layers[name]].append(name)
    down: Links = {name: [] for name in names}
    up: Links = {name: [] for name in names}
    chains: dict[tuple[str, str], list[str]] = {}
    for number, (source, target) in enumerate(pairs):
        chain = [f"\0{number}:{depth}" for depth in range(layers[source] + 1, layers[target])]
        for depth, item in zip(range(layers[source] + 1, layers[target]), chain, strict=True):
            rows[depth].append(item)
            down[item], up[item] = [], []
        chains[(source, target)] = chain
        path = [source, *chain, target]
        for above, below in itertools.pairwise(path):
            down[above].append(below)
            up[below].append(above)
    return rows, down, up, chains


def _order(rows: Rows, down: Links, up: Links) -> Rows:
    """Alternating barycentre sweeps; keeps the row orders with the fewest crossings."""
    rows = [list(row) for row in rows]
    best = [list(row) for row in rows]
    fewest = _crossings(rows, down)
    sweeps = _passes(rows, SWEEPS)
    for _ in range(sweeps):
        if fewest == 0:
            break
        for depth in range(1, len(rows)):
            _sort_row(rows[depth], rows[depth - 1], up)
        for depth in range(len(rows) - 2, -1, -1):
            _sort_row(rows[depth], rows[depth + 1], down)
        found = _crossings(rows, down)
        if found < fewest:
            fewest, best = found, [list(row) for row in rows]
    return best


def _passes(rows: Rows, most: int) -> int:
    """`most` passes, fewer (at least one) once the rows hold many nodes and bends."""
    return max(1, min(most, WORK // max(1, sum(map(len, rows)))))


def _sort_row(row: list[str], fixed: list[str], links: Links) -> None:
    """Sort `row` by the mean position of its links in `fixed`; ties keep their order."""
    at = {name: i for i, name in enumerate(fixed)}
    keys = {}
    for i, name in enumerate(row):
        near = [at[n] for n in links[name]]
        keys[name] = (sum(near) / len(near) if near else float(i), i)
    row.sort(key=keys.__getitem__)


def _crossings(rows: Rows, down: Links) -> int:
    """Edge crossings between each pair of adjacent rows, counted as inversions."""
    total = 0
    for upper, lower in itertools.pairwise(rows):
        below = {name: i for i, name in enumerate(lower)}
        ends = sorted((i, below[t]) for i, name in enumerate(upper) for t in down[name])
        seen: list[int] = []
        for _, end in ends:
            total += len(seen) - bisect.bisect_right(seen, end)
            bisect.insort(seen, end)
    return total


def _place(rows: Rows, down: Links, up: Links, real: set[str]) -> dict[str, float]:
    """x per item: each row pulled towards its neighbours' mean, order and gaps kept."""
    offsets = [_offsets(row, real) for row in rows]
    weights = [[1.0 if name in real else _DUMMY_WEIGHT for name in row] for row in rows]
    x: dict[str, float] = {}
    for row, row_offsets in zip(rows, offsets, strict=True):
        x.update(zip(row, row_offsets, strict=True))
    rounds = _passes(rows, PLACE_ROUNDS)
    for _ in range(rounds):
        for depth in range(1, len(rows)):
            _pull(rows[depth], offsets[depth], weights[depth], up, x)
        for depth in range(len(rows) - 2, -1, -1):
            _pull(rows[depth], offsets[depth], weights[depth], down, x)
    return x


def _offsets(row: list[str], real: set[str]) -> list[float]:
    """Each item's least distance from the row's first item."""
    out = [0.0]
    for left, right in itertools.pairwise(row):
        out.append(out[-1] + _GAP[(left in real, right in real)])
    return out


def _pull(
    row: list[str], offsets: list[float], weights: list[float], links: Links, x: dict[str, float]
) -> None:
    """The weighted least-squares fit to the neighbour means that keeps the gaps (PAVA)."""
    blocks: list[list[float]] = []  # [weighted sum, weight, items]
    for name, offset, weight in zip(row, offsets, weights, strict=True):
        near = links[name]
        if len(near) == 1:  # every bend and most nodes: no mean to take
            wanted = x[near[0]]
        else:
            wanted = sum(x[n] for n in near) / len(near) if near else x[name]
        blocks.append([(wanted - offset) * weight, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] > blocks[-1][0] * blocks[-2][1]:
            last = blocks.pop()
            head = blocks[-1]
            head[0] += last[0]
            head[1] += last[1]
            head[2] += last[2]
    index = 0
    for total, weight, size in blocks:
        mean = total / weight
        for _ in range(int(size)):
            x[row[index]] = mean + offsets[index]
            index += 1
