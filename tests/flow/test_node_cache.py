"""`cache=NodeCache(...)`: hits skip the node, keys, scopes, signed entries and safety rules."""

import json
import logging
from typing import Annotated

import pytest
from pydantic import BaseModel

from agent_adapter.approval import AutoApprove
from agent_adapter.cache import MemoryCache
from agent_adapter.core.errors import FlowDefinitionError, FlowError
from agent_adapter.flow import Call, Flow, FlowRunner, Map, Merge, NodeCache, SubFlow, _cache
from tests.fakes.sinks import RecordingSink
from tests.flow.helpers import CALLS, registry, runner
from tests.flow.subflow_helpers import Order, child_flow

RAN: list[str] = []


class Quote(BaseModel):
    item: str = ""
    qty: int = 0
    price: int = 0
    notes: Annotated[list[str], Merge.append] = []


def price(state: Quote) -> dict:
    RAN.append(state.item)
    return {"price": len(state.item) * 10, "notes": [f"priced {state.item}"]}


def quote_flow(cache: NodeCache, fn=price) -> Flow[Quote]:
    return Flow("quote", state=Quote).node("price", fn, cache=cache).end("price")


@pytest.fixture(autouse=True)
def _reset():
    RAN.clear()
    _cache._WARNED.clear()


def cached_runner(**options):
    return runner(cache=MemoryCache(), **options)


async def test_a_hit_skips_the_node_and_returns_the_same_updates():
    sink = RecordingSink()
    run, _, _ = cached_runner(sink=sink)
    flow = quote_flow(NodeCache(scope="global"))
    first = await run.run(flow, {"item": "mug"})
    second = await run.run(flow, {"item": "mug"})
    assert RAN == ["mug"] and second.state == first.state
    assert second.state.notes == ["priced mug"]
    cached = [e for e in sink.events if e.type == "node.cached"]
    assert len(cached) == 1 and cached[0].data == {"flow": "quote", "node": "price", "visit": 1}
    finished = [e for e in sink.events if e.type == "node.finished"]
    assert "cached" not in finished[0].data and finished[1].data["cached"] is True


async def test_other_inputs_miss_and_reads_narrow_the_key():
    run, _, _ = cached_runner()
    whole = quote_flow(NodeCache(scope="global"))
    await run.run(whole, {"item": "mug", "qty": 1})
    await run.run(whole, {"item": "mug", "qty": 2})  # another state: a miss
    narrow = quote_flow(NodeCache(scope="global", reads=["item"]))
    await run.run(narrow, {"item": "cup", "qty": 1})
    await run.run(narrow, {"item": "cup", "qty": 2})  # qty is not read: a hit
    assert RAN == ["mug", "mug", "cup"]


async def test_a_key_function_decides_the_key_and_its_failure_fails_the_node():
    run, _, _ = cached_runner()
    flow = quote_flow(NodeCache(scope="global", key=lambda s: s.item.lower()))
    await run.run(flow, {"item": "Mug"})
    await run.run(flow, {"item": "MUG"})
    assert RAN == ["Mug"]
    broken = quote_flow(NodeCache(key=lambda s: 1 / 0))
    with pytest.raises(FlowError, match="cache key failed: ZeroDivisionError") as caught:
        await run.run(broken, {"item": "x"})
    assert caught.value.node == "price"


async def test_run_scope_is_the_default_and_thread_scope_needs_a_thread():
    run, executor, _ = cached_runner()
    flow = quote_flow(NodeCache())
    await run.run(flow, {"item": "a"})
    await run.run(flow, {"item": "a"})  # a new run: the run-scoped entry is not shared
    assert RAN == ["a", "a"]
    threaded = quote_flow(NodeCache(scope="thread"))
    await run.run(threaded, {"item": "b"})  # no thread id: not cached
    await run.run(threaded, {"item": "b"}, executor.context(thread_id="t1"))
    await run.run(threaded, {"item": "b"}, executor.context(thread_id="t1"))
    assert RAN == ["a", "a", "b", "b"]


async def test_the_ttl_reaches_the_cache_port():
    ttls = []

    class Port(MemoryCache):
        async def set(self, key, value, ttl=None):
            ttls.append(ttl)
            await super().set(key, value, ttl)

    run, _, _ = runner(cache=Port())
    await run.run(quote_flow(NodeCache(ttl=30, scope="global")), {"item": "x"})
    assert ttls == [30]


async def test_without_a_cache_port_the_node_runs_and_says_so_once(caplog):
    run, _, _ = runner()
    flow = quote_flow(NodeCache(scope="global"))
    with caplog.at_level(logging.WARNING, logger="agent_adapter.flow._cache"):
        await run.run(flow, {"item": "x"})
        await run.run(flow, {"item": "x"})
    assert RAN == ["x", "x"]
    assert [r.getMessage() for r in caplog.records] == [
        "flow quote node price: cache= but no cache port; not cached"
    ]


class Broken(MemoryCache):
    async def get(self, key):
        raise ConnectionError("down")

    async def set(self, key, value, ttl=None):
        raise ConnectionError("down")


async def test_a_failing_backend_is_a_miss(caplog):
    run, _, _ = runner(cache=Broken())
    flow = quote_flow(NodeCache(scope="global"))
    with caplog.at_level(logging.WARNING):
        done = await run.run(flow, {"item": "x"})
    assert done.state.price == 10 and RAN == ["x"]
    assert any("node cache get failed" in r.getMessage() for r in caplog.records)
    assert any("node cache set failed" in r.getMessage() for r in caplog.records)


async def test_an_entry_with_a_wrong_signature_or_shape_is_ignored(caplog):
    port = MemoryCache()
    run, _, _ = runner(cache=port)
    flow = quote_flow(NodeCache(scope="global"))
    await run.run(flow, {"item": "x"})
    [key] = list(port._data)
    entry = json.loads(port._data[key][0])
    entry["body"]["items"][0]["price"] = 999
    for bad in (json.dumps(entry).encode(), b"not json", b'{"v": 9}'):
        await port.set(key, bad)
        with caplog.at_level(logging.WARNING):
            done = await run.run(flow, {"item": "x"})
        assert done.state.price == 10
    assert RAN == ["x", "x", "x", "x"]
    assert any("node cache entry for quote.price ignored" in r.getMessage() for r in caplog.records)


async def test_entries_signed_by_another_runner_key_are_not_trusted():
    port = MemoryCache()
    flow = quote_flow(NodeCache(scope="global"))
    first = FlowRunner(runner(cache=port)[1], checkpoint_key=b"k" * 16)
    same = FlowRunner(runner(cache=port)[1], checkpoint_key=b"k" * 16)
    other = FlowRunner(runner(cache=port)[1], checkpoint_key=b"o" * 16)
    for run in (first, same, other):
        await run.run(flow, {"item": "x"})
    assert RAN == ["x", "x"]  # the same key read the entry; the other key ran it again


async def test_values_without_a_canonical_form_are_not_cached():
    class Box(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        thing: object = None

    def make(state):
        RAN.append("made")
        return {"thing": object()}

    flow = Flow("box", state=Box).node("make", make, cache=NodeCache(scope="global")).end("make")
    run, _, _ = cached_runner()
    await run.run(flow, {})
    await run.run(flow, {})
    assert RAN == ["made", "made"]


async def test_a_map_node_caches_its_items_and_view():
    class Hits(BaseModel):
        queries: list[str] = []
        search: Annotated[list, Merge.append] = []

    flow = (
        Flow("many", state=Hits)
        .node(
            "search",
            Map("queries", tool="search", args_fn=lambda q: {"query": q}),
            cache=NodeCache(scope="global"),
        )
        .end("search")
    )
    run, _, _ = cached_runner()
    CALLS.clear()
    first = await run.run(flow, {"queries": ["a", "b"]})
    second = await run.run(flow, {"queries": ["a", "b"]})
    assert second.state.search == first.state.search and len(first.state.search) == 4
    assert sorted(CALLS) == ["a", "b"]  # the second run's items came from the cache


async def test_a_cached_subflow_reuses_the_whole_child_run():
    sink = RecordingSink()
    flow = (
        Flow("outer", state=Order)
        .node("sub", SubFlow(child_flow()), cache=NodeCache(scope="global"))
        .end("sub")
    )
    run, _, _ = cached_runner(sink=sink)
    await run.run(flow, {"item": "a"})
    done = await run.run(flow, {"item": "a"})
    assert done.state.note == "checked 1"
    child_starts = [
        e for e in sink.events if e.type == "flow.started" and e.data["flow"] == "child"
    ]
    assert len(child_starts) == 1


async def test_a_node_that_asked_for_approval_is_never_stored(caplog):
    executor_holder = {}

    async def pays(state, ctx):
        RAN.append("pays")
        return {"note": await executor_holder["e"].run("pay", {"amount": 1}, ctx)}

    # key= covers the captured executor holder, so the approval watch is what stops the store
    cache = NodeCache(scope="global", key=lambda s: "same")
    flow = Flow("payer", state=Order).node("go", pays, cache=cache).end("go")
    run, executor, _ = cached_runner(approval=AutoApprove())
    executor_holder["e"] = executor
    with caplog.at_level(logging.WARNING):
        await run.run(flow, {})
        await run.run(flow, {})
    assert RAN == ["pays", "pays"]
    assert any("asked for approval; not cached" in r.getMessage() for r in caplog.records)


def test_cache_options_and_unsafe_nodes_are_rejected():
    def p(flow):
        with pytest.raises(FlowDefinitionError) as caught:
            flow.validate()
        return caught.value.problems

    def on(cache, action=price, **options):
        return Flow("q", state=Quote).node("n", action, cache=cache, **options).end("n")

    assert p(on("yes")) == ["node 'n': cache must be a NodeCache, got str"]
    assert p(on(NodeCache(ttl=0))) == [
        "node 'n': cache ttl must be a positive number or None, got 0"
    ]
    assert p(on(NodeCache(ttl=True))) == [
        "node 'n': cache ttl must be a positive number or None, got True"
    ]
    assert p(on(NodeCache(key=3))) == ["node 'n': cache key must be callable, got 3"]
    assert p(on(NodeCache(key=len, reads=["item"]))) == [
        "node 'n': cache takes key= or reads=, not both"
    ]
    assert p(on(NodeCache(reads="item"))) == [
        "node 'n': cache reads must be a non-empty list of state fields, got 'item'"
    ]
    assert p(on(NodeCache(reads=["nope"]))) == ["node 'n': cache reads 'nope', not a state field"]
    assert p(on(NodeCache(scope="world"))) == [
        "node 'n': cache scope must be one of ['run', 'thread', 'global'], got 'world'"
    ]
    assert p(on(NodeCache(), pause=True)) == [
        "node 'n': cache cannot be used on a node with a pause gate"
    ]
    assert p(on(NodeCache(), Call(registry.get("pay"), args={"amount": 1}))) == [
        "node 'n': cache cannot be used on a node whose tool needs approval"
    ]
    gated = (
        Flow("o", state=Order).node("n", SubFlow(child_flow(gate=True)), cache=NodeCache()).end("n")
    )
    assert p(gated) == [
        "node 'n': cache cannot be used on a sub-flow with pause gates or approvals"
    ]


async def test_a_cached_node_naming_a_confirm_tool_is_refused_by_the_runner():
    flow = (
        Flow("q", state=Order)
        .node("n", Call("pay", args={"amount": 1}), cache=NodeCache())
        .end("n")
    )
    inner = Flow("o", state=Order).node("sub", SubFlow(flow)).end("sub")
    run, _, _ = cached_runner()
    for target in (flow, inner):
        with pytest.raises(FlowDefinitionError, match="cache cannot be used with tool 'pay'"):
            await run.run(target, {})
    unknown = Flow("u", state=Order).node("n", Call("nope"), cache=NodeCache()).end("n")
    with pytest.raises(FlowError):
        await run.run(unknown, {})  # an unknown tool fails when the node runs, as uncached


async def test_a_runner_without_a_registry_skips_the_tool_check():
    from agent_adapter.exec import Executor

    flow = Flow("q", state=Quote).node("n", price, cache=NodeCache()).end("n")
    done = await FlowRunner(Executor(cache=MemoryCache())).run(flow, {"item": "x"})
    assert done.state.price == 10 and registry is not None
