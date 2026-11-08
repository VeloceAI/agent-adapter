"""Review regressions: entries bound to their key, captured config in the key, additive rules."""

import functools
import logging
from collections.abc import Sequence
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from agent_adapter.cache import MemoryCache
from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.flow import Call, Flow, FlowRunner, NodeCache, SubFlow, _cache
from agent_adapter.flow.state import Merge, is_additive
from tests.flow.helpers import registry, runner


class Price(BaseModel):
    item: str = ""
    price: int = 0


@pytest.fixture(autouse=True)
def _reset():
    _cache._WARNED.clear()


def make(rate: int) -> Flow[Price]:
    def quote(state: Price) -> dict:
        return {"price": rate}

    return Flow("pricing", state=Price).node("q", quote, cache=NodeCache(scope="global")).end("q")


async def test_an_entry_moved_to_another_key_does_not_verify():
    port = MemoryCache()
    run, executor, _ = runner(cache=port)
    calls = []

    def quote(state):
        calls.append(state.item)
        return {"price": 10}

    flow = Flow("t", state=Price).node("q", quote, cache=NodeCache(scope="thread")).end("q")
    await run.run(flow, {"item": "x"}, executor.context(thread_id="tenant-a"))
    [key_a] = list(port._data)
    key_b = key_a.replace("thread=tenant-a", "thread=tenant-b")
    await port.set(key_b, port._data[key_a][0])  # a writer copies A's signed entry to B's key
    await run.run(flow, {"item": "x"}, executor.context(thread_id="tenant-b"))
    assert calls == ["x", "x"]  # B ran its own node; the copied entry did not verify


async def test_flows_whose_closures_captured_other_values_do_not_share_entries():
    run, _, _ = runner(cache=MemoryCache())
    first = await run.run(make(10), {})
    second = await run.run(make(99), {})
    again = await run.run(make(10), {})
    assert (first.state.price, second.state.price, again.state.price) == (10, 99, 10)


async def test_partials_with_other_arguments_do_not_share_entries():
    def fetch(state: Price, *, rate: int) -> dict:
        return {"price": rate}

    def flow(rate: int) -> Flow[Price]:
        node = functools.partial(fetch, rate=rate)
        return Flow("p", state=Price).node("q", node, cache=NodeCache(scope="global")).end("q")

    run, _, _ = runner(cache=MemoryCache())
    assert (await run.run(flow(1), {})).state.price == 1
    assert (await run.run(flow(2), {})).state.price == 2


async def test_static_call_arguments_are_part_of_the_key():
    def flow(query: str) -> Flow[Any]:
        class Hits(BaseModel):
            found: Any = None

        call = Call("search", args={"query": query}, into="found")
        return Flow("s", state=Hits).node("q", call, cache=NodeCache(scope="global")).end("q")

    run, _, _ = runner(cache=MemoryCache())
    a = await run.run(flow("a"), {})
    b = await run.run(flow("b"), {})
    assert a.state.found[0]["text"] == "a" and b.state.found[0]["text"] == "b"


NO_FORM = object()  # a default with no canonical form


class Service:
    def __init__(self, rate: Any) -> None:
        self.rate = rate

    def quote(self, state: Price) -> dict:
        return {"price": self.rate if isinstance(self.rate, int) else 0}


def cached(node: Any, **cache: Any) -> Flow[Price]:
    options = NodeCache(scope="global", **cache)
    return Flow("svc", state=Price).node("q", node, cache=options).end("q")


async def prices(run: Any, *flows: Flow[Price]) -> list[int]:
    return [(await run.run(flow, {})).state.price for flow in flows]


async def test_a_bound_methods_object_attributes_are_part_of_the_key():
    run, _, _ = runner(cache=MemoryCache())
    flows = [cached(Service(r).quote) for r in (1, 2, 1)]
    assert await prices(run, *flows) == [1, 2, 1]


async def test_captured_values_without_a_canonical_form_are_not_cached(caplog):
    run, _, _ = runner(cache=MemoryCache())
    with caplog.at_level(logging.WARNING, logger="agent_adapter.flow._cache"):
        await prices(run, cached(Service(object()).quote))
    assert any("captures values with no canonical form" in r.getMessage() for r in caplog.records)
    # with key= the user covers what the code captured, so it caches under that key
    calls = []

    def counted(state: Price, *, conn: Any = NO_FORM) -> dict:
        calls.append(1)
        return {"price": 1}

    flow = cached(counted, key=lambda s: "k")
    assert await prices(run, flow, flow) == [1, 1] and calls == [1]


async def test_defaults_and_keyword_defaults_are_part_of_the_key():
    def by_default(rate: int) -> Any:
        def quote(state: Price, ctx: Any, r: int = rate) -> dict:
            return {"price": r}

        return quote

    def by_keyword(rate: int) -> Any:
        def quote(state: Price, *, r: int = rate) -> dict:
            return {"price": r}

        return quote

    run, _, _ = runner(cache=MemoryCache())
    for make_fn in (by_default, by_keyword):
        assert await prices(run, *(cached(make_fn(r)) for r in (10, 99, 10))) == [10, 99, 10]


class Quoter:
    def __init__(self, rate: int) -> None:
        self.rate = rate

    def __call__(self, state: Price, *, extra: int = 0) -> dict:
        return {"price": self.rate + extra}


async def test_a_callable_objects_attributes_and_partial_keywords_are_part_of_the_key():
    run, _, _ = runner(cache=MemoryCache())
    assert await prices(run, *(cached(Quoter(r)) for r in (3, 4))) == [3, 4]
    parts = [functools.partial(Quoter(1), extra=e) for e in (5, 6)]
    assert await prices(run, *(cached(p) for p in parts)) == [6, 7]


async def test_a_subflow_childs_captured_values_count_in_its_key():
    def child(rate: int) -> Flow[Price]:
        return Flow("c", state=Price).node("q", lambda s: {"price": rate}).end("q")

    def outer(rate: int) -> Flow[Price]:
        sub = SubFlow(child(rate))
        return Flow("o", state=Price).node("s", sub, cache=NodeCache(scope="global")).end("s")

    run = FlowRunner(runner(cache=MemoryCache())[1])
    assert (await run.run(outer(3), {})).state.price == 3
    assert (await run.run(outer(4), {})).state.price == 4


class Tally:
    """A custom additive rule: counts every write."""

    name = "tally"
    additive = True

    def merge(self, current: Any, updates: Sequence[Any]) -> Any:
        return (current or 0) + sum(updates)


class Counts(BaseModel):
    hits: Annotated[int, Tally()] = 0


def test_a_rule_that_declares_itself_additive_is_treated_like_sum():
    assert is_additive(Tally()) and is_additive(Merge.sum) and not is_additive(Merge.replace)
    child = Flow("c", state=Counts).node("n", lambda s: {"hits": 1}).end("n")
    twice = Flow("p", state=Counts).node("s", SubFlow(child, input={"hits": "hits"})).end("s")
    with pytest.raises(FlowDefinitionError, match="merge rule 'tally'"):
        twice.validate()


async def test_a_custom_additive_field_starts_empty_in_the_child():
    child = Flow("c", state=Counts).node("n", lambda s: {"hits": s.hits + 1}).end("n")
    flow = (
        Flow("p", state=Counts)
        .node("start", lambda s: {"hits": 5})
        .node("s", SubFlow(child))
        .edge("start", "s")
        .end("s")
    )
    run, _, _ = runner()
    assert (await run.run(flow, {})).state.hits == 6  # 5, plus the child's 1 counted once
    assert registry is not None
