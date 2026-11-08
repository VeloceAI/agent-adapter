"""CallableMemory: host functions, sync or async, with the shared checks and typed errors."""

import pytest

from agent_adapter.errors import IntegrationError, ToolUsageError
from agent_adapter.memory import CallableMemory, MemoryHit, MemoryPort
from tests.memory.helpers import ALICE, ALICE_ALL, days_ago, rec


class Store:
    """A dict backend with every function, half of them async."""

    def __init__(self):
        self.items = {}

    def add(self, records):
        for record in records:
            self.items[record.id] = record

    async def search(self, query, scope, k, filters):
        found = [r for r in self.items.values() if query in r.text and scope.covers(r.scope)]
        return [(r, 1.0) for r in found][:k]

    def get(self, id, scope):
        return self.items.get(id)

    async def delete(self, ids, scope):
        targets = [i for i in self.items if (ids is None or i in ids)]
        targets = [i for i in targets if scope is None or scope.covers(self.items[i].scope)]
        for i in targets:
            del self.items[i]
        return len(targets)

    def export(self, scope):
        return [r.model_dump() for r in self.items.values() if scope.covers(r.scope)]


def _memory(store=None, **kw):
    store = store or Store()
    fns = dict(
        add=store.add, search=store.search, get=store.get, delete=store.delete, export=store.export
    )
    return CallableMemory(**{**fns, **kw}), store


async def test_round_trip_through_host_functions():
    memory, store = _memory()
    assert isinstance(memory, MemoryPort)
    (stored,) = await memory.add([rec(ALICE, "museum on friday")])
    hits = await memory.search("museum", scope=ALICE)
    assert hits == [MemoryHit(record=stored, score=1.0)]
    assert (await memory.get(stored.id)).text == "museum on friday"
    assert [r.id for r in await memory.export(ALICE_ALL)] == [stored.id]
    assert await memory.delete(scope=ALICE_ALL) == 1
    assert store.items == {}


async def test_search_accepts_records_hits_and_dicts():
    item = rec(ALICE, "x", id="m1", created_at=days_ago(0))
    for found in ([item], [MemoryHit(record=item, score=2)], [item.model_dump()]):
        memory = CallableMemory(add=lambda r: None, search=lambda q, s, k, f, found=found: found)
        assert [h.record.id for h in await memory.search("x", scope=ALICE)] == ["m1"]


async def test_add_may_return_the_stored_records():
    memory = CallableMemory(
        add=lambda records: [r.model_copy(update={"id": "backend-1"}) for r in records],
        search=lambda *a: [],
    )
    assert [r.id for r in await memory.add([rec(ALICE, "x")])] == ["backend-1"]


async def test_missing_functions_raise_tool_usage_error():
    memory = CallableMemory(add=lambda r: None, search=lambda *a: [])
    with pytest.raises(ToolUsageError):
        await memory.get("m1")
    with pytest.raises(ToolUsageError):
        await memory.delete(["m1"])
    with pytest.raises(ToolUsageError):
        await memory.export(ALICE)


async def test_backend_failures_are_integration_errors_without_their_text(caplog):
    def boom(*args):
        raise RuntimeError("password=hunter2")

    memory = CallableMemory(add=boom, search=boom, backend="vault")
    with pytest.raises(IntegrationError) as info:
        await memory.search("x", scope=ALICE)
    assert info.value.integration == "vault"
    assert "hunter2" not in caplog.text and "RuntimeError" in caplog.text
    with pytest.raises(IntegrationError):
        await memory.add([rec(ALICE, "x")])


@pytest.mark.parametrize("bad", [{"not": "a list"}, "text", 5])
async def test_a_wrong_return_shape_is_an_integration_error(bad):
    memory = CallableMemory(add=lambda r: bad, search=lambda *a: bad, delete=lambda i, s: "x")
    with pytest.raises(IntegrationError):
        await memory.search("x", scope=ALICE)
    with pytest.raises(IntegrationError):
        await memory.add([rec(ALICE, "x")])
    with pytest.raises(IntegrationError):
        await memory.delete(["m1"])


def test_bad_construction_is_refused():
    with pytest.raises(ToolUsageError):
        CallableMemory(add=1, search=lambda *a: [])
    with pytest.raises(ToolUsageError):
        CallableMemory(add=lambda r: None, search=lambda *a: [], backend="")


async def test_filters_still_apply_after_the_backend():
    item = rec(ALICE, "x", id="m1", metadata={"kind": "a"})
    memory = CallableMemory(add=lambda r: None, search=lambda *a: [item])
    assert await memory.search("x", scope=ALICE, filters={"kind": "b"}) == []


async def test_an_awaitable_returned_by_a_sync_function_is_awaited():
    async def later():
        return [rec(ALICE, "late", id="m9")]

    memory = CallableMemory(add=lambda r: None, search=lambda *a: later())
    assert [h.record.id for h in await memory.search("x", scope=ALICE)] == ["m9"]


def test_sync_twins_block_and_return():
    memory, _ = _memory()
    (stored,) = memory.add_sync([rec(ALICE, "sync museum")])
    assert [h.record.id for h in memory.search_sync("museum", scope=ALICE)] == [stored.id]
    assert memory.get_sync(stored.id).text == "sync museum"
    assert [r.id for r in memory.export_sync(ALICE)] == [stored.id]
    assert memory.delete_sync([stored.id]) == 1
    assert memory.purge_sync() == 0


async def test_a_future_created_at_is_refused():
    memory, _ = _memory()
    with pytest.raises(ToolUsageError):
        await memory.add([rec(ALICE, "x", created_at=days_ago(-3650))])
    (ok,) = await memory.add([rec(ALICE, "y", created_at=days_ago(-0.001))])  # clock skew
    assert ok.text == "y"
