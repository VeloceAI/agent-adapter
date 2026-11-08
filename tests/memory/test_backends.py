"""Mem0Memory, ZepMemory and GraphitiMemory over fake clients; real-library contract checks."""

import inspect
import sys

import pytest

from agent_adapter.errors import ConfigError, IntegrationError, MissingDependency, ToolUsageError
from agent_adapter.memory import GraphitiMemory, Mem0Memory, MemoryScope, ZepMemory
from agent_adapter.memory._clients import group_id, results, timestamp
from tests.memory.fakes import (
    FakeAsyncMem0,
    FakeGraphiti,
    FakeMem0,
    FakeZep,
    install_graphiti,
    install_module,
)
from tests.memory.helpers import ALICE, ALICE_ALL, ALICE_S2, BOB, BOB_ALL, days_ago, rec

# --- Mem0 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("client", [FakeMem0, FakeAsyncMem0])
async def test_mem0_round_trip_maps_the_scope(client):
    fake = client()
    memory = Mem0Memory(fake)
    (stored,) = await memory.add([rec(ALICE, "likes the museum", metadata={"kind": "pref"})])
    assert stored.id.startswith("m0-") and fake.calls[0] == ("add", False)
    assert fake.items[stored.id]["run_id"] == "s1" and fake.items[stored.id]["user_id"] == "alice"
    hits = await memory.search("museum", scope=ALICE_ALL, filters={"kind": "pref"})
    assert [h.record.id for h in hits] == [stored.id] and hits[0].score == 0.9
    assert hits[0].record.metadata == {"kind": "pref"} and hits[0].record.scope == ALICE
    assert (await memory.get(stored.id, scope=ALICE_ALL)).text == "likes the museum"
    assert [r.id for r in await memory.export(ALICE_ALL)] == [stored.id]


async def test_mem0_never_crosses_users_even_when_its_filter_is_wider():
    fake = FakeMem0()
    memory = Mem0Memory(fake)
    await memory.add([rec(ALICE, "cat"), rec(BOB, "cat")])
    agent_wide = MemoryScope(session_id="s1")
    assert await memory.search("cat", scope=agent_wide) == []  # Mem0 alone would return both
    (bob,) = [i for i, item in fake.items.items() if item["user_id"] == "bob"]
    assert await memory.get(bob, scope=ALICE_ALL) is None
    assert await memory.delete([bob], scope=ALICE_ALL) == 0
    assert await memory.delete(scope=ALICE_ALL) == 1
    assert [r.scope.user_id for r in await memory.export(BOB_ALL)] == ["bob"]


async def test_mem0_keeps_our_times_and_honours_expiry():
    fake = FakeMem0()
    memory = Mem0Memory(fake, retention_days=30)
    (fresh,) = await memory.add([rec(ALICE, "fresh")])
    await memory.add([rec(ALICE, "old", created_at=days_ago(40))])
    await memory.add([rec(ALICE, "gone", expires_at=days_ago(1))])
    assert [r.text for r in await memory.export(ALICE_ALL)] == ["fresh"]
    assert (await memory.get(fresh.id)).expires_at is not None


async def test_mem0_skips_unscoped_or_empty_memories():
    fake = FakeMem0()
    fake.items["x"] = {
        "id": "x",
        "memory": "orphan",
        "user_id": None,
        "agent_id": None,
        "run_id": None,
    }
    fake.items["y"] = {"id": "y", "memory": "", "user_id": "alice"}
    memory = Mem0Memory(fake)
    assert await memory.get("x") is None and await memory.get("y") is None


async def test_mem0_without_the_library_raises_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mem0", None)
    with pytest.raises(MissingDependency):
        await Mem0Memory().search("x", scope=ALICE)


async def test_mem0_builds_its_default_client_lazily(monkeypatch):
    install_module(monkeypatch, "mem0", Memory=FakeMem0)
    memory = Mem0Memory(infer=True)
    await memory.add([rec(ALICE, "x")])
    assert isinstance(memory.client, FakeMem0) and memory.client.calls[0] == ("add", True)


async def test_mem0_add_keeps_only_added_or_updated_memories():
    fake = FakeMem0()
    fake.events = ["ADD", "UPDATE", "DELETE", "NONE"]
    stored = await Mem0Memory(fake, infer=True).add([rec(ALICE, "x")])
    assert [r.id.rsplit("-", 1)[1] for r in stored] == ["ADD", "UPDATE"]


async def test_mem0_erases_a_user_with_delete_all_and_checks_it_is_gone():
    fake = FakeMem0()
    memory = Mem0Memory(fake)
    await memory.add([rec(ALICE, "a"), rec(ALICE_S2, "b"), rec(BOB, "c")])
    assert await memory.delete(scope=ALICE_ALL) == 2
    assert ("delete_all", "alice", None, None) in fake.calls
    assert [i["user_id"] for i in fake.items.values()] == ["bob"]


async def test_mem0_without_a_user_deletes_one_by_one_and_spares_users():
    fake = FakeMem0()
    memory = Mem0Memory(fake)
    anon = MemoryScope(session_id="s1")
    await memory.add([rec(anon, "anon"), rec(ALICE, "alice in s1")])
    assert await memory.delete(scope=anon) == 1
    assert not any(call[0] == "delete_all" for call in fake.calls)
    assert [i["memory"] for i in fake.items.values()] == ["alice in s1"]


async def test_mem0_erasure_that_leaves_records_raises():
    class Stubborn(FakeMem0):
        def delete_all(self, **kw):
            return {"message": "ok"}  # claims success, deletes nothing

    memory = Mem0Memory(Stubborn())
    await memory.add([rec(ALICE, "kept")])
    with pytest.raises(IntegrationError):
        await memory.delete(scope=ALICE_ALL)


async def test_mem0_metadata_filters_cannot_replace_the_scope():
    fake = FakeMem0()
    memory = Mem0Memory(fake)
    await memory.add([rec(BOB, "cat", metadata={"kind": "pet"})])
    hits = await memory.search("cat", scope=ALICE_ALL, filters={"user_id": "bob", "kind": "pet"})
    assert hits == []
    assert fake.calls[-1] == ("search", {"kind": "pet", "user_id": "alice"})


async def test_mem0_empty_query_lists_the_newest():
    memory = Mem0Memory(FakeMem0())
    await memory.add([rec(ALICE, "older")])
    await memory.add([rec(ALICE, "newer")])
    assert [h.record.text for h in await memory.search("", scope=ALICE_ALL, k=1)] == ["newer"]


async def test_mem0_client_errors_become_integration_errors():
    class Broken(FakeMem0):
        def search(self, *a, **kw):
            raise ConnectionError("api key sk-123")

    with pytest.raises(IntegrationError):
        await Mem0Memory(Broken()).search("x", scope=ALICE)


# --- Zep -------------------------------------------------------------------------------------


async def test_zep_round_trip_per_user():
    fake = FakeZep()
    memory = ZepMemory(fake)
    (stored,) = await memory.add([rec(ALICE_ALL, "aisle seat please")])
    await memory.add([rec(BOB_ALL, "aisle seat for bob")])
    assert stored.id.startswith("ep-")
    assert [h.record.text for h in await memory.search("aisle", scope=ALICE_ALL)] == [
        "aisle seat please"
    ]
    assert [h.record.text for h in await memory.search("", scope=ALICE_ALL)] == [
        "aisle seat please"
    ]
    assert (await memory.get(stored.id, scope=ALICE_ALL)).text == "aisle seat please"
    assert await memory.get(stored.id, scope=BOB_ALL) is None
    assert await memory.delete([stored.id], scope=BOB_ALL) == 0
    assert await memory.delete([stored.id], scope=ALICE_ALL) == 1
    await memory.add([rec(ALICE_ALL, "again")])
    assert await memory.delete(scope=ALICE_ALL) == 1
    assert fake.deleted_users == [group_id("alice")]
    assert len(await memory.export(BOB_ALL)) == 1
    assert all("alice" not in key and "bob" not in key for key in fake.users)  # hashed ids


async def test_zep_creates_the_user_before_the_first_episode_and_reads_without_creating():
    fake = FakeZep()
    memory = ZepMemory(fake)
    assert await memory.search("x", scope=ALICE_ALL) == []
    assert await memory.export(ALICE_ALL) == [] and fake.users == {}
    assert await memory.delete(scope=ALICE_ALL) == 0
    await memory.add([rec(ALICE_ALL, "first")])
    assert list(fake.users) == [group_id("alice")]
    again = ZepMemory(fake)  # a new process finds the existing user by lookup
    assert [r.text for r in await again.export(ALICE_ALL)] == ["first"]


async def test_zep_create_race_falls_back_to_lookup():
    fake = FakeZep()
    real_create = fake.user.create

    def racing(*, user_id):
        real_create(user_id=user_id)  # someone else won
        raise RuntimeError("conflict")

    fake.user.create = racing
    memory = ZepMemory(fake)
    await memory.add([rec(ALICE_ALL, "x")])
    assert len(await memory.export(ALICE_ALL)) == 1


async def test_zep_drops_episodes_of_another_graph(caplog):
    fake = FakeZep()
    memory = ZepMemory(fake)
    await memory.add([rec(ALICE_ALL, "alice cat")])
    fake.leak = type("Ep", (), {"uuid_": "ep-x", "content": "bob cat", "graph_uuid": "g-bob"})()
    assert [h.record.text for h in await memory.search("cat", scope=ALICE_ALL)] == ["alice cat"]
    assert [r.text for r in await memory.export(ALICE_ALL)] == ["alice cat"]
    assert "another graph" in caplog.text and "bob cat" not in caplog.text


async def test_zep_lookup_errors_other_than_not_found_fail():
    fake = FakeZep()

    def down(*, user_id):
        raise ConnectionError("down")

    fake.user.lookup = down
    with pytest.raises(IntegrationError):
        await ZepMemory(fake).add([rec(ALICE_ALL, "x")])


@pytest.mark.parametrize("scope", [ALICE, MemoryScope(session_id="s"), MemoryScope(agent_id="a")])
async def test_zep_refuses_scopes_it_cannot_keep_apart(scope):
    memory = ZepMemory(FakeZep())
    with pytest.raises(ToolUsageError):
        await memory.search("x", scope=scope)
    with pytest.raises(ToolUsageError):
        await memory.add([rec(scope, "x")])


async def test_zep_refuses_metadata_filters_expiry_and_scopeless_calls():
    memory = ZepMemory(FakeZep())
    with pytest.raises(ToolUsageError):
        await memory.add([rec(ALICE_ALL, "x", metadata={"a": 1})])
    with pytest.raises(ToolUsageError):
        await memory.add([rec(ALICE_ALL, "x", expires_at=days_ago(-1))])
    with pytest.raises(ToolUsageError):
        await memory.search("x", scope=ALICE_ALL, filters={"a": 1})
    with pytest.raises(ToolUsageError):
        await memory.get("ep-1")
    with pytest.raises(ToolUsageError):
        await memory.delete(["ep-1"])


async def test_zep_without_the_library_raises_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "zep_cloud", None)
    with pytest.raises(MissingDependency):
        await ZepMemory().export(ALICE_ALL)
    install_module(monkeypatch, "zep_cloud.client", Zep=FakeZep)
    assert isinstance(ZepMemory().client, FakeZep)


# --- Graphiti --------------------------------------------------------------------------------


async def test_graphiti_round_trip_in_a_hashed_group(monkeypatch):
    graph = FakeGraphiti()
    cleared = install_graphiti(monkeypatch, graph)
    memory = GraphitiMemory(graph)
    (stored,) = await memory.add([rec(ALICE_ALL, "visits the museum on fridays")])
    await memory.add([rec(BOB_ALL, "visits the museum too")])
    node = graph.nodes[stored.id]
    assert node.group_id == group_id("alice") and "alice" not in node.group_id
    hits = await memory.search("museum", scope=ALICE_ALL)
    assert [h.record.text for h in hits] == ["fact: visits the museum on fridays"]
    assert [h.record.text for h in await memory.search("", scope=ALICE_ALL)] == [
        "visits the museum on fridays"
    ]
    assert (await memory.get(stored.id, scope=ALICE_ALL)).id == stored.id
    assert await memory.delete([stored.id], scope=BOB_ALL) == 0
    assert await memory.delete([stored.id], scope=ALICE_ALL) == 1
    await memory.add([rec(ALICE_ALL, "again")])
    assert await memory.delete(scope=ALICE_ALL) == 1 and cleared == [[group_id("alice")]]
    assert len(await memory.export(BOB_ALL)) == 1


async def test_graphiti_drops_edges_from_another_group(monkeypatch, caplog):
    graph = FakeGraphiti()
    install_graphiti(monkeypatch, graph)
    memory = GraphitiMemory(graph)
    await memory.add([rec(ALICE_ALL, "cat person")])
    graph.leak = type(
        "Edge", (), {"uuid": "e-x", "fact": "bob secret", "group_id": group_id("bob")}
    )()
    assert [h.record.text for h in await memory.search("cat", scope=ALICE_ALL)] == [
        "fact: cat person"
    ]


async def test_graphiti_refusals(monkeypatch):
    graph = FakeGraphiti()
    install_graphiti(monkeypatch, graph)
    with pytest.raises(ConfigError):
        GraphitiMemory(None)
    memory = GraphitiMemory(graph)
    for call in (
        memory.add([rec(ALICE_S2, "x")]),
        memory.add([rec(ALICE_ALL, "x", metadata={"a": 1})]),
        memory.search("x", scope=ALICE_ALL, filters={"a": 1}),
        memory.get("gn-1"),
        memory.delete(["gn-1"]),
    ):
        with pytest.raises(ToolUsageError):
            await call


async def test_graphiti_without_the_library_raises_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "graphiti_core", None)
    monkeypatch.setitem(sys.modules, "graphiti_core.nodes", None)
    with pytest.raises(MissingDependency):
        await GraphitiMemory(FakeGraphiti()).export(ALICE_ALL)


# --- helpers ---------------------------------------------------------------------------------


def test_client_helpers():
    assert results(None) == [] and results({"results": [1]}) == [1] and results([2]) == [2]
    with pytest.raises(IntegrationError):
        results("text")
    assert timestamp("2026-01-01T00:00:00Z").tzinfo is not None
    assert timestamp("2026-01-01T00:00:00").tzinfo is not None
    assert timestamp("not a time") is None and timestamp(5) is None
    assert timestamp(days_ago(0).replace(tzinfo=None)).tzinfo is not None


# --- contract checks against the real libraries (skipped when not installed) -----------------


def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


@pytest.mark.contract
def test_mem0_contract():
    mem0 = pytest.importorskip("mem0")
    for cls in (mem0.Memory, mem0.AsyncMemory):
        assert {"user_id", "agent_id", "run_id", "metadata", "infer"} <= _params(cls.add)
        assert {"top_k", "filters"} <= _params(cls.search)
        assert {"top_k", "filters"} <= _params(cls.get_all)
        assert {"user_id", "agent_id", "run_id"} <= _params(cls.delete_all)
        assert {"memory_id"} <= _params(cls.get) and {"memory_id"} <= _params(cls.delete)


@pytest.mark.contract
def test_zep_contract():
    pytest.importorskip("zep_cloud")
    from zep_cloud.client import AsyncZep, Zep
    from zep_cloud.types import AddEpisodeResult, Episode, User

    for client in (Zep(api_key="test"), AsyncZep(api_key="test")):  # builds no connection
        assert {"user_id"} <= _params(client.user.lookup) and {"user_id"} <= _params(
            client.user.create
        )
        assert {"user_uuid"} <= _params(client.user.delete)
        assert {"graph_uuid", "data", "type", "source_description"} <= _params(
            client.graph.episode.add
        )
        assert {"graph_uuid", "page_size", "order_by", "order"} <= _params(
            client.graph.episode.list
        )
        assert {"graph_uuid", "episode_uuid"} <= _params(client.graph.episode.delete)
        assert {"graph_uuid", "query", "page_size"} <= _params(client.graph.search_episodes)
    assert {"uuid_", "graph_uuid"} <= set(User.model_fields)
    assert {"uuid_", "content", "graph_uuid", "created_at"} <= set(Episode.model_fields)
    assert "episode" in AddEpisodeResult.model_fields


@pytest.mark.contract
def test_graphiti_contract():
    graphiti_core = pytest.importorskip("graphiti_core")
    from graphiti_core.nodes import EpisodeType, EpisodicNode
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data

    params = _params(graphiti_core.Graphiti.add_episode)
    assert {"name", "episode_body", "source", "reference_time", "group_id"} <= params
    assert {"group_ids", "num_results"} <= _params(graphiti_core.Graphiti.search)
    assert {"group_id", "content", "uuid", "created_at"} <= set(EpisodicNode.model_fields)
    assert EpisodeType.text and callable(EpisodicNode.get_by_group_ids) and callable(clear_data)


def test_user_only_marks_the_backends_that_partition_by_user():
    from agent_adapter.memory import CallableMemory, MemoryPort, SQLiteMemory

    graph = FakeGraphiti()
    assert ZepMemory(FakeZep()).user_only is True and GraphitiMemory(graph).user_only is True
    assert Mem0Memory(FakeMem0()).user_only is False
    assert SQLiteMemory("x.db").user_only is False
    assert CallableMemory(add=lambda r: None, search=lambda *a: []).user_only is False
    assert all(isinstance(m, MemoryPort) for m in (ZepMemory(FakeZep()), GraphitiMemory(graph)))

    class NoFlag:
        backend = "x"

        async def add(self, records, *, ctx=None): ...
        async def search(self, query, *, scope, k=5, filters=None, ctx=None): ...
        async def get(self, id, *, scope=None, ctx=None): ...
        async def delete(self, ids=None, *, scope=None, ctx=None): ...
        async def export(self, scope, *, ctx=None): ...

    assert not isinstance(NoFlag(), MemoryPort)  # the flag is part of the port
