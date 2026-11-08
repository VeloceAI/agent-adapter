"""Fake Mem0 2.x, Zep 4.x and Graphiti clients and modules, shaped like their current APIs."""

import itertools
import sys
import types
from datetime import UTC, datetime

_ids = itertools.count(1)
ENTITY = ("user_id", "agent_id", "run_id")


class FakeMem0:
    """`mem0.Memory` 2.x: entity ids go in `filters` for search and get_all."""

    def __init__(self):
        self.items = {}
        self.calls = []
        self.events = None  # set to make add report these events, as infer=True does

    def add(self, messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True):
        self.calls.append(("add", infer))
        memory_id = f"m0-{next(_ids)}"
        self.items[memory_id] = {
            "id": memory_id,
            "memory": messages[0]["content"],
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata,
            "created_at": datetime.now(UTC).isoformat(),
        }
        if self.events:
            return {
                "results": [
                    {"id": f"{memory_id}-{e}", "memory": "x", "event": e} for e in self.events
                ]
            }
        return {"results": [{"id": memory_id, "memory": messages[0]["content"], "event": "ADD"}]}

    def _matching(self, filters):
        if not any(filters.get(key) for key in ENTITY):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id")
        found = []
        for item in self.items.values():
            meta = item["metadata"] or {}
            ok = all(
                (item[key] if key in ENTITY else meta.get(key)) == value
                for key, value in filters.items()
            )
            if ok:
                found.append(item)
        return found

    def search(self, query, *, top_k=20, filters=None, **kwargs):
        if set(kwargs) & set(ENTITY):
            raise ValueError("Top-level entity parameters are not supported in search()")
        self.calls.append(("search", filters))
        found = [i for i in self._matching(filters or {}) if query.lower() in i["memory"].lower()]
        return {"results": [{**i, "score": 0.9} for i in found[:top_k]]}

    def get(self, memory_id):
        return self.items.get(memory_id)

    def get_all(self, *, filters=None, top_k=20, **kwargs):
        if set(kwargs) & set(ENTITY):
            raise ValueError("Top-level entity parameters are not supported in get_all()")
        return {"results": self._matching(filters or {})[:top_k]}

    def delete(self, memory_id):
        if memory_id not in self.items:
            raise ValueError(f"Memory with id {memory_id} not found")
        del self.items[memory_id]
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        self.calls.append(("delete_all", user_id, agent_id, run_id))
        filters = {k: v for k, v in zip(ENTITY, (user_id, agent_id, run_id), strict=True) if v}
        for item in self._matching(filters):
            del self.items[item["id"]]
        return {"message": "Memories deleted successfully!"}


class FakeAsyncMem0(FakeMem0):
    """`mem0.AsyncMemory`: the same, with async methods."""

    async def add(self, messages, **kw):
        return FakeMem0.add(self, messages, **kw)

    async def search(self, query, **kw):
        return FakeMem0.search(self, query, **kw)

    async def get(self, memory_id):
        return FakeMem0.get(self, memory_id)

    async def get_all(self, **kw):
        return FakeMem0.get_all(self, **kw)

    async def delete(self, memory_id):
        return FakeMem0.delete(self, memory_id)

    async def delete_all(self, **kw):
        return FakeMem0.delete_all(self, **kw)


class NotFoundError(Exception):
    """Like `zep_cloud.errors.NotFoundError`: an API error with status 404."""

    status_code = 404


class FakePager:
    """Like zep_cloud's `SyncPager`: `.items` is the first page; iterating walks every page."""

    def __init__(self, items, page_size):
        self.all = list(items)
        self.items = self.all[:page_size]
        self.pages_read = 0

    def __iter__(self):
        self.pages_read += 1
        yield from self.all


class FakeZep:
    """`zep_cloud.client.Zep` 4.x: users own graphs; episodes live in a graph."""

    def __init__(self):
        self.users = {}  # zep user_id -> user object
        self.episodes = {}  # uuid -> episode
        self.deleted_users = []
        self.leak = None  # an episode of another graph that list/search wrongly return
        episode = types.SimpleNamespace(add=self._add, list=self._list, delete=self._delete)
        self.graph = types.SimpleNamespace(episode=episode, search_episodes=self._search)
        self.user = types.SimpleNamespace(
            lookup=self._lookup, create=self._create, delete=self._delete_user
        )

    def _lookup(self, *, user_id):
        if user_id not in self.users:
            raise NotFoundError("not found")
        return self.users[user_id]

    def _create(self, *, user_id):
        user = types.SimpleNamespace(
            uuid_=f"u-{next(_ids)}", graph_uuid=f"g-{next(_ids)}", user_id=user_id
        )
        self.users[user_id] = user
        return user

    def _delete_user(self, user_uuid):
        for key, user in list(self.users.items()):
            if user.uuid_ == user_uuid:
                del self.users[key]
                self.deleted_users.append(key)
                for eid in [k for k, e in self.episodes.items() if e.graph_uuid == user.graph_uuid]:
                    del self.episodes[eid]

    def _add(self, graph_uuid, *, data, type, source_description=None):
        assert any(u.graph_uuid == graph_uuid for u in self.users.values()), "user first"
        episode = types.SimpleNamespace(
            uuid_=f"ep-{next(_ids)}",
            content=data,
            graph_uuid=graph_uuid,
            created_at=datetime.now(UTC).isoformat(),
        )
        self.episodes[episode.uuid_] = episode
        return types.SimpleNamespace(episode=episode, task=None)

    def _in(self, graph_uuid):
        found = [e for e in self.episodes.values() if e.graph_uuid == graph_uuid]
        return found + ([self.leak] if self.leak else [])

    def _list(self, graph_uuid, *, page_size=None, order_by=None, order=None):
        return FakePager(self._in(graph_uuid), page_size or 10)

    def _search(self, graph_uuid, *, query, page_size=None):
        return FakePager([e for e in self._in(graph_uuid) if query in e.content], page_size or 10)

    def _delete(self, graph_uuid, episode_uuid):
        del self.episodes[episode_uuid]


class FakeGraphiti:
    """A `graphiti_core.Graphiti`: episodes per group; search returns fact edges."""

    def __init__(self):
        self.driver = object()
        self.nodes = {}  # uuid -> node
        self.leak = None  # an edge from another group that search wrongly returns

    async def add_episode(
        self, *, name, episode_body, source, source_description, reference_time, group_id
    ):
        node = types.SimpleNamespace(
            uuid=f"gn-{next(_ids)}",
            content=episode_body,
            group_id=group_id,
            created_at=reference_time,
        )
        self.nodes[node.uuid] = node
        return types.SimpleNamespace(episode=node)

    async def search(self, query, *, group_ids, num_results):
        edges = [
            types.SimpleNamespace(
                uuid=f"edge-{n.uuid}",
                fact=f"fact: {n.content}",
                group_id=n.group_id,
                created_at=n.created_at,
            )
            for n in self.nodes.values()
            if n.group_id in group_ids and query in n.content
        ]
        return ([self.leak] if self.leak else []) + edges[:num_results]

    async def remove_episode(self, uuid):
        del self.nodes[uuid]


def install_graphiti(monkeypatch, graphiti: FakeGraphiti) -> list:
    """Put fake `graphiti_core` modules in `sys.modules`; returns the clear_data calls."""
    cleared = []

    class EpisodicNode:
        @staticmethod
        async def get_by_group_ids(driver, group_ids):
            return [n for n in graphiti.nodes.values() if n.group_id in group_ids]

    async def clear_data(driver, group_ids=None):
        cleared.append(group_ids)
        for key in [k for k, n in graphiti.nodes.items() if n.group_id in group_ids]:
            del graphiti.nodes[key]

    nodes = types.ModuleType("graphiti_core.nodes")
    nodes.EpisodeType = types.SimpleNamespace(text="text")
    nodes.EpisodicNode = EpisodicNode
    ops = types.ModuleType("graphiti_core.utils.maintenance.graph_data_operations")
    ops.clear_data = clear_data
    for name, module in {
        "graphiti_core": types.ModuleType("graphiti_core"),
        "graphiti_core.nodes": nodes,
        "graphiti_core.utils": types.ModuleType("graphiti_core.utils"),
        "graphiti_core.utils.maintenance": types.ModuleType("graphiti_core.utils.maintenance"),
        "graphiti_core.utils.maintenance.graph_data_operations": ops,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return cleared


def install_module(monkeypatch, name: str, **attrs) -> None:
    """A fake top-level module (and parents) holding `attrs`."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        monkeypatch.setitem(sys.modules, ".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    for key, value in attrs.items():
        setattr(sys.modules[name], key, value)
