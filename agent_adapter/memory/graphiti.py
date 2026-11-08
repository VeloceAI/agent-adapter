"""GraphitiMemory: a `graphiti_core.Graphiti` graph as a `MemoryPort`, one group per user."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import ConfigError, MissingDependency, ToolUsageError
from agent_adapter.memory import _clients as c
from agent_adapter.memory._base import GuardedMemory
from agent_adapter.memory.records import FilterValue, MemoryHit, MemoryRecord, MemoryScope


class GraphitiMemory(GuardedMemory):
    """Graphiti behind the memory port: episodes in a hashed group per user; search returns facts.

    Raises:
        ConfigError: `graphiti` is None (pass your connected `Graphiti` instance).
    """

    backend = "graphiti"
    keeps_expiry = False
    user_only = True

    def __init__(
        self,
        graphiti: Any,
        *,
        retention_days: float | None = None,
        capture_content: bool = False,
    ) -> None:
        super().__init__(retention_days=retention_days, capture_content=capture_content)
        if graphiti is None:
            raise ConfigError("GraphitiMemory needs your Graphiti instance")
        self.graphiti = graphiti

    async def _add(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        episode_type = _modules()[0]
        stored = []
        for record in records:
            group = c.group_id(c.user_only(record.scope, self.backend))
            if record.metadata:
                raise ToolUsageError("graphiti memory keeps no metadata; leave metadata empty")
            added = await c.call(
                self.graphiti.add_episode,
                name=record.id,
                episode_body=record.text,
                source=episode_type.text,
                source_description="agent_adapter memory",
                reference_time=record.created_at,
                group_id=group,
            )
            episode_id = c.field(c.field(added, "episode"), "uuid")
            stored.append(record.model_copy(update={"id": str(episode_id or record.id)}))
        return stored

    async def _search(
        self, query: str, scope: MemoryScope, k: int, filters: dict[str, FilterValue]
    ) -> tuple[list[MemoryHit], str]:
        group = c.group_id(c.user_only(scope, self.backend))
        if filters:
            raise ToolUsageError("graphiti memory keeps no metadata, so it cannot filter by it")
        if not query.strip():
            return [
                MemoryHit(record=r) for r in reversed((await self._export(scope))[-k:])
            ], "recent"
        edges = await c.call(self.graphiti.search, query, group_ids=[group], num_results=k)
        hits = []
        for rank, edge in enumerate(c.results(edges)):
            if c.field(edge, "group_id") != group:
                continue  # never trust the backend's group filter alone
            record = _record(edge, scope, "fact")
            if record is not None:
                hits.append(MemoryHit(record=record, score=1.0 / (rank + 1)))
        return hits, "backend"

    async def _get(self, id: str, scope: MemoryScope | None) -> MemoryRecord | None:
        if scope is None:
            raise ToolUsageError("graphiti memory get needs scope=MemoryScope(user_id=...)")
        return next((r for r in await self._export(scope) if r.id == id), None)

    async def _delete(self, ids: list[str] | None, scope: MemoryScope | None) -> int:
        if scope is None:
            raise ToolUsageError("graphiti memory delete needs scope=MemoryScope(user_id=...)")
        owned = await self._export(scope)
        if ids is None:
            clear_data = _modules()[2]
            group = c.group_id(c.user_only(scope, self.backend))
            await c.call(clear_data, self.graphiti.driver, group_ids=[group])
            return len(owned)
        targets = [record.id for record in owned if record.id in set(ids)]
        for episode_id in targets:
            await c.call(self.graphiti.remove_episode, episode_id)
        return len(targets)

    async def _export(self, scope: MemoryScope) -> list[MemoryRecord]:
        episodic_node = _modules()[1]
        group = c.group_id(c.user_only(scope, self.backend))
        nodes = await c.call(episodic_node.get_by_group_ids, self.graphiti.driver, [group])
        records = [
            record
            for node in c.results(nodes)
            if c.field(node, "group_id") == group
            and (record := _record(node, scope, "content")) is not None
        ]
        return sorted(records, key=lambda r: r.created_at.timestamp() if r.created_at else 0.0)


def _modules() -> tuple[Any, Any, Any]:
    """`EpisodeType`, `EpisodicNode` and `clear_data` from graphiti_core.

    Raises:
        MissingDependency: graphiti-core is not installed.
    """
    try:
        from graphiti_core.nodes import EpisodeType, EpisodicNode
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data
    except ImportError as exc:
        raise MissingDependency(
            'GraphitiMemory needs graphiti-core: pip install "agent-adapter[graphiti]"'
        ) from exc
    return EpisodeType, EpisodicNode, clear_data


def _record(item: Any, scope: MemoryScope, text_field: str) -> MemoryRecord | None:
    """A Graphiti node or edge as a record in `scope`, or None without an id or text."""
    item_id, text = c.field(item, "uuid"), c.field(item, text_field)
    if not (isinstance(item_id, str) and item_id and isinstance(text, str) and text):
        return None
    return MemoryRecord(
        id=item_id, scope=scope, text=text, created_at=c.timestamp(c.field(item, "created_at"))
    )
