"""ZepMemory: a Zep Cloud 4.x client (`Zep` or `AsyncZep`) as a `MemoryPort`, one Zep user each."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_adapter.core.errors import (
    IntegrationError,
    MissingDependency,
    ToolUsageError,
    log_failure,
)
from agent_adapter.memory import _clients as c
from agent_adapter.memory._base import GuardedMemory
from agent_adapter.memory.records import FilterValue, MemoryHit, MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)
PAGE = 100


class ZepMemory(GuardedMemory):
    """Zep behind the memory port: records are text episodes in a Zep user of our own (hashed id).

    Raises:
        MissingDependency: no client and `zep-cloud` is not installed (on first use).
    """

    backend = "zep"
    keeps_expiry = False
    user_only = True

    def __init__(
        self,
        client: Any = None,
        *,
        retention_days: float | None = None,
        capture_content: bool = False,
    ) -> None:
        super().__init__(retention_days=retention_days, capture_content=capture_content)
        self._client = client
        self._users: dict[str, tuple[str, str]] = {}  # user id -> (Zep user uuid, graph uuid)

    @property
    def client(self) -> Any:
        """The Zep client, built on first use when none was given (it reads `ZEP_API_KEY`)."""
        if self._client is None:
            try:
                from zep_cloud.client import Zep
            except ImportError as exc:
                raise MissingDependency(
                    'ZepMemory needs zep-cloud: pip install "agent-adapter[zep]"'
                ) from exc
            self._client = Zep()
        return self._client

    async def _add(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        stored = []
        for record in records:
            user = c.user_only(record.scope, self.backend)
            if record.metadata:
                raise ToolUsageError("zep memory keeps no metadata; leave metadata empty")
            found = await self._user(user, create=True)
            if found is None:  # create=True always returns a user or raises
                raise IntegrationError("zep has no user to add to", integration=self.backend)
            graph = found[1]
            added = await c.call(
                self.client.graph.episode.add,
                graph,
                data=record.text,
                type="text",
                source_description="agent_adapter memory",
            )
            episode_id = c.field(c.field(added, "episode"), "uuid_")
            if not isinstance(episode_id, str) or not episode_id:
                raise IntegrationError("zep returned no episode id", integration=self.backend)
            stored.append(record.model_copy(update={"id": episode_id}))
        return stored

    async def _search(
        self, query: str, scope: MemoryScope, k: int, filters: dict[str, FilterValue]
    ) -> tuple[list[MemoryHit], str]:
        user = c.user_only(scope, self.backend)
        if filters:
            raise ToolUsageError("zep memory keeps no metadata, so it cannot filter by it")
        if not query.strip():
            recent = (await self._export(scope))[-k:]
            return [MemoryHit(record=r) for r in reversed(recent)], "recent"
        found = await self._user(user, create=False)
        if found is None:
            return [], "backend"
        pager = await c.call(self.client.graph.search_episodes, found[1], query=query, page_size=k)
        hits = []
        for item in c.results(pager, "items")[:k]:
            record = _record(item, user, found[1])
            if record is not None:
                hits.append(MemoryHit(record=record, score=float(c.field(item, "score") or 0.0)))
        return hits, "backend"

    async def _get(self, id: str, scope: MemoryScope | None) -> MemoryRecord | None:
        if scope is None:
            raise ToolUsageError("zep memory get needs scope=MemoryScope(user_id=...)")
        return next((r for r in await self._export(scope) if r.id == id), None)

    async def _delete(self, ids: list[str] | None, scope: MemoryScope | None) -> int:
        if scope is None:
            raise ToolUsageError("zep memory delete needs scope=MemoryScope(user_id=...)")
        user = c.user_only(scope, self.backend)
        owned = await self._export(scope)
        found = self._users.get(user)
        if found is None:
            return 0
        if ids is None:
            # The Zep user is ours alone (its id is a hash), so deleting it erases only our data.
            await c.call(self.client.user.delete, found[0])
            self._users.pop(user, None)
            return len(owned)
        targets = [record.id for record in owned if record.id in set(ids)]
        for episode_id in targets:
            await c.call(self.client.graph.episode.delete, found[1], episode_id)
        return len(targets)

    async def _export(self, scope: MemoryScope) -> list[MemoryRecord]:
        user = c.user_only(scope, self.backend)
        found = await self._user(user, create=False)
        if found is None:
            return []
        pager = await c.call(
            self.client.graph.episode.list,
            found[1],
            page_size=PAGE,
            order_by="created_at",
            order="asc",
        )
        items = await _every(pager)
        return [record for item in items if (record := _record(item, user, found[1])) is not None]

    async def _user(self, user: str, *, create: bool) -> tuple[str, str] | None:
        """The Zep user uuid and graph uuid for `user`; created first when `create`."""
        if user in self._users:
            return self._users[user]
        zep_id = c.group_id(user)  # Zep never sees the raw user id (often an e-mail)
        try:
            found = await c.call(self.client.user.lookup, user_id=zep_id)
        except Exception as exc:
            if not _not_found(exc):
                raise
            if not create:
                return None
            found = await self._create(zep_id)
        uuid, graph = c.field(found, "uuid_"), c.field(found, "graph_uuid")
        if not (isinstance(uuid, str) and uuid and isinstance(graph, str) and graph):
            raise IntegrationError("zep returned a user without a graph", integration=self.backend)
        self._users[user] = (uuid, graph)
        return uuid, graph

    async def _create(self, zep_id: str) -> Any:
        try:
            return await c.call(self.client.user.create, user_id=zep_id)
        except Exception as exc:
            # Another worker may have created it first; then the lookup finds it.
            log_failure(logger, logging.INFO, exc, "zep user create failed; looking it up again")
            return await c.call(self.client.user.lookup, user_id=zep_id)


def _not_found(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404


async def _every(pager: Any) -> list[Any]:
    """Every item of a Zep pager (sync or async); following pages calls the API."""
    if isinstance(pager, list):
        return pager
    if hasattr(pager, "__aiter__"):
        return [item async for item in pager]
    if hasattr(pager, "__iter__"):
        return await asyncio.to_thread(list, pager)
    return c.results(pager, "items")


def _record(item: Any, user: str, graph: str) -> MemoryRecord | None:
    """A Zep episode of `graph` as a record of `user`, or None (no id or text, another graph)."""
    episode_id, text = c.field(item, "uuid_"), c.field(item, "content")
    if not (isinstance(episode_id, str) and episode_id and isinstance(text, str) and text):
        return None
    owner = c.field(item, "graph_uuid")
    if owner is not None and owner != graph:
        logger.warning("zep returned an episode of another graph; dropped it")
        return None
    return MemoryRecord(
        id=episode_id,
        scope=MemoryScope(user_id=user),
        text=text,
        created_at=c.timestamp(c.field(item, "created_at")),
    )
