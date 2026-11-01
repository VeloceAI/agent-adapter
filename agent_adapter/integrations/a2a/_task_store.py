"""The default A2A task store: the SDK's in-memory store with a TTL, a total and a per-owner cap."""

from __future__ import annotations

import builtins
import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from agent_adapter.integrations.a2a._owned import Entry, OwnedTable

logger = logging.getLogger(__name__)

TASK_TTL_S = 24 * 3600.0
MAX_TASKS = 10_000
MAX_TASKS_PER_OWNER = 1_000


def bounded_task_store(
    *,
    ttl: float = TASK_TTL_S,
    max_tasks: int = MAX_TASKS,
    max_per_owner: int = MAX_TASKS_PER_OWNER,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """An SDK `TaskStore` in memory: tasks expire `ttl` s after their last update.

    Over a cap the oldest finished task goes first (an owner's cap only drops its own tasks).
    A rejected task keeps its status but not its message history.

    Raises:
        ValueError: a cap or `ttl` is not positive.
    """
    return _store_class()(ttl=ttl, max_tasks=max_tasks, max_per_owner=max_per_owner, clock=clock)


@functools.cache
def _store_class() -> type[Any]:
    from a2a.auth.user import User
    from a2a.server.context import ServerCallContext
    from a2a.server.owner_resolver import resolve_user_scope
    from a2a.server.tasks import InMemoryTaskStore, TaskStore
    from a2a.types import Task, TaskState

    finished = frozenset(
        {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        }
    )

    class OwnerUser(User):  # type: ignore[misc,unused-ignore]
        """A user naming only the task owner, to delete that owner's task."""

        def __init__(self, owner: str) -> None:
            self.owner = owner

        @property
        def is_authenticated(self) -> bool:
            return bool(self.owner)

        @property
        def user_name(self) -> str:
            return self.owner

    class BoundedTaskStore(TaskStore):  # type: ignore[misc,unused-ignore]
        """`InMemoryTaskStore` plus an index that expires and evicts tasks."""

        def __init__(
            self, *, ttl: float, max_tasks: int, max_per_owner: int, clock: Callable[[], float]
        ) -> None:
            self.inner = InMemoryTaskStore()
            self.index: OwnedTable[bool] = OwnedTable(  # value: the task is finished
                ttl=ttl,
                max_entries=max_tasks,
                max_per_owner=max_per_owner,
                evict_own=True,
                disposable=bool,
                clock=clock,
            )

        def __len__(self) -> int:
            return len(self.index)

        async def save(self, task: Any, context: Any) -> None:
            """Store `task` (a rejected one without its history) and enforce the bounds."""
            if task.status.state == TaskState.TASK_STATE_REJECTED and len(task.history):
                slim = Task()
                slim.CopyFrom(task)
                del slim.history[:]
                task = slim
            owner = resolve_user_scope(context)
            expired, evicted = self.index.put(owner, task.id, task.status.state in finished)
            await self._forget(expired)
            if evicted:
                logger.warning("A2A task store full: dropped %d oldest task(s)", len(evicted))
                await self._forget(evicted)
            await self.inner.save(task, context)

        async def get(self, task_id: str, context: Any) -> Any:
            """The caller's live task, or None."""
            await self._forget(self.index.expired())
            if self.index.get(resolve_user_scope(context), task_id) is None:
                return None
            return await self.inner.get(task_id, context)

        async def list(self, params: Any, context: Any) -> Any:
            """The caller's live tasks (the SDK's filters and paging)."""
            await self._forget(self.index.expired())
            return await self.inner.list(params, context)

        async def delete(self, task_id: str, context: Any) -> None:
            """Delete the caller's task."""
            self.index.pop(resolve_user_scope(context), task_id)
            await self.inner.delete(task_id, context)

        async def _forget(self, entries: builtins.list[Entry[bool]]) -> None:
            for owner, task_id, _ in entries:
                await self.inner.delete(task_id, ServerCallContext(user=OwnerUser(owner)))
            if entries:
                logger.debug("A2A task store: %d task(s) removed", len(entries))

    return BoundedTaskStore
