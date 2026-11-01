"""MCP tasks extension options: which slow or pending calls become task handles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.mcp._task_store import MemoryTaskStore, RedisTaskStore, TaskStore

__all__ = [
    "APPROVAL_URL_PLACEHOLDER",
    "TASKS_EXTENSION",
    "TASK_ID_META",
    "MemoryTaskStore",
    "RedisTaskStore",
    "TaskOptions",
    "TaskStore",
]

TASKS_EXTENSION = "io.modelcontextprotocol/tasks"
TASK_ID_META = "agent-adapter/task_id"  # an imported tool's error `details` / resume key
APPROVAL_URL_PLACEHOLDER = "{approval_id}"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class TaskOptions:
    """How `mcp_server(tasks=...)` answers 2026-07-28 clients that accept tasks.

    Args:
        threshold_s: A call still running after this many seconds becomes a task; `None`
            makes tasks only of calls that wait for approval.
        ttl_s: How long a task is kept after it is created.
        poll_interval_s: The poll interval suggested to clients, and how often a task
            waiting for approval re-reads the approval store.
        approval_url: An https page where a person decides the approval, with
            `{approval_id}` in it; clients with URL-mode elicitation are sent there.
        max_tasks: Tasks kept at once; past it, calls are answered without a task.
        max_per_principal: Tasks kept at once for one caller.
        store: Shared task state (e.g. `RedisTaskStore`) so any worker answers `tasks/*`;
            `None` keeps tasks in this process.

    Raises:
        ConfigError: a value out of range, or an `approval_url` that is not https (or
            http on a loopback host) or lacks `{approval_id}`; a `store` that is not a
            `TaskStore`.
    """

    threshold_s: float | None = 5.0
    ttl_s: float = 3600.0
    poll_interval_s: float = 2.0
    approval_url: str | None = None
    max_tasks: int = 10_000
    max_per_principal: int = 100
    store: TaskStore | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.threshold_s is not None and not _positive(self.threshold_s, allow_zero=True):
            raise ConfigError(f"threshold_s must be >= 0 or None, got {self.threshold_s!r}")
        if not _positive(self.ttl_s) or not _positive(self.poll_interval_s):
            raise ConfigError("ttl_s and poll_interval_s must be finite and > 0")
        if self.poll_interval_s > self.ttl_s:
            raise ConfigError("poll_interval_s must not exceed ttl_s")
        for name in ("max_tasks", "max_per_principal"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ConfigError(f"{name} must be an int >= 1, got {value!r}")
        if self.approval_url is not None:
            _check_approval_url(self.approval_url)
        if self.store is not None and not isinstance(self.store, TaskStore):
            raise ConfigError(f"store must be a TaskStore, got {type(self.store).__name__}")

    def approval_link(self, approval_id: str) -> str | None:
        """The approval page for `approval_id` (the id URL-quoted), or None without one."""
        if self.approval_url is None:
            return None
        return self.approval_url.replace(APPROVAL_URL_PLACEHOLDER, quote(approval_id, safe=""))


def _positive(value: object, *, allow_zero: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return False
    return value >= 0 if allow_zero else value > 0


def _check_approval_url(url: object) -> None:
    if not isinstance(url, str) or APPROVAL_URL_PLACEHOLDER not in url:
        raise ConfigError(f"approval_url must be a str containing {APPROVAL_URL_PLACEHOLDER}")
    try:
        parts = urlsplit(url.replace(APPROVAL_URL_PLACEHOLDER, "x"))
        host = parts.hostname
    except ValueError as exc:
        raise ConfigError(f"approval_url is not a URL: {exc}") from exc
    secure = parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK)
    if not host or not secure or parts.username or parts.password:
        raise ConfigError("approval_url must be https (http only on a loopback host), no userinfo")
