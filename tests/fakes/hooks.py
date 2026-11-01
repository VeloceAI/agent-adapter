"""Hook fakes: one that records every call, one that raises."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agent_adapter.hooks.base import Hook

PHASES = (
    "before_tool",
    "after_tool",
    "on_tool_error",
    "before_llm",
    "after_llm",
    "on_llm_error",
    "before_node",
    "after_node",
    "on_node_error",
)


class RecordingHook(Hook):
    """Records `(phase, args)` for every call; returns `returns[phase]` when given."""

    name = "recording"

    def __init__(self, returns: Mapping[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.returns = dict(returns or {})

    def phases(self) -> list[str]:
        """Phase names in call order."""
        return [phase for phase, _ in self.calls]

    async def _record(self, phase: str, *args: Any) -> Any:
        self.calls.append((phase, args))
        return self.returns.get(phase)

    async def before_tool(self, inv: Any) -> Any:
        return await self._record("before_tool", inv)

    async def after_tool(self, inv: Any, result: Any) -> Any:
        return await self._record("after_tool", inv, result)

    async def on_tool_error(self, inv: Any, error: BaseException) -> Any:
        return await self._record("on_tool_error", inv, error)

    async def before_llm(self, inv: Any) -> Any:
        return await self._record("before_llm", inv)

    async def after_llm(self, inv: Any, response: Any) -> Any:
        return await self._record("after_llm", inv, response)

    async def on_llm_error(self, inv: Any, error: BaseException) -> Any:
        return await self._record("on_llm_error", inv, error)

    async def before_node(self, inv: Any) -> Any:
        return await self._record("before_node", inv)

    async def after_node(self, inv: Any, update: Any) -> Any:
        return await self._record("after_node", inv, update)

    async def on_node_error(self, inv: Any, error: BaseException) -> Any:
        return await self._record("on_node_error", inv, error)


class RaisingHook(RecordingHook):
    """Raises `error` from each phase in `phases` (all phases by default)."""

    name = "raising"

    def __init__(self, error: BaseException, phases: Iterable[str] = PHASES) -> None:
        super().__init__()
        self.error = error
        self.raise_in = frozenset(phases)

    async def _record(self, phase: str, *args: Any) -> Any:
        await super()._record(phase, *args)
        if phase in self.raise_in:
            raise self.error
        return None
