"""CallableGuard: any guard service (your HTTP client, a moderation endpoint) as a guard hook."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.builtin.guardrail import DEFAULT_REASON, Stage
from agent_adapter.hooks.builtin.guards.base import (
    DEFAULT_STAGES,
    Action,
    ContentGuard,
    GuardVerdict,
)

GuardResult = bool | str | GuardVerdict
GuardFn = Callable[[str, Any], GuardResult | Awaitable[GuardResult]]


class CallableGuard(ContentGuard):
    """Wraps `fn(stage, payload)`: True passes; False, a reason or a GuardVerdict says no."""

    name = "callable_guard"

    def __init__(
        self,
        fn: GuardFn,
        *,
        on: Iterable[Stage] | Stage = DEFAULT_STAGES,
        action: Action = "block",
    ) -> None:
        if not callable(fn):
            raise ConfigError("CallableGuard fn must be callable")
        super().__init__(on=on, action=action)
        self.fn = fn

    async def inspect(self, stage: str, payload: Any) -> GuardVerdict:
        if is_async(self.fn):
            answer = await self.fn(stage, payload)  # type: ignore[misc]
        else:
            answer = await asyncio.to_thread(self.fn, stage, payload)
            if inspect.isawaitable(answer):
                answer = await answer
        return as_verdict(answer)


def as_verdict(answer: Any) -> GuardVerdict:
    """A guard function's answer as a GuardVerdict."""
    if isinstance(answer, GuardVerdict):
        return answer
    if answer is True:
        return GuardVerdict(True)
    if answer is False:
        return GuardVerdict(False, DEFAULT_REASON)
    if isinstance(answer, str):
        return GuardVerdict(False, answer or DEFAULT_REASON)
    raise TypeError(f"a guard must return a bool, a reason or a GuardVerdict, not {type(answer)}")
