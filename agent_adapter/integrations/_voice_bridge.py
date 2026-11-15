"""Shared by the LiveKit and Pipecat voice bridges: background speech that is not lost, one
run per voice session, and a repeat window on by default."""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Coroutine, Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import log_failure
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec import Executor
from agent_adapter.integrations._default_runner import DefaultRunner, lazy_runner
from agent_adapter.integrations.base import ToolRunner
from agent_adapter.runtime.ids import new_id
from agent_adapter.voice.agent import DEDUPE_WINDOW_S

logger = logging.getLogger(__name__)
_BACKGROUND: set[asyncio.Task[Any]] = set()


def spawn(work: Coroutine[Any, Any, Any]) -> None:
    """Run `work` in the background; keep a reference so it is not collected mid-flight."""
    task = asyncio.ensure_future(work)
    _BACKGROUND.add(task)
    task.add_done_callback(_finished)


_RUNS: dict[int, str] = {}  # by identity: session objects need not be hashable
_WINDOWED: weakref.WeakKeyDictionary[Executor, Executor] = weakref.WeakKeyDictionary()


def session_run_id(session: Any) -> str:
    """One run id per voice session object, so repeats and run limits span the whole call."""
    if session is None:
        return ""
    key = id(session)
    run_id = _RUNS.get(key)
    if run_id is None:
        try:
            weakref.finalize(session, _RUNS.pop, key, None)
        except TypeError:  # not weak-referenceable: no shared run
            return ""
        run_id = _RUNS[key] = new_id("run_")
    return run_id


class VoiceRunner:
    """Runs calls on the runner's executor, adding a repeat window when it has none: a voice
    model that asks again for a booking it never heard back about gets the first result."""

    def __init__(self, runner: ToolRunner | None) -> None:
        self._runner = lazy_runner(runner)

    def executor(self) -> ToolRunner:
        base = self._runner
        if isinstance(base, DefaultRunner):
            base = base.executor()
        if not isinstance(base, Executor) or base.dedupe_window_s is not None:
            return base
        windowed = _WINDOWED.get(base)
        if windowed is None:
            windowed = _WINDOWED[base] = base.replace(dedupe_window_s=DEDUPE_WINDOW_S)
        return windowed

    async def run_spec(
        self,
        spec: ToolSpec,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any:
        return await self.executor().run_spec(spec, args, ctx, call_id=call_id)


def _finished(task: asyncio.Task[Any]) -> None:
    _BACKGROUND.discard(task)
    error = None if task.cancelled() else task.exception()
    if error is not None:
        log_failure(logger, logging.WARNING, error, "voice bridge task failed")
