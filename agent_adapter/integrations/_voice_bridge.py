"""Shared by the LiveKit and Pipecat voice bridges: background speech that is not lost."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from agent_adapter.core.errors import log_failure

logger = logging.getLogger(__name__)
_BACKGROUND: set[asyncio.Task[Any]] = set()


def spawn(work: Coroutine[Any, Any, Any]) -> None:
    """Run `work` in the background; keep a reference so it is not collected mid-flight."""
    task = asyncio.ensure_future(work)
    _BACKGROUND.add(task)
    task.add_done_callback(_finished)


def _finished(task: asyncio.Task[Any]) -> None:
    _BACKGROUND.discard(task)
    error = None if task.cancelled() else task.exception()
    if error is not None:
        log_failure(logger, logging.WARNING, error, "voice bridge task failed")
