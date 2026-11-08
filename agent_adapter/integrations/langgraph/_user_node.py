"""`one_use`: your own LangGraph node, run as one task execution that replays finished calls.

LangGraph re-runs an interrupted node from the top. Inside `one_use` each approval answer is
used once (`_spent`), and the executor calls the node finished before a pause are kept in an
in-process replay journal keyed by run id and task namespace: the re-run claims their
answers at their resume positions before the node runs (so any failure spends them), then
replays them (an approved tool runs once). A new process has no journal, so it re-runs the
node's finished calls. A call whose arguments have no journal key cannot replay, so a pause
after it fails closed (`ToolUsageError`) instead of running it twice.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.exec.integration import (
    RAN_UNKEYED,
    Journal,
    JournalData,
    make_journal,
    unsealable_calls,
    visit_scope,
)
from agent_adapter.integrations.langgraph._context import run_ids, task_namespace
from agent_adapter.integrations.langgraph._integration import INTEGRATION
from agent_adapter.integrations.langgraph._memo import TaskMemo
from agent_adapter.integrations.langgraph._spent import Asked, Owner, rerun_follows, task_scope
from agent_adapter.integrations.langgraph.approval import catch_up

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Kept:
    """A paused execution's finished calls and the answers it read, in resume order."""

    journal: JournalData
    asked: tuple[Asked, ...]


# Finished calls of one_use nodes that paused, by (run id, task namespace).
USER_NODE_MEMO: TaskMemo[_Kept] = TaskMemo()


def one_use(node: Callable[..., Any]) -> Callable[..., Any]:
    """Your own LangGraph node, run as one task execution: each approval answer is used once.

    Needed around a node that calls `LangGraphApproval` itself (tool nodes, tool tasks and
    compiled flows do this already). Executor calls the node finished before a pause replay
    when it re-runs in this process. Sync or async nodes, with or without `config`.

    Raises:
        MissingDependency: langgraph or langchain-core is not installed.
    """
    INTEGRATION.require()
    from langchain_core.runnables import RunnableConfig

    wants_config = "config" in inspect.signature(node).parameters

    @functools.wraps(node)
    async def wrapped(state: Any, config: RunnableConfig) -> Any:
        run_id, namespace = run_ids(config)[0], task_namespace(config)
        key = (run_id, namespace) if run_id and namespace else None
        kept = USER_NODE_MEMO.get(key) if key is not None else None
        journal = make_journal(kept.journal if kept is not None else None)
        with task_scope(run_id, namespace) as owner, visit_scope(None, journal):
            owner.replays.extend(kept.asked if kept is not None else ())
            try:
                catch_up(owner)
                result = node(state, config) if wants_config else node(state)
                if inspect.isawaitable(result):
                    result = await result
            except BaseException as exc:
                unkeyed = _unkeyed_pause(journal, exc)
                if unkeyed:
                    if key is not None and kept is not None:
                        USER_NODE_MEMO.drop_if(key, kept)
                    logger.error("one_use node %s paused after unreplayable calls", namespace)
                    raise ToolUsageError(
                        f"one_use node cannot pause after a call it cannot replay: {unkeyed}"
                    ) from exc
                if key is not None:
                    _settle(key, kept, journal, owner, exc)
                raise
            if key is not None and kept is not None:
                USER_NODE_MEMO.drop_if(key, kept)
            return result

    parameters = [
        inspect.Parameter("state", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "config", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunnableConfig
        ),
    ]
    wrapped.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    return wrapped


def _unkeyed_pause(journal: Journal, exc: BaseException) -> str:
    """The calls a pause would run again (their arguments have no journal key), else ""."""
    if isinstance(exc, asyncio.CancelledError) or not is_control_flow(exc):
        return ""
    return "; ".join(c for c in unsealable_calls(journal.data()) if RAN_UNKEYED in c)


def _settle(
    key: tuple[str, str], kept: _Kept | None, journal: Journal, owner: Owner, exc: BaseException
) -> None:
    """Keep the journal for a re-run (a pause), forget it (a failure: a retry starts over)."""
    if is_control_flow(exc):
        data = journal.data()
        finished = _finished_ids(data)
        asked = [
            a if a.finished else Asked(a.request, a.key, a.request.call_id in finished)
            for a in (*owner.asked, *owner.replays)  # replays not reached yet stay in order
        ]
        USER_NODE_MEMO.put(key, _Kept(data, tuple(asked)))
        logger.debug("one_use node %s paused with %d finished calls", key[1], len(finished))
    elif not rerun_follows(exc) and kept is not None:
        USER_NODE_MEMO.drop_if(key, kept)
    # an invalid resume value: the re-run starts from what this execution loaded


def _finished_ids(data: JournalData) -> set[str]:
    """Call ids of the finished calls in journal data (children included)."""
    ids = {str(result["call_id"]) for *_, result in data.get("calls", ()) if "call_id" in result}
    for *_, child in data.get("children", ()):
        ids |= _finished_ids(child)
    return ids
