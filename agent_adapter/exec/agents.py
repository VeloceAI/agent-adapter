"""Agents as tools: any agent becomes a registry tool that runs inside the calling run."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from agent_adapter.approval.builtin import ConfirmRule, evaluate_confirm
from agent_adapter.core.callables import is_async
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalPending,
    ConfigError,
    RunLimitExceeded,
    ToolError,
    ToolUsageError,
)
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ToolCall, ToolResult
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.tool_loop import ToolLoopResult, run_tool_loop

logger = logging.getLogger(__name__)

TaskAgent = Callable[[str, ToolContext], str | Awaitable[str]]
AGENT_KIND = "agent"
DEFAULT_MAX_DEPTH = 3
MAX_PARKED = 256  # paused sub-agent histories kept for a resume, oldest dropped first
PARK_TTL_S = 3600.0  # a paused history older than this is dropped (it may hold PII)

# How many agent tools are running around the current task; child tasks and threads inherit it.
_DEPTH: ContextVar[int] = ContextVar("aa_agent_depth", default=0)


class AgentDepthExceeded(RunLimitExceeded):
    """An agent tool was called deeper than its `max_depth` (agents calling agents)."""

    kind = "agent-depth-exceeded"

    def __init__(self, tool: str, max_depth: int, *, run_id: str = "") -> None:
        super().__init__(tool, "agent_depth", run_id=run_id)
        self.max_depth = max_depth


@dataclass(frozen=True)
class LoopAgent:
    """A `run_tool_loop` agent: its model, the tools it may use and its instructions."""

    model: LLMPort
    registry: Registry
    instructions: str = ""
    max_steps: int = 8
    request_options: Mapping[str, Any] | None = None

    def first_messages(self, task: str) -> list[Message]:
        """The history a fresh run starts from: the instructions, then the task."""
        system = [Message(role="system", content=self.instructions)] if self.instructions else []
        return [*system, Message(role="user", content=task)]


def agent_depth() -> int:
    """How many agent tools are running around the current call (0 outside any)."""
    return _DEPTH.get()


def agent_tool(
    agent: LoopAgent | TaskAgent,
    *,
    name: str,
    description: str,
    executor: Executor | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    timeout: float | None = None,
    confirm: ConfirmRule = False,
    tags: Iterable[str] = (),
) -> ToolSpec:
    """A tool (kind "agent") that runs `agent` on the model's `task` inside the calling run."""
    if isinstance(agent, LoopAgent):
        if executor is None:
            raise ConfigError(
                f"agent tool {name!r}: a LoopAgent needs executor= (the caller's executor), "
                "so its tool calls get the same hooks, budgets and approval port"
            )
        if agent.max_steps < 1:
            raise ConfigError(f"agent tool {name!r}: max_steps must be >= 1")
    elif not callable(agent):
        raise ConfigError(f"agent tool {name!r}: agent must be a LoopAgent or a callable")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ConfigError(f"agent tool {name!r}: max_depth must be an int >= 1")
    runner = _AgentRunner(agent, name, executor, max_depth)

    async def run_agent(task: str, ctx: ToolContext) -> str:
        """Hand a task to the agent.

        Args:
            task: What the agent should do, with every detail it needs.
        """
        return await runner.run(task, ctx)

    async def confirm_unless_resumed(call: ToolCall, ctx: ToolContext) -> bool:
        # A paused call already passed approval for this run, call id and task: resuming it
        # must not ask the reviewer a second time.
        if runner.resumes(ctx.run_id, call.id, call.arguments.get("task")):
            return False
        return await evaluate_confirm(confirm, call, ctx)

    return build_spec(
        run_agent,
        name=name,
        description=description,
        kind=AGENT_KIND,
        tags=frozenset(tags),
        timeout=timeout,
        confirm=confirm if confirm is False else confirm_unless_resumed,
    )


@dataclass(frozen=True)
class _Parked:
    task: str
    messages: list[Message]
    at: float


class _AgentRunner:
    """Runs one agent tool's agent per call, with the depth guard and resumable pauses."""

    def __init__(
        self, agent: LoopAgent | TaskAgent, name: str, executor: Executor | None, max_depth: int
    ) -> None:
        self.agent = agent
        self.name = name
        self.executor = executor
        self.max_depth = max_depth
        self._parked: OrderedDict[tuple[str, str], _Parked] = OrderedDict()

    def resumes(self, run_id: str, call_id: str, task: object) -> bool:
        """True when this run's call `call_id` paused on `task` and will resume, not restart."""
        self._expire()
        parked = self._parked.get((run_id, call_id))
        return parked is not None and parked.task == task

    async def run(self, task: str, ctx: ToolContext) -> str:
        depth = _DEPTH.get()
        if depth >= self.max_depth:
            logger.warning(
                "agent %s refused: depth limit %d reached (run %s)",
                self.name,
                self.max_depth,
                ctx.run_id,
            )
            ctx.emit("agent.refused", agent=self.name, depth=depth + 1)
            raise AgentDepthExceeded(self.name, self.max_depth, run_id=ctx.run_id)
        token = _DEPTH.set(depth + 1)
        ctx.emit("agent.started", agent=self.name, depth=depth + 1)
        status = "error"
        try:
            if isinstance(self.agent, LoopAgent):
                answer = await self._loop(self.agent, task, ctx)
            else:
                answer = await _call_task(self.agent, task, ctx)
            if not isinstance(answer, str):
                raise ToolUsageError(
                    f"agent {self.name!r} must answer with a str, got {type(answer).__name__}"
                )
            status = "ok"
        except ApprovalPending:
            status = "pending"
            raise
        finally:
            _DEPTH.reset(token)
            ctx.emit("agent.finished", agent=self.name, depth=depth + 1, status=status)
        logger.info("agent %s answered (run %s, call %s)", self.name, ctx.run_id, ctx.call_id)
        return answer

    async def _loop(self, agent: LoopAgent, task: str, ctx: ToolContext) -> str:
        key = (ctx.run_id, ctx.call_id)
        self._expire()
        parked = self._parked.pop(key, None)
        resumed = parked is not None and parked.task == task
        messages = parked.messages if parked is not None and resumed else agent.first_messages(task)
        result = await run_tool_loop(
            agent.model,
            agent.registry,
            messages,
            max_steps=agent.max_steps,
            executor=self.executor,
            ctx=ctx,
            stage=f"agent.{self.name}",
            request_options=agent.request_options,
        )
        if result.finish == "pending":
            self._park(key, task, result)
            raise _pending(result.tool_results)
        if result.finish != "stop":
            logger.info("agent %s stopped: %s (run %s)", self.name, result.finish, ctx.run_id)
            raise ToolError(f"agent {self.name!r} stopped before answering ({result.finish})")
        return result.final.text

    def _park(self, key: tuple[str, str], task: str, result: ToolLoopResult) -> None:
        """Keep a paused history, so the same call resumes the agent instead of restarting it."""
        self._parked[key] = _Parked(task, list(result.messages), time.monotonic())
        self._parked.move_to_end(key)
        while len(self._parked) > MAX_PARKED:
            dropped, _ = self._parked.popitem(last=False)
            logger.warning("agent %s dropped the paused history of run %s", self.name, dropped[0])

    def _expire(self) -> None:
        """Drop paused histories older than PARK_TTL_S (oldest first)."""
        cutoff = time.monotonic() - PARK_TTL_S
        while self._parked:
            key, parked = next(iter(self._parked.items()))
            if parked.at > cutoff:
                return
            del self._parked[key]
            logger.info("agent %s expired the paused history of run %s", self.name, key[0])


async def _call_task(agent: TaskAgent, task: str, ctx: ToolContext) -> Any:
    if is_async(agent):
        return await agent(task, ctx)  # type: ignore[misc]
    answer = await asyncio.to_thread(agent, task, ctx)
    if inspect.isawaitable(answer):
        answer = await answer
    return answer


def _pending(results: Iterable[ToolResult]) -> ApprovalPending:
    """The pending approval a sub-agent's tool call is waiting for, to raise in the parent."""
    for result in results:
        if result.status != "pending":
            continue
        if isinstance(result.exception, ApprovalPending):
            return result.exception
        details = result.error.details if result.error is not None else {}
        return ApprovalPending(
            str(details.get("approval_id", "")), tool=result.name, call_id=result.call_id
        )
    return ApprovalPending("", tool="")  # pragma: no cover - a pending loop has a pending result
