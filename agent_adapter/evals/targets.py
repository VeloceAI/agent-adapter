"""What an eval runs: any agent callable, a chat agent, a `Flow`, or a tool loop."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_adapter.chat import ChatMessage, ChatSession
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.messages import Message, OutputSchema
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.evals._calls import call, joined
from agent_adapter.evals._text import user_text
from agent_adapter.evals.model import EvalCase
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.tool_loop import run_tool_loop
from agent_adapter.flow import Flow, FlowRunner


@dataclass(frozen=True)
class Outcome:
    """A target's output, and why it did not finish ("" when it did); unfinished is an error."""

    output: Any
    unfinished: str = ""


@runtime_checkable
class Target(Protocol):
    """Runs one case in the given run (`ctx`) on `executor`; returns the output or an Outcome."""

    name: str

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: ToolContext) -> Any: ...


def is_observed(target: object) -> bool:
    """True when all of `target`'s calls surely run in the case's run (`observed`, default True).

    A target with `observed = False` may call tools or models elsewhere: its cost and tool calls
    count as measured only when the run saw its events.
    """
    return bool(getattr(target, "observed", True))


class CallableTarget:
    """`fn(input)` or `fn(input, ctx=ctx)`, sync or async; take `ctx` so tool calls join the run."""

    def __init__(self, fn: Callable[..., Any], *, name: str = "") -> None:
        """Raises: ConfigError: `fn` is not callable."""
        if not callable(fn):
            raise ConfigError(f"CallableTarget needs a callable, got {type(fn).__name__}")
        self.fn = fn
        self.name = name or _name_of(fn)
        self._takes_ctx = "ctx" in _parameters(fn)
        self.observed = False  # it may use its own executor and ledger

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: ToolContext) -> Any:
        """The function's output for `case.input` (streamed text is joined)."""
        kwargs = {"ctx": ctx} if self._takes_ctx else {}
        return await joined(await call(self.fn, case.input, **kwargs), "an eval target")


class ChatTarget:
    """A chat-page agent `fn(message, session)`; each case is a new one-turn conversation."""

    def __init__(self, agent: Callable[..., Any], *, name: str = "") -> None:
        """Raises: ConfigError: `agent` is not callable."""
        if not callable(agent):
            raise ConfigError(f"ChatTarget needs a callable, got {type(agent).__name__}")
        self.agent = agent
        self.name = name or _name_of(agent)
        self.observed = False  # it may ignore `session.context()`

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: ToolContext) -> Any:
        """The agent's reply text.

        Raises:
            ToolUsageError: the reply is neither text nor an async iterator of text.
        """
        session = ChatSession(
            id=f"eval-{case.id}", history=(), state={}, executor=executor, run_id=ctx.run_id
        )
        reply = await joined(
            await call(self.agent, ChatMessage(text=user_text(case)), session), "a chat agent"
        )
        if not isinstance(reply, str):
            raise ToolUsageError(
                f"a chat agent must return str or an async iterator of str, "
                f"got {type(reply).__name__}"
            )
        return reply


class ToolLoopTarget:
    """`run_tool_loop` over `registry` with an optional system prompt; output is the answer."""

    def __init__(
        self,
        model: LLMPort,
        registry: Registry,
        *,
        system: str = "",
        max_steps: int = 8,
        output_schema: OutputSchema | None = None,
        name: str = "tool_loop",
    ) -> None:
        """Raises: ConfigError: `registry` is not a Registry."""
        if not isinstance(registry, Registry):
            raise ConfigError(f"ToolLoopTarget needs a Registry, got {type(registry).__name__}")
        self.model = model
        self.registry = registry
        self.system = system
        self.max_steps = max_steps
        self.output_schema = output_schema
        self.name = name

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: ToolContext) -> Any:
        """The final text (or parsed output with `output_schema`).

        Raises:
            ApprovalPending: a tool call is waiting for an approval.
        """
        messages = [Message(role="user", content=user_text(case))]
        if self.system:
            messages.insert(0, Message(role="system", content=self.system))
        result = await run_tool_loop(
            self.model,
            self.registry,
            messages,
            max_steps=self.max_steps,
            executor=executor,
            ctx=ctx,
            output_schema=self.output_schema,
        )
        if result.finish == "pending":
            for pending in result.tool_results:
                if pending.status == "pending":
                    pending.unwrap()  # raises the ApprovalPending it holds
        output = result.final.parsed if self.output_schema is not None else result.final.text
        return Outcome(output, "" if result.finish == "stop" else f"tool loop {result.finish}")


class FlowTarget:
    """A `Flow` run by `FlowRunner`; `initial_state(case)` builds the state, `output` reads it."""

    def __init__(
        self,
        flow: Flow[Any],
        *,
        initial_state: Callable[[EvalCase], BaseModel | Mapping[str, Any]],
        output: Callable[[Any], Any] | None = None,
        model: LLMPort | None = None,
        name: str = "",
    ) -> None:
        """Raises: ConfigError: `flow` is not a Flow or `initial_state` is not callable."""
        if not isinstance(flow, Flow):
            raise ConfigError(f"FlowTarget needs a Flow, got {type(flow).__name__}")
        if not callable(initial_state):
            raise ConfigError("FlowTarget needs initial_state=, a function of the case")
        self.flow = flow
        self.initial_state = initial_state
        self.output = output
        self.model = model
        self.name = name or flow.name

    async def run_case(self, case: EvalCase, *, executor: Executor, ctx: ToolContext) -> Any:
        """`output(final state)` (the state itself without `output`)."""
        runner = FlowRunner(executor, model=self.model)
        result = await runner.run(self.flow, self.initial_state(case), ctx)
        value = self.output(result.state) if self.output is not None else result.state
        return Outcome(value, "" if result.status == "done" else f"flow {result.status}")


def as_target(target: object) -> Target:
    """`target` as a Target: a Target as is, a 2-argument callable as a chat agent, else a callable.

    Raises:
        ConfigError: a Flow (wrap it in FlowTarget), or not a callable of one or two arguments.
    """
    if isinstance(target, Target):
        return target
    if isinstance(target, Flow):
        raise ConfigError("wrap a Flow in FlowTarget(flow, initial_state=...) to evaluate it")
    if not callable(target):
        raise ConfigError(f"cannot evaluate a {type(target).__name__}: pass a callable or Target")
    required = [
        p
        for p in _parameters(target).values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
        and p.name != "ctx"
    ]
    if len(required) == 2:
        return ChatTarget(target)
    if len(required) == 1:
        return CallableTarget(target)
    raise ConfigError(
        f"{_name_of(target)} takes {len(required)} required arguments: an eval target takes "
        "fn(input) or a chat agent fn(message, session)"
    )


def _parameters(fn: Callable[..., Any]) -> Mapping[str, inspect.Parameter]:
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"cannot read the signature of {_name_of(fn)}") from exc


def _name_of(fn: object) -> str:
    return str(getattr(fn, "__name__", "") or type(fn).__name__)
