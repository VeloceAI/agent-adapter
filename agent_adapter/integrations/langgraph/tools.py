"""Registry tools in LangGraph: a tool node, `Send` tasks and a fan-out helper."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from agent_adapter.approval.store import request_digest
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ToolNotFound, ToolUsageError
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalRequest, ToolCall
from agent_adapter.exec.integration import gather_unwrapped
from agent_adapter.integrations._default_runner import DefaultRunner
from agent_adapter.integrations.base import MODEL_VISIBLE_ERRORS, ToolRunner, model_visible_error
from agent_adapter.integrations.langgraph._context import call_context, task_namespace
from agent_adapter.integrations.langgraph._integration import INTEGRATION, TARGET
from agent_adapter.integrations.langgraph._memo import TaskMemo
from agent_adapter.integrations.langgraph._spent import task_scope
from agent_adapter.integrations.langgraph.approval import (
    LangGraphApproval,
    ask,
    collect_asked,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _Done:
    """A finished call of an interrupted tool node: its message and the approval it used."""

    digest: str  # of the call's name and arguments
    message: Any  # ToolMessage
    asked: tuple[ApprovalRequest, bytes] | None  # answered by a resume value (and its key)


# Finished calls of tool-node tasks that paused, by (run id, task namespace).
TOOL_NODE_MEMO: TaskMemo[dict[str, _Done]] = TaskMemo()


def tool_node(
    registry: Registry,
    *,
    executor: ToolRunner | None = None,
    messages_key: str = "messages",
    concurrency: int | None = None,
) -> ToolNode:
    """A LangGraph node that runs the last AI message's tool calls through the executor.

    Calls whose `confirm` rule says yes run one at a time after the others, so their
    interrupts are raised and resumed in order. `executor=None` uses the default executor
    with `LangGraphApproval()` as its port.

    Raises:
        ToolUsageError: `concurrency` is not None or an int >= 1.
        MissingDependency: langgraph is not installed.
    """
    INTEGRATION.require()
    if concurrency is not None and (
        not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1
    ):
        raise ToolUsageError(f"concurrency must be an int >= 1 or None, got {concurrency!r}")
    runner = executor if executor is not None else _default_runner()
    specs = {spec.name: spec for spec in registry.for_target(TARGET)}
    return ToolNode(specs, runner, messages_key, concurrency)


def _default_runner() -> ToolRunner:
    """The default executor at call time, with one `LangGraphApproval` (its key stays fixed)."""
    return DefaultRunner(approval=LangGraphApproval())


class ToolNode:
    """Runs tool calls as ToolMessages; `tools` are the definitions to bind to a model."""

    name = "tools"

    def __init__(
        self,
        specs: Mapping[str, ToolSpec],
        runner: ToolRunner,
        messages_key: str,
        concurrency: int | None,
    ) -> None:
        self.specs = dict(specs)
        self.runner = runner
        self.messages_key = messages_key
        self.concurrency = concurrency
        self.tools = [_definition(spec) for spec in self.specs.values()]
        self.__name__ = self.name  # `graph.add_node(node)` names the node "tools"

    async def __call__(self, state: Any, config: RunnableConfig) -> Any:
        """The ToolMessages answering the last AI message, as a state update."""
        messages = _messages(state, self.messages_key)
        calls = _tool_calls(messages)
        run = _Run(self, config)
        with task_scope(run.run_id, run.namespace):  # each answer is used once per task run
            return await self._answer(state, run, calls)

    async def _answer(self, state: Any, run: _Run, calls: Sequence[Mapping[str, Any]]) -> Any:
        plain, gated = await run.partition(calls)
        results: dict[int, Any] = {}
        limit = self.concurrency or max(1, len(plain))
        for index, message in zip(
            plain,
            await gather_unwrapped([run.factory(calls[i], i) for i in plain], concurrency=limit),
            strict=True,
        ):
            results[index] = message
        for index in gated:  # one at a time, in call order: one interrupt per resume
            results[index] = await run.one(calls[index], index)
        run.finished()
        ordered = [results[index] for index in range(len(calls))]
        return ordered if isinstance(state, list) else {self.messages_key: ordered}

    def __repr__(self) -> str:
        return f"ToolNode(tools={list(self.specs)})"


class _Run:
    """One execution of a tool node task."""

    def __init__(self, node: ToolNode, config: Any) -> None:
        self.node = node
        self.config = config
        self.namespace = namespace = task_namespace(config)
        self.run_id = call_context(config, {}).run_id
        self.key = (self.run_id, namespace) if namespace and self.run_id else None
        self.done = dict((TOOL_NODE_MEMO.get(self.key) if self.key else None) or {})  # a copy
        self.seen: set[str] = set()

    async def partition(self, calls: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
        plain: list[int] = []
        gated: list[int] = []
        for index, call in enumerate(calls):
            spec = self.node.specs.get(str(call.get("name", "")))
            needs = spec is not None and await _needs_approval(spec, call, self._ctx(call, index))
            (gated if needs else plain).append(index)
        return plain, gated

    def factory(self, call: Mapping[str, Any], index: int) -> Callable[[], Awaitable[Any]]:
        return lambda: self.one(call, index)

    def _ctx(self, call: Mapping[str, Any], index: int) -> ToolContext:
        native = {**call, "id": self._call_id(call, index)}
        return call_context(self.config, native)

    def _call_id(self, call: Mapping[str, Any], index: int) -> str:
        given = call.get("id")
        if isinstance(given, str) and given:
            return given
        namespace = task_namespace(self.config) or "tools"
        return f"{namespace}:{index}"  # stable across re-runs of this task

    async def one(self, call: Mapping[str, Any], index: int) -> Any:
        name, args = str(call.get("name", "")), call.get("args") or {}
        call_id = self._call_id(call, index)
        if call_id in self.seen:
            return _error(name, call_id, ToolUsageError(f"duplicate tool call id {call_id!r}"))
        self.seen.add(call_id)
        spec = self.node.specs.get(name)
        if spec is None:
            return _error(name, call_id, ToolNotFound(f"no tool named {name!r} in this node"))
        digest = _call_digest(name, args)
        done = self.done.get(call_id)
        if done is not None and done.digest == digest:
            if done.asked is not None:  # take this call's resume position again (re-checked)
                request, key = done.asked
                ask(request, key=key, started=False)  # its call does not run again
            logger.debug("tool node: call %s replayed after a resume", call_id)
            return done.message
        message, asked = await self._execute(spec, args, self._ctx(call, index))
        self.done[call_id] = _Done(digest, message, asked)
        if self.key is not None:
            TOOL_NODE_MEMO.put(self.key, dict(self.done))
        return message

    async def _execute(
        self, spec: ToolSpec, args: Mapping[str, Any], ctx: ToolContext
    ) -> tuple[Any, tuple[ApprovalRequest, bytes] | None]:
        with collect_asked() as asked:
            try:
                value = await self.node.runner.run_spec(spec, args, ctx, call_id=ctx.call_id)
            except MODEL_VISIBLE_ERRORS as exc:
                logger.debug("tool node: %s returned %s", spec.name, type(exc).__name__)
                message = _error(spec.name, ctx.call_id, exc)
            else:
                message = _message(spec.name, ctx.call_id, _content(value), "success")
        return message, (asked[-1] if asked else None)

    def finished(self) -> None:
        if self.key is not None:
            TOOL_NODE_MEMO.drop(self.key)


async def _needs_approval(spec: ToolSpec, call: Mapping[str, Any], ctx: ToolContext) -> bool:
    """Whether the call's confirm rule says yes; a rule that fails counts as yes."""
    rule = spec.confirm
    if isinstance(rule, bool):
        return rule
    tool_call = ToolCall(name=spec.name, arguments=dict(call.get("args") or {}), id=ctx.call_id)
    try:
        answer = rule(tool_call, ctx)
        if inspect.isawaitable(answer):
            answer = await answer
    except Exception as exc:
        if is_control_flow(exc):
            raise
        logger.debug("tool node: confirm rule of %s failed; ordering it last", spec.name)
        return True
    return answer is not False


def _messages(state: Any, key: str) -> list[Any]:
    if isinstance(state, list):
        return state
    found = state.get(key) if isinstance(state, Mapping) else getattr(state, key, None)
    return list(found or [])


def _tool_calls(messages: Sequence[Any]) -> list[Mapping[str, Any]]:
    last = messages[-1] if messages else None
    calls = getattr(last, "tool_calls", None)
    if getattr(last, "type", None) != "ai" or not calls:
        raise ToolUsageError("tool_node needs an AIMessage with tool_calls as the last message")
    return list(calls)


def _call_digest(name: str, args: Any) -> str:
    try:
        return request_digest(
            ApprovalRequest(approval_id="", run_id="", call_id="", tool=name, arguments=args)
        )
    except (ToolUsageError, ValueError):
        return repr((name, args))


def _definition(spec: ToolSpec) -> dict[str, Any]:
    function = {"name": spec.name, "description": spec.description}
    return {"type": "function", "function": {**function, "parameters": spec.json_schema()}}


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _message(name: str, call_id: str, content: str, status: str) -> Any:
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=content, name=name, tool_call_id=call_id, status=status)


def _error(name: str, call_id: str, exc: BaseException) -> Any:
    return _message(name, call_id, json.dumps(model_visible_error(exc)), "error")


class ToolTask:
    """A `Send` target: runs one tool with the `Send` argument and writes `{into: value}`."""

    def __init__(self, spec: ToolSpec, runner: ToolRunner, into: str) -> None:
        self.spec = spec
        self.runner = runner
        self.into = into
        self.name = self.__name__ = spec.name

    async def __call__(self, arg: Any, config: RunnableConfig) -> dict[str, Any]:
        """Run the tool; model-visible errors become `model_visible_error` values."""
        if not isinstance(arg, Mapping):
            raise ToolUsageError(
                f"tool_task {self.spec.name!r} needs a mapping of arguments, "
                f"got {type(arg).__name__}"
            )
        call_id = task_namespace(config)  # names this Send task: stable across re-runs
        ctx = call_context(config, {"name": self.spec.name, "args": dict(arg), "id": call_id})
        with task_scope(ctx.run_id, call_id):  # a model-visible error still ends the task
            try:
                value = await self.runner.run_spec(
                    self.spec, dict(arg), ctx, call_id=call_id or None
                )
            except MODEL_VISIBLE_ERRORS as exc:
                logger.debug("tool task %s: %s", self.spec.name, type(exc).__name__)
                value = model_visible_error(exc)
        return {self.into: value}

    def __repr__(self) -> str:
        return f"ToolTask({self.spec.name!r}, into={self.into!r})"


def tool_task(spec: ToolSpec, *, executor: ToolRunner | None = None, into: str) -> ToolTask:
    """A node for `Send(node, args)`: runs `spec` with `args`, returns `{into: value}`.

    Raises:
        ToolUsageError: `spec.targets` excludes langgraph.
        MissingDependency: langgraph is not installed.
    """
    check_target(spec, TARGET)
    INTEGRATION.require()
    runner = executor if executor is not None else _default_runner()
    return ToolTask(spec, runner, into)


def fan_out(
    target: str, over: str | Callable[[Any], Iterable[Mapping[str, Any]]]
) -> Callable[[Any], list[Any]]:
    """A conditional-edge path sending each item (a mapping) of `over` to node `target`.

    `over` is a state key holding a list, or a function of the state returning the items.
    """
    INTEGRATION.require()

    def route(state: Any) -> list[Any]:
        from langgraph.types import Send

        items = over(state) if callable(over) else _read(state, over)
        sends = []
        for item in items or ():
            if not isinstance(item, Mapping):
                raise ToolUsageError(
                    f"fan_out to {target!r}: each item must be a mapping, got {type(item).__name__}"
                )
            sends.append(Send(target, dict(item)))
        return sends

    return route


def _read(state: Any, key: str) -> Any:
    return state.get(key) if isinstance(state, Mapping) else getattr(state, key, None)


__all__ = ["ToolNode", "ToolTask", "fan_out", "tool_node", "tool_task"]
