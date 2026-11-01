"""The `AgentExecutor`: one A2A task runs one registry tool through the executor."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError, ApprovalPending, log_failure
from agent_adapter.core.registry import Registry, exported_to
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, to_jsonable
from agent_adapter.exec.control import is_control_flow
from agent_adapter.exec.defaults import runner_for
from agent_adapter.integrations._native_gate import (
    TokenTable,
    denial,
    dump_arguments,
    needs_approval,
    take_approval,
)
from agent_adapter.integrations.a2a._pending import PendingCall, PendingCalls
from agent_adapter.integrations.a2a._wire import (
    JSON_MEDIA,
    MAX_DATA_BYTES,
    TARGET,
    WireError,
    approval_payload,
    parse_call,
    parse_decision,
    require_a2a,
)
from agent_adapter.integrations.base import ToolRunner, model_visible_error
from agent_adapter.runtime.ids import new_id

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.tasks import TaskUpdater

logger = logging.getLogger(__name__)

PRINCIPAL_STATE = "aa.a2a.principal"  # ServerCallContext.state key the app's gate fills
DECIDED_BY = "a2a-client"
Authorizer = Callable[[Any, ToolSpec, Mapping[str, Any] | None], Awaitable[bool]]


def owner_ref(owner: str) -> str:
    """A short hash of an owner id for logs (ids may be e-mails)."""
    if not owner:
        return "anonymous"
    return hashlib.sha256(owner.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def scoped_run_id(owner: str, context_id: str) -> str:
    """The run id of an A2A context, scoped by its owner so clients never share runs."""
    digest = hashlib.sha256(f"{owner}\0{context_id}".encode("utf-8", "surrogatepass"))
    return "a2a_" + digest.hexdigest()[:32]


class ToolAgent:
    """Reads a `{tool, arguments}` data part, runs it, and reports through `TaskUpdater`.

    A pending approval parks the call server-side and moves the task to `input-required`.
    The next message on that task re-runs the parked call: the executor's port decides,
    or, with `client_approval`, an explicit `{approval_id, approved}` answer becomes a
    one-use token bound to the run, task id, tool and argument digest.
    """

    def __init__(
        self,
        registry: Registry,
        executor: ToolRunner | None = None,
        *,
        client_approval: bool = False,
        authorize: Authorizer | None = None,
        max_data_bytes: int = MAX_DATA_BYTES,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.client_approval = client_approval
        self.authorize = authorize
        self.max_data_bytes = max_data_bytes
        self.pending = PendingCalls()
        self.tokens = TokenTable()

    async def execute(self, context: Any, event_queue: Any) -> None:
        """Run the call in `context.message`, or resume the task's parked call."""
        from a2a.helpers import get_data_parts
        from a2a.server.tasks import TaskUpdater
        from a2a.types import Task, TaskState, TaskStatus

        task_id, context_id = context.task_id or "", context.context_id or ""
        principal, owner = _caller(context)
        message = context.message
        parts = get_data_parts(message.parts) if message is not None else []
        updater = TaskUpdater(event_queue, task_id, context_id)
        if context.current_task is None:
            history = [message] if message is not None else []
            status = TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)
            await event_queue.enqueue_event(
                Task(id=task_id, context_id=context_id, status=status, history=history)
            )
            await self._start(updater, parts, principal, owner, context)
            return
        parked = self.pending.take(owner, task_id)
        if parked is None:
            await _refuse(updater, "bad-request", "this task is not waiting for an approval")
            return
        await self._resume(updater, parked, parts, principal, owner, context)

    async def cancel(self, context: Any, event_queue: Any) -> None:
        """Forget the parked call and mark the task cancelled."""
        from a2a.server.tasks import TaskUpdater

        _, owner = _caller(context)
        self.pending.drop(owner, context.task_id or "")
        await TaskUpdater(event_queue, context.task_id or "", context.context_id or "").cancel()

    async def _start(
        self, updater: TaskUpdater, parts: list[Any], principal: Any, owner: str, context: Any
    ) -> None:
        try:
            call = parse_call(parts, limit=self.max_data_bytes)
        except WireError as exc:
            log_failure(logger, logging.WARNING, exc, "A2A message refused")
            await _refuse(updater, "bad-request", str(exc))
            return
        spec = await self._allowed(updater, call.tool, call.arguments, principal)
        if spec is None:
            return
        ctx = self._context(context, principal, scoped_run_id(owner, context.context_id or ""))
        await updater.start_work()
        if self.client_approval:
            dumped = dump_arguments(spec, call.arguments)
            if dumped is not None and await needs_approval(
                spec, dumped, run_id=ctx.run_id, call_id=ctx.call_id
            ):
                parked = PendingCall(
                    spec.name, call.arguments, ctx.run_id, new_id("apr_"), "client"
                )
                await self._park(updater, owner, ctx.call_id, parked)
                return
        await self._run(updater, spec, call.arguments, ctx, owner)

    async def _resume(
        self,
        updater: TaskUpdater,
        parked: PendingCall,
        parts: list[Any],
        principal: Any,
        owner: str,
        context: Any,
    ) -> None:
        spec = await self._allowed(updater, parked.tool, parked.arguments, principal)
        if spec is None:
            return
        ctx = self._context(context, principal, parked.run_id)
        decision = parse_decision(parts)
        if parked.mode == "port":
            if decision is not None:
                logger.info(
                    "A2A task %s: a client decision is ignored; the port decides", ctx.call_id
                )
        elif decision is None or decision.approval_id != parked.approval_id:
            logger.warning(
                "A2A task %s: no answer for approval %s", ctx.call_id, parked.approval_id
            )
            await self._park(updater, owner, ctx.call_id, parked)
            return
        else:
            ctx = dataclasses.replace(
                ctx, approval=self._client_decision(spec, parked, ctx, decision)
            )
        await updater.start_work()
        await self._run(updater, spec, parked.arguments, ctx, owner)

    def _client_decision(
        self, spec: ToolSpec, parked: PendingCall, ctx: ToolContext, answer: Any
    ) -> ApprovalDecision | None:
        """A one-use token for an explicit yes; a denial otherwise."""
        logger.info(
            "A2A client %s tool %s (task %s)",
            "approved" if answer.approved else "denied",
            spec.name,
            ctx.call_id,
        )
        if not answer.approved:
            return denial(answer.reason or "denied by the A2A client", by=DECIDED_BY)
        dumped = dump_arguments(spec, parked.arguments)
        if dumped is None:
            return None  # invalid arguments: the executor rejects them before approval
        yes = ApprovalDecision(
            approved=True, reason=answer.reason, decided_by=DECIDED_BY, decided_at=time.time()
        )
        self.tokens.issue(spec.name, dumped, yes, call_id=ctx.call_id, run_id=ctx.run_id)
        return take_approval(self.tokens, spec, parked.arguments, ctx)

    async def _run(
        self,
        updater: TaskUpdater,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
        owner: str,
    ) -> None:
        from a2a.helpers import new_text_part

        try:
            value = await runner_for(self.executor).run_spec(
                spec, arguments, ctx, call_id=ctx.call_id
            )
            # JSON text, not a data part: protobuf `Value` would turn every int into a float.
            text = json.dumps(to_jsonable(value), ensure_ascii=False, allow_nan=False)
            part = new_text_part(text, media_type=JSON_MEDIA)
        except ApprovalPending as exc:
            parked = PendingCall(spec.name, arguments, ctx.run_id, exc.approval_id, "port")
            await self._park(updater, owner, ctx.call_id, parked)
            return
        except Exception as exc:
            if is_control_flow(exc):
                raise
            _log_failure(spec, ctx, exc)
            await _fail(updater, exc)
            return
        await updater.add_artifact([part], name="result")
        await updater.complete()
        logger.info("A2A task %s: tool %s completed", ctx.call_id, spec.name)

    async def _park(
        self, updater: TaskUpdater, owner: str, task_id: str, parked: PendingCall
    ) -> None:
        from a2a.helpers import new_data_part, new_text_part

        if not self.pending.park(owner, task_id, parked):
            await _refuse(
                updater, "too-many-pending", "too many of your calls already wait for approval"
            )
            return
        payload = approval_payload(parked.approval_id, parked.tool, client=parked.mode == "client")
        message = updater.new_agent_message(
            [
                new_text_part(f"approval required: {parked.approval_id}"),
                new_data_part(payload, media_type=JSON_MEDIA),
            ]
        )
        logger.info(
            "A2A task %s (owner %s): tool %s waits for approval %s",
            task_id,
            owner_ref(owner),
            parked.tool,
            parked.approval_id,
        )
        await updater.requires_input(message)

    async def _allowed(
        self, updater: TaskUpdater, name: str, arguments: Mapping[str, Any], principal: Any
    ) -> ToolSpec | None:
        spec = self.registry.get(name) if name in self.registry else None
        if spec is None or not exported_to(spec, TARGET):
            logger.warning("A2A call to unknown or unexported tool %r", name)
            await _refuse(updater, "tool-not-found", f"unknown tool {name!r}")
            return None
        if self.authorize is None:
            return spec
        try:
            allowed = await self.authorize(principal, spec, arguments)
        except Exception as exc:  # a raising hook refuses (fails closed)
            if is_control_flow(exc):
                raise
            log_failure(logger, logging.ERROR, exc, "A2A authorize hook failed for %s", name)
            allowed = False
        if allowed is not True:
            logger.warning("A2A call to %s refused by authorize", name)
            await _refuse(updater, "forbidden", f"not allowed to call tool {name!r}")
            return None
        return spec

    @staticmethod
    def _context(context: Any, principal: Any, run_id: str) -> ToolContext:
        return ToolContext(
            run_id=run_id,
            call_id=context.task_id or "",
            extras={TARGET: context, "principal": principal},
        )


def _caller(context: Any) -> tuple[Any, str]:
    """The gate's principal (or None) and the owner id that scopes tasks and runs."""
    call_context = context.call_context
    principal = call_context.state.get(PRINCIPAL_STATE)
    if principal is not None:
        owner = getattr(principal, "id", "")
        return principal, owner if isinstance(owner, str) else ""
    return None, call_context.user.user_name or ""


def _log_failure(spec: ToolSpec, ctx: ToolContext, exc: Exception) -> None:
    if isinstance(exc, AdapterError):
        logger.warning("A2A task %s: tool %s failed: %s", ctx.call_id, spec.name, type(exc).kind)
    else:
        log_failure(
            logger, logging.ERROR, exc, "A2A task %s: tool %s raised", ctx.call_id, spec.name
        )


async def _fail(updater: TaskUpdater, exc: BaseException) -> None:
    from a2a.helpers import new_data_part, new_text_part

    payload = model_visible_error(exc)
    error = payload["error"]
    parts = [
        new_text_part(f"{error['kind']}: {error['message']}"),
        new_data_part(payload, media_type=JSON_MEDIA),
    ]
    await updater.failed(updater.new_agent_message(parts))


async def _refuse(updater: TaskUpdater, kind: str, text: str) -> None:
    from a2a.helpers import new_data_part, new_text_part

    payload = {"error": {"kind": kind, "message": text}}
    parts = [new_text_part(f"{kind}: {text}"), new_data_part(payload, media_type=JSON_MEDIA)]
    await updater.reject(updater.new_agent_message(parts))


def a2a_executor(
    registry: Registry,
    *,
    executor: ToolRunner | None = None,
    client_approval: bool = False,
    authorize: Authorizer | None = None,
    max_data_bytes: int = MAX_DATA_BYTES,
) -> AgentExecutor:
    """An `a2a-sdk` `AgentExecutor` running `registry.for_target("a2a")` tools.

    Raises:
        MissingDependency: `a2a-sdk` is not installed.
    """
    require_a2a()
    agent: AgentExecutor = _executor_class()(
        registry,
        executor,
        client_approval=client_approval,
        authorize=authorize,
        max_data_bytes=max_data_bytes,
    )
    return agent


@functools.cache
def _executor_class() -> type[Any]:
    from a2a.server.agent_execution import AgentExecutor

    class ToolAgentExecutor(ToolAgent, AgentExecutor):  # type: ignore[misc,unused-ignore]
        """`ToolAgent` as the SDK's `AgentExecutor`."""

    return ToolAgentExecutor
