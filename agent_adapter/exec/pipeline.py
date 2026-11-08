"""The per-call pipeline: one ToolCall in, one ToolResult out, in the contract's fixed order."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from pydantic import BaseModel

from agent_adapter.core._attempt import ATTEMPT, Attempt
from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import control_flow_member
from agent_adapter.core.errors import (
    AdapterError,
    ApprovalPending,
    ToolFailed,
    ToolNotFound,
    ToolTimeout,
    ToolUsageError,
    log_failure,
    qualified_name,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, call_in_attempt
from agent_adapter.core.types import ErrorInfo, LedgerEntry, ToolCall, ToolResult
from agent_adapter.exec._approve import approve
from agent_adapter.exec._call_content import NOT_PASSED, emit_finished
from agent_adapter.exec._call_steps import NOT_STORED, CallSteps
from agent_adapter.exec._journal import Journal, JournalSlot
from agent_adapter.exec._resume import current_journal, restore_visit, visit_snapshot
from agent_adapter.exec._settings import _Settings
from agent_adapter.exec.args import dump_args
from agent_adapter.exec.guards import (
    CallIdentity,
    _call_identity,
    _run_slot,
    _RunState,
)
from agent_adapter.exec.retry import _attempt
from agent_adapter.hooks.base import ToolInvocation
from agent_adapter.hooks.dispatch import run_after, run_before, run_on_error
from agent_adapter.runtime.current import current_context, use_context
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import first_sink

logger = logging.getLogger(__name__)


def current_attempt() -> Attempt | None:
    """The executor attempt running now (None outside one, e.g. a direct `spec.invoke`)."""
    return ATTEMPT.get()


P = TypeVar("P")


async def _execute(
    settings: _Settings,
    call: ToolCall,
    ctx: ToolContext,
    *,
    spec: ToolSpec | None = None,
    registry: Registry | None = None,
    call_id: str | None = None,
    single: bool = False,
) -> ToolResult:
    """Run one call; failures become results, control flow propagates. `ctx.run_id` is set.

    `single`: `ctx` was passed for this one call (`run`, `run_spec`, `run_call`); only then
    does `ctx.approval` apply, and only when `ctx.call_id` is empty or this call's id.
    Inside a resumed flow node, a call the node already finished returns its recorded result
    (no guards, approval, ledger or events); new finished calls are recorded.
    """
    journal = current_journal()
    slot = _claim(journal, call, spec, registry) if journal is not None else None
    if slot is not None and journal is not None:
        await journal.started(slot)  # a durable run logs the call (or its replay) first
    if slot is not None and slot.result is not None:
        logger.debug("tool %s replayed from the node journal (run %s)", slot.tool, ctx.run_id)
        own_id = call.id or call_id or slot.result.call_id
        return dataclasses.replace(slot.result, call_id=own_id)
    run = _Call(settings, call, ctx, call_id, single=single)
    with use_context(run.ctx):
        result = await run.execute(spec, registry)
    if slot is not None and journal is not None:
        await journal.settle(slot, result)
    return result


def _claim(
    journal: Journal, call: ToolCall, spec: ToolSpec | None, registry: Registry | None
) -> JournalSlot | None:
    """The journal slot of a call, keyed by the validated arguments the caller passed.

    Unset fields are left out at every depth (a `default_factory` differs on every call); the
    arguments as given are used when they do not validate.
    """
    name = spec.name if spec is not None else call.name
    arguments: Any = call.arguments
    found: ToolSpec | None = None
    try:
        found = spec if spec is not None else registry.get(name) if registry else None
        if found is not None:
            validated = found.validate_args(call.arguments)
            arguments = validated.model_dump(mode="python", by_alias=True, exclude_unset=True)
    except (AdapterError, ValueError):  # ValueError: a field serializer failed
        logger.debug("tool %s: journal key from the arguments as given", name)
    return journal.claim(name, arguments, found.returns if found is not None else None)


class _Call(CallSteps):
    """State of one call through steps 1-12."""

    def __init__(
        self,
        settings: _Settings,
        call: ToolCall,
        ctx: ToolContext,
        call_id: str | None,
        *,
        single: bool = False,
    ) -> None:
        self.settings = settings
        self.name = call.name
        self.arguments = call.arguments
        self.call_id = call.id or call_id or new_id("call_")
        self.parent_id = _parent_id(ctx, self.call_id)
        bound = single and ctx.call_id in ("", self.call_id)
        # Held by the call, never by its context: the tool body and its nested calls get none.
        self.pre_approval = ctx.approval if bound else None
        self.ctx = ctx.child(
            call_id=self.call_id,
            ledger=_first(ctx.ledger, settings.ledger),
            cache=_first(ctx.cache, settings.cache),
            sink=first_sink(ctx.sink, settings.sink),  # an empty hub falls through
        )
        self._event_ctx = ToolContext(run_id=ctx.run_id, call_id=self.parent_id)
        self.attempts = 0
        self.spec: ToolSpec | None = None
        self.passed: Any = NOT_PASSED  # the arguments, once the before-tool guards passed
        self.started = settings.clock()
        self.started_at = time.time()

    def emit(self, type: str, /, **data: Any) -> None:
        emit(self.ctx.sink, make_event(type, self._event_ctx, call_id=self.call_id, **data))

    def emit_finished(self, value: Any, /, **data: Any) -> None:
        event = make_event(EventType.TOOL_FINISHED, self._event_ctx, call_id=self.call_id, **data)
        emit_finished(self.ctx.sink, event, self.spec, self.passed, value)

    def latency_ms(self) -> float:
        return max(0.0, (self.settings.clock() - self.started) * 1000)

    async def execute(self, spec: ToolSpec | None, registry: Registry | None) -> ToolResult:
        stored: Any = NOT_STORED
        found: ToolSpec | None = None
        try:
            found = self._resolve(spec, registry)
            self.spec = found
            self.emit(EventType.TOOL_STARTED, tool=self.name, kind=found.kind)
            result, stored = await self._admitted(found)
        except BaseException as exc:
            member = control_flow_member(exc)
            if member is exc or not isinstance(exc, Exception):
                raise
            if member is not None:
                raise member from member.__cause__
            if found is None:
                self.emit(EventType.TOOL_STARTED, tool=self.name, kind="")
            result = self._failure(exc)
        self._record(result)
        if stored is not NOT_STORED and result.ok and found is not None:
            await self._store(found, stored)
        self._finish(result)
        return result

    def _resolve(self, spec: ToolSpec | None, registry: Registry | None) -> ToolSpec:
        if spec is not None:
            self.name = spec.name
            found = spec
        elif registry is None:
            raise ToolNotFound(f"no tool named {self.name!r}: the executor has no registry")
        else:
            found = registry.get(self.name)
        policy = self.settings.policy
        return found if policy is None else policy.effective(found)

    async def _admitted(self, spec: ToolSpec) -> tuple[ToolResult, Any]:
        runs = self.settings.runs
        identity = _call_identity(self.call_id, spec.name, self.arguments)
        state = runs.admit(self.ctx.run_id, identity, **self._guards(spec))
        try:
            return await self._steps(spec, state, identity)
        finally:
            runs.leave(state)

    def _guards(self, spec: ToolSpec) -> dict[str, Any]:
        return {"once_per_run": spec.once_per_run, "max_calls": self.settings.limits.max_calls}

    async def _steps(
        self, spec: ToolSpec, state: _RunState, identity: CallIdentity
    ) -> tuple[ToolResult, Any]:
        hooks = self.settings.hooks
        validated = spec.validate_args(self.arguments)
        inv = ToolInvocation(spec, self._tool_call(validated), self.ctx)
        stored: Any = NOT_STORED
        short = await run_before(hooks, "before_tool", inv)
        if short is None:
            validated, approved = await self._approve(state, validated, inv)
            if approved.call.arguments != inv.call.arguments:
                # edited arguments pass the same checks (guardrails, budget) as the originals
                short = await run_before(hooks, "before_tool", approved)
            inv = approved
        self.passed = inv.call.arguments
        if short is not None:
            result = self._ok(short.value)
        else:
            self.settings.runs.commit(state, identity, **self._guards(spec))
            cached = await self._lookup(spec, inv)
            if cached is not NOT_STORED:
                result = self._ok(cached, cached=True)
            else:
                result, stored = await self._run(spec, state, validated, inv)
        replaced = await run_after(hooks, "after_tool", inv, result)
        if not isinstance(replaced, ToolResult):
            raise ToolUsageError(
                f"after_tool hooks must return a ToolResult or None, got {type(replaced).__name__}"
            )
        return replaced, stored

    async def _approve(
        self, state: _RunState, validated: BaseModel, inv: ToolInvocation
    ) -> tuple[BaseModel, ToolInvocation]:
        runs = self.settings.runs
        resumed = runs.pending_approval(state, self.call_id)
        waiting = ""
        try:
            return await approve(
                inv,
                validated,
                pre_approval=self.pre_approval,
                port=self.settings.approval,
                emit=self.emit,
                approval_id=resumed,
            )
        except ApprovalPending as exc:
            waiting = exc.approval_id
            raise
        finally:
            if waiting or resumed:
                runs.set_pending_approval(state, self.call_id, waiting)

    def _tool_call(self, validated: BaseModel) -> ToolCall:
        return ToolCall(name=self.name, arguments=dump_args(validated), id=self.call_id)

    def _attempt_fn(
        self, spec: ToolSpec, validated: BaseModel
    ) -> Callable[[], Coroutine[Any, Any, Any]]:
        """One call of the tool per attempt, with `current_attempt()` set while it runs."""
        number = 0

        async def call() -> Any:
            nonlocal number
            number += 1
            attempt = Attempt(self.call_id, number, spec.retries + 1)
            return await call_in_attempt(spec, validated, self.ctx, attempt)

        return call

    async def _run(
        self, spec: ToolSpec, state: _RunState, validated: BaseModel, inv: ToolInvocation
    ) -> tuple[ToolResult, Any]:
        settings = self.settings
        async with _run_slot(
            state,
            self.call_id,
            self.parent_id,
            exclusive=not spec.parallel_safe,
            kind=spec.kind,
            cap=settings.limits.per_kind.get(spec.kind),
        ):
            next_attempt = self._attempt_fn(spec, validated)
            outcome = await _attempt(
                lambda: _guarded(next_attempt()),
                retries=spec.retries,
                timeout=spec.timeout,
                backoff=settings.backoff,
                rng=settings.rng,
                sleep=settings.sleep,
                on_timeout=lambda: ToolTimeout(
                    spec.name, spec.timeout or 0.0, call_id=self.call_id
                ),
                on_retry=self._retrying,
            )
        self.attempts = outcome.attempts
        if outcome.error is None:
            return self._ok(outcome.value), (inv.call.arguments, outcome.value)
        error = self._final_error(spec, outcome.error)
        if isinstance(error, ApprovalPending):
            raise error  # a nested call waits for approval: not a failure to recover from
        recovered = await run_on_error(settings.hooks, "on_tool_error", inv, error)
        if recovered is None:
            raise error
        logger.info("tool %s recovered by an on_tool_error hook (call %s)", spec.name, self.call_id)
        return self._ok(recovered.value), NOT_STORED

    def _retrying(self, number: int, delay: float, error: Exception) -> None:
        kind = ErrorInfo.from_exception(error).kind
        logger.warning(
            "tool %s attempt %d failed (%s, run %s, call %s); retrying in %.3gs",
            self.name,
            number,
            kind,
            self.ctx.run_id,
            self.call_id,
            delay,
        )
        self.emit(
            EventType.TOOL_RETRY, tool=self.name, attempt=number, delay_s=delay, error_kind=kind
        )

    def _final_error(self, spec: ToolSpec, error: Exception) -> Exception:
        if isinstance(error, ToolTimeout) or (
            isinstance(error, AdapterError) and not type(error).retryable
        ):
            return error
        failed = ToolFailed(spec.name, call_id=self.call_id, attempts=self.attempts)
        failed.__cause__ = error
        failed.add_note(
            f"cause: {qualified_name(type(error))} (its text is hidden from the model and "
            f"logged at DEBUG by {__name__}); raise ToolError('...') in the tool to show a "
            "message to the model"
        )
        return failed

    def _failure(self, exc: Exception) -> ToolResult:
        pending = isinstance(exc, ApprovalPending)
        if isinstance(exc, ToolFailed):
            cause = exc.__cause__ if exc.__cause__ is not None else exc
            # A tool's exception text can carry secrets: only its type at ERROR.
            logger.error(
                "tool %s failed (%s, run %s, call %s)",
                self.name,
                qualified_name(type(cause)),
                self.ctx.run_id,
                self.call_id,
            )
            logger.debug("tool %s failure (call %s)", self.name, self.call_id, exc_info=exc)
        elif isinstance(exc, ToolTimeout):
            logger.warning(
                "tool %s timed out (run %s, call %s)", self.name, self.ctx.run_id, self.call_id
            )
        else:
            logger.info("tool %s: %s (call %s)", self.name, type(exc).__name__, self.call_id)
        return ToolResult(
            call_id=self.call_id,
            name=self.name,
            status="pending" if pending else "error",
            error=ErrorInfo.from_exception(exc),
            exception=exc,
            attempts=self.attempts,
            latency_ms=self.latency_ms(),
        )

    def _record(self, result: ToolResult) -> None:
        ledger = self.ctx.ledger
        if ledger is None or result.status == "pending":
            return
        entry = LedgerEntry(
            kind="tool",
            run_id=self.ctx.run_id,
            call_id=self.call_id,
            entry_id=self.call_id,
            stage=self.name,
            provider="tool",
            model=self.name,
            cost_usd=0.0,
            latency_ms=result.latency_ms,
            started_at=self.started_at,
            ended_at=time.time(),
            ok=result.ok,
            error_kind=result.error.kind if result.error is not None else "",
            cache_hit=result.cached,
            attempts=result.attempts,
            thread_id=self.ctx.thread_id,
        )
        try:
            ledger.record(entry)
        except Exception as exc:
            # The port must not raise; a broken one must not fail the call it measures.
            log_failure(
                logger, logging.WARNING, exc, "ledger.record failed for call %s", self.call_id
            )


def _first(own: P | None, fallback: P | None) -> P | None:
    """The caller's port when set (even a falsy one, e.g. an empty ledger), else the executor's."""
    return own if own is not None else fallback


def _parent_id(ctx: ToolContext, call_id: str) -> str:
    """The enclosing call: the current context's call in this run, else the caller's ctx."""
    current = current_context()
    if current is not None and current.run_id == ctx.run_id and current.call_id != call_id:
        return current.call_id
    return ctx.call_id if ctx.call_id != call_id else ""


async def _guarded(work: Coroutine[Any, Any, P]) -> P:
    """Run a tool body; a flow visit scope it changes is put back for its caller."""
    saved = visit_snapshot()
    try:
        return await work
    finally:
        restore_visit(saved)
