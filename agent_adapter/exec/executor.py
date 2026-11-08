"""The Executor: runs tools through the fixed pipeline, alone, in batches, plans or streams."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from agent_adapter.core._default import default_registry
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalNotFound, ToolUsageError
from agent_adapter.core.ports import ApprovalPort, CachePort, EventSink, LedgerPort, LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ToolCall, ToolResult
from agent_adapter.exec._batches import Batches
from agent_adapter.exec._settings import ToolPolicy, _Settings
from agent_adapter.exec._sugar import (
    ToolRef,
    check_args,
    check_options,
    resume_context,
    resume_target,
    tool_name,
)
from agent_adapter.exec._sync_twins import SyncTwins
from agent_adapter.exec.guards import RunLimits, _RunTracker
from agent_adapter.exec.pipeline import _execute
from agent_adapter.exec.retry import Backoff
from agent_adapter.hooks.base import Hook
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import SinkHub

logger = logging.getLogger(__name__)


class Executor(Batches, SyncTwins):
    """Runs tools with run guards, hooks, approval, cache, retries, ledger and events.

    With no `registry` it uses the default one (`@tool`). Sync code uses the `SyncTwins`
    methods. Run state (guards, run locks) is shared with every executor made by `replace`.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        hooks: Sequence[Hook] = (),
        approval: ApprovalPort | None = None,
        cache: CachePort | None = None,
        ledger: LedgerPort | None = None,
        sink: EventSink | None = None,
        limits: RunLimits | None = None,
        backoff: Backoff | None = None,
        namespace: str = "aa",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
        max_tracked_runs: int = 10_000,
        policy: ToolPolicy | None = None,
    ) -> None:
        check_options(registry, approval)
        if policy is not None and not isinstance(policy, ToolPolicy):
            raise ToolUsageError(
                f"policy must be a ToolPolicy (e.g. a Policy), got {type(policy).__name__}"
            )
        if registry is None:
            # Every @tool of every imported module is then callable, by name, by a model too.
            registry = default_registry
            logger.info(
                "executor uses the process-wide default registry (%d tools); pass a "
                "Registry to limit what it runs",
                len(registry),
            )
        self._options: dict[str, Any] = {
            "registry": registry,
            "hooks": tuple(hooks),
            "approval": approval,
            "cache": cache,
            "ledger": ledger,
            "sink": sink,
            "limits": limits,
            "backoff": backoff,
            "namespace": namespace,
            "clock": clock,
            "sleep": sleep,
            "rng": rng,
            "max_tracked_runs": max_tracked_runs,
            "policy": policy,
        }
        bad_hooks = [h for h in self._options["hooks"] if not isinstance(h, Hook)]
        if bad_hooks:
            raise ToolUsageError(f"hooks must be Hook instances, got {bad_hooks!r}")
        # Contexts and models capture the hub, so add_sink reaches them later too.
        self._hub = SinkHub(sink)
        # The policy's hooks and limits are added here, not kept in the options: replace(hooks=,
        # limits=) changes only the caller's own, never what the policy enforces.
        enforced = policy.hooks() if policy is not None else ()
        self._settings = _Settings(
            hooks=(*enforced, *self._options["hooks"]),
            approval=approval,
            cache=cache,
            ledger=ledger,
            sink=self._hub,
            limits=_limits(policy, limits or RunLimits()),
            backoff=backoff or Backoff(),
            namespace=namespace,
            clock=clock,
            sleep=sleep,
            rng=rng,
            runs=_RunTracker(max_tracked_runs),
            policy=policy,
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Refuse pickling with a clear error (it holds locks and loop-bound state).

        Raises:
            ToolUsageError: always.
        """
        raise ToolUsageError(
            f"{type(self).__name__} cannot be pickled: build it on the worker with a "
            "module-level functools.cache factory (see docs/backends/pipelines.md)"
        )

    @property
    def registry(self) -> Registry:
        return self._options["registry"]  # type: ignore[no-any-return]

    @property
    def hooks(self) -> tuple[Hook, ...]:
        """The executor's own hooks; a policy's hooks run before them and are not listed."""
        return self._options["hooks"]  # type: ignore[no-any-return]

    @property
    def approval(self) -> ApprovalPort | None:
        return self._settings.approval

    @property
    def cache(self) -> CachePort | None:
        return self._settings.cache

    @property
    def ledger(self) -> LedgerPort | None:
        return self._settings.ledger

    @property
    def sink(self) -> SinkHub:
        """The executor's stable sink: the constructor's `sink`, then every added one."""
        return self._hub

    @property
    def limits(self) -> RunLimits:
        return self._settings.limits

    @property
    def backoff(self) -> Backoff:
        return self._settings.backoff

    @property
    def namespace(self) -> str:
        return self._settings.namespace

    @property
    def policy(self) -> ToolPolicy | None:
        """The policy every call's spec passes through (`Policy.apply` sets it), or None."""
        return self._settings.policy

    def replace(self, **changes: Any) -> Executor:
        """A new Executor with some constructor arguments replaced; `self` is unchanged.

        Raises:
            ToolUsageError: an unknown argument name.
        """
        unknown = sorted(set(changes) - set(self._options))
        if unknown:
            raise ToolUsageError(f"Executor.replace: unknown arguments {unknown}")
        options = {**self._options, **changes}
        executor = Executor(options.pop("registry"), **options)
        if "sink" not in changes:  # a new sink is the caller's whole choice
            for added in self._hub.extra:
                executor._hub.add(added)
        if options["max_tracked_runs"] == self._options["max_tracked_runs"]:
            # A replaced executor is another view of the same runs: guards count across both.
            executor._settings = dataclasses.replace(executor._settings, runs=self._settings.runs)
        return executor

    def add_sink(self, sink: EventSink) -> None:
        """Also send events to `sink`, beside the existing ones; contexts made earlier get it too.

        Raises:
            ToolUsageError: `sink` is not an EventSink.
        """
        if not isinstance(sink, EventSink):
            raise ToolUsageError(f"add_sink needs an EventSink, got {sink!r}")
        if self._hub.add(sink):
            logger.debug("executor sink: added %s", type(sink).__name__)

    def remove_sink(self, sink: EventSink) -> bool:
        """Stop sending events to a sink `add_sink` added; False when it was not added."""
        removed = self._hub.remove(sink)
        if removed:
            logger.debug("executor sink: removed %s", type(sink).__name__)
        return removed

    def model(self, model: LLMPort, *, cache_ttl: float | None = None) -> InstrumentedModel:
        """`model` with this executor's hooks, response cache, ledger and sink around it."""
        return InstrumentedModel(
            model,
            hooks=self._settings.hooks,
            ledger=self.ledger,
            cache=self.cache,
            cache_ttl=cache_ttl,
            sink=self.sink,
            namespace=self.namespace,
        )

    def context(
        self,
        *,
        run_id: str | None = None,
        thread_id: str = "",
        user_id: str = "",
        state: Mapping[str, Any] | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> ToolContext:
        """A context for a new (or given) run, carrying this executor's ledger, cache and sink."""
        return ToolContext(
            run_id=run_id or new_id("run_"),
            state=state if state is not None else {},
            extras=dict(extras or {}),
            thread_id=thread_id,
            user_id=user_id,  # the authenticated end user: memory and retrieval scope by it
            ledger=self.ledger,
            cache=self.cache,
            sink=self.sink,
        )

    def run_scope(self, **context_kwargs: Any) -> AbstractAsyncContextManager[ToolContext]:
        """`async with ex.run_scope() as ctx:` emits run.started/run.finished, then ends the run."""
        return self._run_scope(self.context(**context_kwargs))

    @asynccontextmanager
    async def _run_scope(self, ctx: ToolContext) -> AsyncIterator[ToolContext]:
        emit(ctx.sink, make_event(EventType.RUN_STARTED, ctx))
        ok = False
        try:
            yield ctx
            ok = True
        finally:
            emit(ctx.sink, make_event(EventType.RUN_FINISHED, ctx, ok=ok))
            self.end_run(ctx.run_id)

    def end_run(self, run_id: str) -> None:
        """Forget a run's guard counters and locks."""
        self._settings.runs.end(run_id)

    async def run(
        self,
        tool: ToolRef,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any:
        """Run a tool (its name, spec or function) and return its value; failures raise.

        Raises:
            ToolUsageError: `tool` is not registered here, or `args` is not a mapping.
        """
        call = ToolCall(name=tool_name(tool, self.registry), arguments=check_args(args))
        result = await self._one(call, ctx, call_id=call_id)
        return result.unwrap()

    async def run_spec(
        self,
        spec: ToolSpec,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any:
        """Run `spec` (registered or not) and return its value; failures raise.

        Raises:
            ToolUsageError: `args` is not a mapping.
        """
        call = ToolCall(name=spec.name, arguments=check_args(args))
        result = await self._one(call, ctx, spec=spec, call_id=call_id)
        return result.unwrap()

    async def run_call(
        self, call: ToolCall, ctx: ToolContext | None = None, *, registry: Registry | None = None
    ) -> ToolResult:
        """Run one call; every failure is a ToolResult, only control flow raises."""
        return await self._one(call, ctx, registry=registry)

    async def _one(
        self,
        call: ToolCall,
        ctx: ToolContext | None,
        *,
        spec: ToolSpec | None = None,
        registry: Registry | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        ctx, owned = self._run_context(ctx)
        try:
            return await _execute(
                self._settings,
                call,
                ctx,
                spec=spec,
                registry=self._registry(registry),
                call_id=call_id,
                single=True,
            )
        finally:
            if owned:
                self.end_run(ctx.run_id)

    def _registry(self, registry: Registry | None) -> Registry:
        return registry if registry is not None else self.registry  # an empty one is falsy

    async def resume(
        self,
        approval_id: str,
        decision: bool | ApprovalDecision | None = None,
        ctx: ToolContext | None = None,
        *,
        decided_by: str = "",
        reason: str = "",
    ) -> ToolResult:
        """Run a parked call once: decide it first if pending (`decision` and `decided_by`
        are then required), else apply the decision already in the store.

        Raises:
            ToolUsageError: no approval store, a flow-node or HTTP-server approval, a pending
                record without `decision` or `decided_by`, or `ctx` of another run.
            ApprovalNotFound: unknown or expired id (an expiry also sends `approval.expired`).
            ApprovalAlreadyDecided: already used, or decided otherwise in the store.
        """
        try:
            call, run_id = await resume_target(
                self.approval, approval_id, decision, decided_by=decided_by, reason=reason, ctx=ctx
            )
        except ApprovalNotFound as exc:
            if exc.expired:  # a store TTL ran out: nobody decided in time
                expired = make_event(
                    EventType.APPROVAL_EXPIRED,
                    ctx,
                    tool=exc.tool,
                    approval_id=approval_id,
                    outcome="ttl",
                )
                emit(ctx.sink if ctx is not None and ctx.sink is not None else self.sink, expired)
            raise
        owned = ctx is None
        ctx = resume_context(ctx, run_id, lambda: self.context(run_id=run_id))
        try:
            return await self.run_call(call, ctx)
        finally:
            if owned:  # the parked run ends here, so its guard state is not left behind
                self.end_run(run_id)

    def _run_context(self, ctx: ToolContext | None) -> tuple[ToolContext, bool]:
        """The caller's context, or a fresh run; True when this call owns (and ends) the run."""
        if ctx is None:
            return self.context(), True
        if not ctx.run_id:
            return dataclasses.replace(ctx, run_id=new_id("run_")), True
        return ctx, False


def _limits(policy: ToolPolicy | None, limits: RunLimits) -> RunLimits:
    """`limits` under `policy`; a non-RunLimits result fails closed.

    Raises:
        ToolUsageError: the policy returned something other than RunLimits.
    """
    if policy is None:
        return limits
    found = policy.limits(limits)
    if not isinstance(found, RunLimits):
        raise ToolUsageError(f"policy.limits must return RunLimits, got {type(found).__name__}")
    return found


def bind_model(executor: Executor, model: LLMPort, ctx: ToolContext) -> InstrumentedModel:
    """`executor.model(model)` bound to `ctx`; an `InstrumentedModel` is bound, never re-wrapped."""
    llm = model if isinstance(model, InstrumentedModel) else executor.model(model)
    return llm.bind(ctx)
