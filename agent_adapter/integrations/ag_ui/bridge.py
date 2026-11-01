"""`AGUIBridge`: run an agent function as an AG-UI run, with approvals as interrupts.

Safety: a resume entry never approves on its own. A `cancelled` entry,
or any payload but `{"approved": true}`, denies. An approving entry is only a request to
the bridge's `ApprovalPort`; when the port approves, the call runs with a one-use token
bound to its run, call id, tool and argument digest. An interrupt is answered once, from
its own thread, by the principal whose run raised it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from agent_adapter.core.errors import ApprovalPending, log_failure
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import to_jsonable
from agent_adapter.exec.executor import Executor
from agent_adapter.integrations._native_gate import TokenTable
from agent_adapter.integrations.ag_ui._channel import MAX_EVENTS
from agent_adapter.integrations.ag_ui._events import Paused
from agent_adapter.integrations.ag_ui._input import parse_input
from agent_adapter.integrations.ag_ui._interrupts import InterruptTable, PendingCall
from agent_adapter.integrations.ag_ui._resume import Resumer
from agent_adapter.integrations.ag_ui._support import core, encode
from agent_adapter.integrations.ag_ui.run import AGUIRun
from agent_adapter.integrations.base import model_visible_error
from agent_adapter.runtime.ids import new_id

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput

logger = logging.getLogger(__name__)

Agent = Callable[[AGUIRun], Awaitable[Any]]


class AGUIBridge:
    """Streams AG-UI runs of agent functions over one executor; holds the open interrupts."""

    def __init__(
        self,
        executor: Executor,
        *,
        port: ApprovalPort | None = None,
        interrupts: InterruptTable | None = None,
        max_events: int = MAX_EVENTS,
        trace: bool = True,
    ) -> None:
        """`port` decides resumed calls (default: the executor's own port).

        Resumed calls are looked up in `executor.registry`, the registry the executor runs
        (the default registry for an `Executor()` built without one).
        """
        self.executor = executor
        self.port = port if port is not None else executor.approval
        self.interrupts = interrupts if interrupts is not None else InterruptTable()
        self.max_events = max_events
        self.trace = trace
        self._resumer = Resumer(executor.registry, self.port, self.interrupts, TokenTable())

    async def stream(
        self,
        run_input: RunAgentInput | Mapping[str, Any] | bytes | str,
        agent: Agent,
        *,
        principal: str = "",
    ) -> AsyncIterator[Any]:
        """The run's AG-UI events, from `RUN_STARTED` to `RUN_FINISHED` or `RUN_ERROR`.

        `principal` is the authenticated caller; only they may answer this run's interrupts.
        Closing the iterator early (a client disconnect) cancels the agent.

        Raises:
            AGUIInputInvalid: `run_input` is invalid (before any event).
        """
        parsed = parse_input(run_input)
        run = AGUIRun(
            parsed,
            executor=self.executor,
            principal=principal,
            max_events=self.max_events,
            trace=self.trace,
        )
        task = asyncio.create_task(self._drive(run, agent))
        try:
            async for event in run.channel:
                yield event
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait([task])
        await task

    async def sse(
        self,
        run_input: RunAgentInput | Mapping[str, Any] | bytes | str,
        agent: Agent,
        *,
        principal: str = "",
    ) -> AsyncIterator[str]:
        """`stream` encoded as `text/event-stream` frames."""
        async for event in self.stream(run_input, agent, principal=principal):
            yield encode(event)

    async def _drive(self, run: AGUIRun, agent: Agent) -> None:
        c = core()
        try:
            run.channel.put(
                c.RunStartedEvent(
                    thread_id=run.thread_id, run_id=run.run_id, protocol_version=c.PROTOCOL_VERSION
                )
            )
            logger.info("ag_ui: run %s started (thread %s)", run.run_id, run.thread_id)
            try:
                await self._resumer.resume(run)
                result = await agent(run)
            except ApprovalPending as exc:
                if not run.events.paused():
                    logger.warning("ag_ui: run %s pending outside a tool call", run.run_id)
                    run.channel.put(c.RunErrorEvent(message="approval pending", code=exc.kind))
                    return
                result = None
            except Exception as exc:
                log_failure(logger, logging.ERROR, exc, "ag_ui: run %s failed", run.run_id)
                error = model_visible_error(exc)["error"]
                run.channel.put(c.RunErrorEvent(message=error["message"], code=error["kind"]))
                return
            self._finish(run, result)
        finally:
            for run_id in run.run_ids:
                self.executor.end_run(run_id)
            run.channel.close()

    def _finish(self, run: AGUIRun, result: Any) -> None:
        c = core()
        paused = run.events.paused()
        if not paused:
            value = to_jsonable(result) if result is not None else None
            run.channel.put(
                c.RunFinishedEvent(thread_id=run.thread_id, run_id=run.run_id, result=value)
            )
            logger.info("ag_ui: run %s finished", run.run_id)
            return
        interrupts = [self._open(run, item).interrupt() for item in paused]
        outcome = c.RunFinishedInterruptOutcome(interrupts=interrupts)
        run.channel.put(
            c.RunFinishedEvent(thread_id=run.thread_id, run_id=run.run_id, outcome=outcome)
        )
        logger.info("ag_ui: run %s interrupted (%d call(s))", run.run_id, len(paused))

    def _open(self, run: AGUIRun, paused: Paused) -> PendingCall:
        if paused.arguments is None:
            logger.warning("ag_ui: %s paused with unknown arguments; resumes deny", paused.tool)
        pending = PendingCall(
            interrupt_id=new_id("int_"),
            thread_id=run.thread_id,
            principal=run.principal,
            run_id=paused.run_id,
            call_id=paused.call_id,
            tool=paused.tool,
            arguments=paused.arguments,
            approval_id=paused.approval_id,
            expires_at=self.interrupts.expiry(),
        )
        self.interrupts.open(pending)
        return pending
