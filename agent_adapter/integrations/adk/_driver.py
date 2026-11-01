"""The driver of a compiled flow: the local runner's superstep loop over ADK child nodes.

Every step with an effect (a gate, a node visit, an answer, a withdrawal) is a child node
with a deterministic run id (`<node>@v<visit>…`). When ADK re-runs the driver on resume it
fast-forwards finished children from the session instead of running them again, so the
driver only replays pure bookkeeping: merges, routes from stored labels, loop counters.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from agent_adapter.core.errors import FlowError, FlowLoopExceeded
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec.integration import AnswerEntry
from agent_adapter.flow.graph import NodeSpec
from agent_adapter.flow.visit import (
    apply_edits,
    graph_digest,
    merge_outcomes,
    next_frontier,
    pick_targets,
    resume_entry,
)
from agent_adapter.integrations.adk._steps import (
    Plan,
    Visit,
    answer_step,
    decoded,
    gate_step,
    read_visit,
    visit_step,
    withdraw_step,
)
from agent_adapter.integrations.adk._support import INTERRUPT_PREFIX, TARGET
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

Work = Callable[[Any], Awaitable[dict[str, Any]]]  # a child's work, given the child's ADK ctx
VERSION_STEP = "aa_graph"  # the child holding the graph digest the session started with


@dataclass(frozen=True)
class Pause:
    pending: ApprovalRequest
    interrupt_id: str


@dataclass(frozen=True)
class End:
    status: str  # "done" or "denied"
    state: BaseModel
    visits: Mapping[str, int]
    denied_at: str | None = None


@dataclass
class Cursor:
    state: BaseModel
    frontier: tuple[str, ...]
    visits: dict[str, int] = field(default_factory=dict)
    loops: dict[str, int] = field(default_factory=dict)
    steps: int = 0


class Turn:
    """One driver run (the first turn or a resume): replays finished children, runs the rest."""

    def __init__(self, plan: Plan, adk: Any, step_class: Any) -> None:
        session = adk.session
        self.plan = plan
        self.adk = adk
        self._step_class = step_class  # the FlowStep node class
        self.ctx = plan.env.executor.context(run_id=adk.invocation_id, thread_id=session.id)
        self.inputs: dict[str, Any] = dict(adk.resume_inputs)
        self._prefix = f"{INTERRUPT_PREFIX}{session.id}:{adk.node_path}/"

    def interrupt_id(self, node: str, run_id: str) -> str:
        return f"{self._prefix}{node}@{run_id}"

    async def child(self, node: str, run_id: str, work: Work) -> dict[str, Any]:
        """The output of child `<node>@<run_id>`: run now, or replayed from the session."""
        from google.adk.workflow._errors import DynamicNodeFailError

        step = self._step_class(name=node)
        step._work = work
        try:
            output = await self.adk.run_node(step, None, run_id=run_id)
        except DynamicNodeFailError as exc:
            error = exc.error  # the step's own error, not ADK's wrapper
            raise error from error.__cause__
        if not isinstance(output, dict):
            raise FlowError(
                f"flow {self.plan.flow!r} node {node!r}: step {run_id} has no output",
                flow=self.plan.flow,
                node=node,
            )
        return output

    async def run(self, cursor: Cursor) -> Pause | End:
        graph = self.plan.graph
        await self._check_version()
        while cursor.frontier:
            if cursor.steps >= self.plan.max_steps:
                logger.warning(
                    "flow %s stopped after max_steps=%d", graph.name, self.plan.max_steps
                )
                raise FlowLoopExceeded(graph.name, cursor.frontier[0], self.plan.max_steps)
            nodes = graph.nodes
            ready = [n for n in cursor.frontier if nodes[n].kind != "join"] or list(cursor.frontier)
            decisions: dict[str, ApprovalDecision] = {}
            for name in ready:
                if nodes[name].pause is False:
                    continue
                gate = await self.gate(nodes[name], cursor.state, cursor.visits.get(name, 0) + 1)
                if isinstance(gate, Pause):
                    self._gate_paused(gate.pending)
                    return gate
                if not gate.approved:
                    logger.info("flow %s node %s denied: %s", graph.name, name, gate.reason)
                    return End("denied", cursor.state, dict(cursor.visits), denied_at=name)
                decisions[name] = gate
            state = apply_edits(graph, cursor.state, ready, decisions)
            visits = dict(cursor.visits)
            for name in ready:
                visits[name] = visits.get(name, 0) + 1
            ran = await self.superstep([nodes[n] for n in ready], state, visits, decisions)
            if isinstance(ran, Pause):
                return ran
            frontier, loops = self._next(cursor, ready, ran)
            outcomes = [visit.outcome() for visit in ran.values()]
            cursor.state = merge_outcomes(graph, state, outcomes, cursor.steps + 1)
            cursor.frontier, cursor.visits, cursor.loops = frontier, visits, loops
            cursor.steps += 1
        return End("done", cursor.state, dict(cursor.visits))

    async def _check_version(self) -> None:
        """Refuse to continue a session a different version of this flow started.

        Raises:
            FlowError: the stored graph digest differs (replayed outputs would not fit).
        """
        digest = graph_digest(self.plan.graph)

        async def record(adk: Any) -> dict[str, Any]:
            return {"digest": digest}

        stored = await self.child(VERSION_STEP, "digest", record)
        if stored.get("digest") != digest:
            flow = self.plan.flow
            logger.warning("flow %s: the session holds a run of another graph", flow)
            raise FlowError(
                f"flow {flow!r}: this session was started by another version of this flow "
                "(its nodes, edges or gates differ); resume refused",
                flow=flow,
            )

    async def gate(self, node: NodeSpec, state: BaseModel, visit: int) -> ApprovalDecision | Pause:
        """A pause gate's decision; each answered interrupt adds a step `v<visit>g<k>`."""
        answered: tuple[ApprovalRequest, Any] | None = None
        attempt = 1
        while True:
            run_id = f"v{visit}g{attempt}"

            async def work(adk: Any, answered: Any = answered) -> dict[str, Any]:
                return await gate_step(self.plan, node, state, self.ctx, visit, answered)

            out = await self.child(node.name, run_id, work)
            if "decision" in out:
                decision: ApprovalDecision = decoded(self.plan, node.name, out["decision"])
                return decision
            pending: ApprovalRequest = decoded(self.plan, node.name, out["pending"])
            interrupt = self.interrupt_id(node.name, run_id)
            if interrupt not in self.inputs:
                return Pause(_with_interrupt(pending, interrupt), interrupt)
            answered = (pending, self.inputs[interrupt])
            attempt += 1

    async def superstep(
        self,
        nodes: Sequence[NodeSpec],
        state: BaseModel,
        visits: Mapping[str, int],
        decisions: Mapping[str, ApprovalDecision],
    ) -> dict[str, Visit] | Pause:
        """Every ready node's outcome; a waiting node re-runs after each answered interrupt."""
        specs = {node.name: node for node in nodes}
        runs = [
            self._visit(node, f"v{visits[node.name]}", state, visits, decisions) for node in nodes
        ]
        results = {visit.node: visit for visit in await self._all(runs)}
        entries: dict[str, tuple[AnswerEntry, ...]] = {}
        resumes = 0
        while True:
            waiting = sorted((v for v in results.values() if v.pending), key=lambda v: v.node)
            if not waiting:
                return results
            resumes += 1
            first = waiting[0]
            answered = await self._answer(first, waiting, f"v{visits[first.node]}", resumes)
            if isinstance(answered, Pause):
                return answered
            if answered is not None:
                entries[first.node] = (*entries.get(first.node, ()), answered)
            reruns = [
                self._visit(
                    specs[v.node],
                    f"v{visits[v.node]}r{resumes}",
                    state,
                    visits,
                    decisions,
                    entries.get(v.node, ()),
                    v,
                )
                for v in waiting
            ]
            results.update({visit.node: visit for visit in await self._all(reruns)})

    async def _answer(
        self, first: Visit, waiting: Sequence[Visit], visit: str, resumes: int
    ) -> Pause | AnswerEntry | None:
        """Pause on `first`'s request, or the answer its resume input carries (None: refused)."""
        pending = cast(ApprovalRequest, first.pending)
        interrupt = self.interrupt_id(first.node, first.run_id)

        async def withdrawn(adk: Any) -> dict[str, Any]:
            return await withdraw_step(self.plan, self.ctx, waiting, pending)

        # every pause withdraws what it supersedes; a replay calls the same children in order
        await self.child(first.node, f"{visit}w{resumes}", withdrawn)
        if interrupt not in self.inputs:
            return Pause(_with_interrupt(pending, interrupt), interrupt)
        response = self.inputs[interrupt]

        async def answer(adk: Any) -> dict[str, Any]:
            return await answer_step(self.plan, first.node, pending, response)

        out = await self.child(first.node, f"{visit}a{resumes}", answer)
        if "decision" not in out:
            return None
        return resume_entry(pending, decoded(self.plan, first.node, out["decision"]))

    def _visit(
        self,
        node: NodeSpec,
        run_id: str,
        state: BaseModel,
        visits: Mapping[str, int],
        decisions: Mapping[str, ApprovalDecision],
        entries: Sequence[AnswerEntry] = (),
        previous: Visit | None = None,
    ) -> Callable[[], Awaitable[Visit]]:
        async def work(adk: Any) -> dict[str, Any]:
            ctx = dataclasses.replace(self.ctx, extras={**self.ctx.extras, TARGET: adk})
            return await visit_step(
                self.plan,
                node,
                state,
                ctx,
                visit=visits[node.name],
                approval=decisions.get(node.name),
                entries=entries,
                previous=previous,
            )

        async def visit() -> Visit:
            return read_visit(
                self.plan, node.name, run_id, await self.child(node.name, run_id, work)
            )

        return visit

    async def _all(self, works: Sequence[Callable[[], Awaitable[Visit]]]) -> list[Visit]:
        if len(works) <= 1:
            return [await work() for work in works]
        executor = self.plan.env.executor
        return await executor.parallel(*works, concurrency=len(works))

    def _next(
        self, cursor: Cursor, ready: Sequence[str], ran: Mapping[str, Visit]
    ) -> tuple[tuple[str, ...], dict[str, int]]:
        """The next frontier (declaration order) and loop counters, from the stored labels."""
        graph = self.plan.graph
        loops = dict(cursor.loops)
        routed: dict[str, list[str]] = {}
        for name in ready:
            route = graph.routes.get(name)
            if route is not None:
                labels = ran[name].labels or ()
                routed[name] = pick_targets(graph.name, route, labels, cursor.visits, loops)
        return next_frontier(graph, cursor.frontier, ready, routed), loops

    def _gate_paused(self, pending: ApprovalRequest) -> None:
        node = str(pending.metadata.get("node", ""))
        event = make_event(
            EventType.NODE_PAUSED,
            self.ctx,
            call_id=pending.call_id,
            flow=self.plan.flow,
            node=node,
            approval_id=pending.approval_id,
        )
        emit(self.ctx.sink, event)
        logger.info(
            "flow %s node %s waits for approval %s", self.plan.flow, node, pending.approval_id
        )


def _with_interrupt(pending: ApprovalRequest, interrupt: str) -> ApprovalRequest:
    return pending.model_copy(update={"metadata": {**pending.metadata, "interrupt_id": interrupt}})
