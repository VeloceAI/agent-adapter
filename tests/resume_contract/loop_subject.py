"""The `run_tool_loop` subject: a model that plans from the history, two effects, a gated refund."""

from __future__ import annotations

from typing import Any

from agent_adapter.approval import DeferredApproval
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.errors import CheckpointNotFound
from agent_adapter.exec import Executor, recover_tool_loop, run_tool_loop
from agent_adapter.exec._durable_seal import open_checkpoint
from agent_adapter.models.callable import CallableModel
from tests.fakes.models import calls, text
from tests.resume_contract.flow_subject import APPROVE, registry
from tests.resume_contract.harness import (
    ACTIVE,
    KEY,
    FaultStore,
    OurStores,
    Outcome,
    SnapApprovals,
    World,
    enter,
)

RUN = "chat-1"
USER = (Message(role="user", content="close order o-1"),)
PLAN = (
    calls(("charge", {"order": "o-1"}), ("quote", {"order": "o-1"})),
    calls(("ship", {"order": "o-1"})),
    calls(("refund", {"order": "o-1", "amount": 5})),
)


def _history(messages: Any) -> list[dict[str, Any]]:
    return [m.model_dump(mode="json") for m in messages]


def plan(request: LLMRequest) -> LLMResponse:
    """The next reply is a function of the history: same history, same reply."""
    enter("model", _history(request.messages))
    turn = sum(1 for m in request.messages if m.role == "assistant")
    ACTIVE.world.boundary(f"inside model {turn}")
    return PLAN[turn] if turn < len(PLAN) else text("order closed")


PLANNER = CallableModel(plan, provider="scripted", model="planner")


class LoopSubject(OurStores):
    """Our `run_tool_loop` with a durable store and a deferred approval port."""

    name = "run_tool_loop"
    run_id = RUN
    effects = ("charge", "ship", "refund")

    def __init__(self, approvals: SnapApprovals | None = None) -> None:
        self.approvals = approvals or SnapApprovals()
        self.executor = Executor(registry, approval=DeferredApproval(self.approvals))

    def fresh(self) -> LoopSubject:
        return LoopSubject(self.approvals.snapshot())

    def _out(self, result: Any) -> Outcome:
        if result.finish == "pending":
            return Outcome("paused", token=result.messages)
        return Outcome("done", final=_history(result.messages))

    async def start(self, store: FaultStore) -> Outcome:
        result = await run_tool_loop(
            PLANNER, registry, USER, executor=self.executor, run_id=RUN, checkpoints=store,
            checkpoint_key=KEY,
        )  # fmt: skip
        return self._out(result)

    async def resume(self, store: FaultStore, token: Any) -> Outcome:
        for record in await self.approvals.list(status="pending"):  # the reviewer, elsewhere
            await self.approvals.decide(record.approval_id, APPROVE)
        result = await run_tool_loop(
            PLANNER, registry, token, executor=self.executor, run_id=RUN, checkpoints=store,
            checkpoint_key=KEY,
        )  # fmt: skip
        return self._out(result)

    async def recover(self, store: FaultStore, *, rerun: bool) -> Outcome:
        result = await recover_tool_loop(
            PLANNER, registry, RUN, executor=self.executor, checkpoints=store, checkpoint_key=KEY,
            rerun_in_doubt=rerun,
        )  # fmt: skip
        return self._out(result)

    async def _stored(self, store: FaultStore) -> tuple[str, list[Message]]:
        found = await store.latest(RUN)
        if found is None:
            raise CheckpointNotFound(RUN)
        body = open_checkpoint(KEY, found).body
        return found.checkpoint.status, [Message.model_validate(m) for m in body["messages"]]

    async def ended(self, store: FaultStore) -> Outcome:
        status, messages = await self._stored(store)
        if status == "paused":
            return Outcome("paused", token=tuple(messages))
        return Outcome(status, final=_history(messages))

    async def position(self, store: FaultStore) -> Any:
        status, messages = await self._stored(store)
        return status, _history(messages)

    def continued(self, position: Any, world: World) -> str | None:
        """Prefix continuation: the first model call sees the durable history as its prefix."""
        status, durable = position
        asked = [seen for node, seen in world.entered if node == "model"]
        if not asked or status not in ("running", "paused"):
            return None
        if asked[0][: len(durable)] != durable:
            return (
                f"the model was asked from {len(asked[0])} messages, not the durable {len(durable)}"
            )
        return None
