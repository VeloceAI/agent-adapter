"""What a SubFlow node visit and its child run hand each other: the host, pauses, denials."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.errors import ApprovalPending, FlowError
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.exec._resume import Pending
from agent_adapter.flow._result import FlowCheckpoint

SubflowRun = Callable[..., Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class SubflowHost:
    """How SubFlow nodes run their child graphs (the local runner sets it)."""

    run: SubflowRun  # (env, node, state, ctx, checkpoint or None) -> the parent update
    port: ApprovalPort | None  # asked by the child's pause gates
    max_steps: int
    concurrency: int | None


class SubFlowPaused(ApprovalPending):
    """A sub-flow run paused: its unsealed checkpoint and the pending records it keeps."""

    def __init__(self, checkpoint: FlowCheckpoint, pendings: tuple[Pending, ...]) -> None:
        request = _request(checkpoint)
        super().__init__(request.approval_id, tool=request.tool, run_id=request.run_id)
        self.checkpoint = checkpoint
        self.pendings = pendings


class SubFlowDenied(FlowError):
    """A pause gate inside a sub-flow was denied; `at` is the path below the SubFlow node."""

    def __init__(self, flow: str, at: str) -> None:
        super().__init__(f"flow {flow!r}: a pause gate was denied at {at!r}", flow=flow)
        self.at = at


def child_request(checkpoint: FlowCheckpoint, ids: Mapping[str, Any]) -> ApprovalRequest:
    """The child's pending request, labelled with the SubFlow node (its digest is unchanged)."""
    request = _request(checkpoint)
    metadata = {**ids, "subflow": dict(request.metadata)}
    return request.model_copy(update={"metadata": metadata})


def _request(checkpoint: FlowCheckpoint) -> ApprovalRequest:
    if checkpoint.pending is None:
        raise FlowError(f"flow {checkpoint.flow!r}: a paused sub-flow holds no request")
    return checkpoint.pending
