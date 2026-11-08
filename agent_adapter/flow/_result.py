"""What a flow run returns: `FlowResult`, and the `FlowCheckpoint` a paused run resumes from."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.exec._journal import JournalData
from agent_adapter.exec._resume import AnswerEntry

S = TypeVar("S", bound=BaseModel)
K = TypeVar("K")
V = TypeVar("V")


class _Frozen(Mapping[K, V]):
    """A read-only, picklable copy of a mapping."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[K, V]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Frozen, (self._data,))


def frozen(data: Mapping[K, V]) -> Mapping[K, V]:
    """A read-only copy."""
    return _Frozen(data)


FlowStatus = Literal["done", "paused", "denied"]
ResumeEntry = AnswerEntry  # (request, decision, exact call id)


def pending_repr(pending: ApprovalRequest | None) -> str:
    """A pending request without its arguments (read them from `pending.arguments`)."""
    if pending is None:
        return "None"
    return (
        f"<{pending.kind} approval {pending.approval_id!r} for {pending.tool!r}, "
        "arguments redacted>"
    )


@dataclass(frozen=True)
class FlowCheckpoint:
    """Where a paused flow stopped; pass it to `FlowRunner.resume` (JSON: `dump_checkpoint`)."""

    flow: str
    state: BaseModel
    frontier: tuple[str, ...]
    visits: Mapping[str, int]
    loops: Mapping[str, int]
    steps: int
    pending: ApprovalRequest | None
    decisions: Mapping[str, ApprovalDecision] = field(default_factory=dict)  # granted gates
    # updates of the paused superstep's finished nodes: only the paused node(s) run again
    _writes: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict, repr=False)
    _owned: bool = field(default=False, repr=False)  # the runner made the run and ends it
    # tool approvals already answered for each unfinished node of the paused superstep:
    # (request, decision, exact call id); each resume answers all of them again, once each
    _answers: Mapping[str, tuple[ResumeEntry, ...]] = field(default_factory=dict, repr=False)
    # finished Map items (index -> value) of each unfinished Map node: they do not run again
    _items: Mapping[str, Mapping[int, tuple[str, Any]]] = field(default_factory=dict, repr=False)
    # finished executor calls of each unfinished node (JSON data): they replay, not run again
    _journal: Mapping[str, JournalData] = field(default_factory=dict, repr=False)
    # where each paused SubFlow node's child run stopped (unsealed: this checkpoint's MAC covers it)
    _children: Mapping[str, FlowCheckpoint] = field(default_factory=dict, repr=False)
    _graph: str = field(default="", repr=False)  # digest of the graph it paused in
    _mac: str = field(default="", repr=False)  # HMAC-SHA256 of everything above (runner key)

    def __repr__(self) -> str:
        return (
            f"FlowCheckpoint(flow={self.flow!r}, frontier={self.frontier!r}, "
            f"steps={self.steps}, pending={pending_repr(self.pending)})"
        )


@dataclass(frozen=True)
class FlowResult(Generic[S]):
    """How a run ended: the state, the status, supersteps run and visits per node."""

    state: S
    status: FlowStatus
    steps: int
    visits: Mapping[str, int]
    checkpoint: FlowCheckpoint | None = None  # set when paused
    pending: ApprovalRequest | None = None
    denied_at: str | None = None

    def __repr__(self) -> str:
        return (
            f"FlowResult(status={self.status!r}, steps={self.steps}, "
            f"visits={dict(self.visits)!r}, state={self.state!r}, "
            f"pending={pending_repr(self.pending)}, denied_at={self.denied_at!r})"
        )
