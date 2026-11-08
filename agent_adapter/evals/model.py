"""Eval data: a case, a grader's score, and the run a grader sees."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_adapter.core.types import Event

_FROZEN = ConfigDict(frozen=True)


class EvalCase(BaseModel):
    """One evaluation input; `expected` is for the graders only."""

    model_config = _FROZEN

    id: str
    input: Any
    tags: tuple[str, ...] = ()
    expected: Any = None


class Score(BaseModel):
    """A grader's verdict on one run; `value` is in [0, 1] and `notes` is the reason."""

    model_config = _FROZEN

    value: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    notes: str = ""

    @property
    def reason(self) -> str:
        """The grader's reason (the same text as `notes`)."""
        return self.notes


@dataclass(frozen=True)
class EvalRun:
    """What graders see of one case run; `tool_calls` and `cost_usd` are None when unobserved."""

    case_id: str
    output: Any
    run_id: str = ""
    events: tuple[Event, ...] = ()
    tool_calls: tuple[str, ...] | None = ()
    cost_usd: float | None = 0.0
    latency_s: float = 0.0
    metrics: Mapping[str, float] = field(default_factory=dict)
