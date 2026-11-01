"""The steps a policy runs when it names none."""

from __future__ import annotations

from agent_adapter.context.shrink import ShrinkToolResults
from agent_adapter.context.step import CompressionStep
from agent_adapter.context.steps import RemoveOldMessages
from agent_adapter.context.summarize import Summarize
from agent_adapter.core.ports import LLMPort


def default_steps(summary_model: LLMPort | None = None) -> tuple[CompressionStep, ...]:
    """ShrinkToolResults, RemoveOldMessages(keep_first=1, keep_last=8), then Summarize(model)."""
    steps: list[CompressionStep] = [
        ShrinkToolResults(),
        RemoveOldMessages(keep_first=1, keep_last=8),
    ]
    if summary_model is not None:
        steps.append(Summarize(model=summary_model))
    return tuple(steps)
