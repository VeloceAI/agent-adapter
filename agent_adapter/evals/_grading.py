"""Grader plumbing: the grader type, the never-raise guard, verdicts and expected values."""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import log_failure
from agent_adapter.evals.model import EvalCase, Score

logger = logging.getLogger(__name__)

# The run is an `EvalRun`, or anything with an `output` (the optimiser passes its `CaseRun`).
Grader = Callable[[EvalCase, Any], Awaitable[Score | None]]


def expected_values(case: EvalCase) -> list[str] | None:
    """The accepted answers as text (one value or a list); None when the case has none."""
    value = case.expected
    if value is None:
        logger.warning("case %s has no expected value; not graded", case.id)
        return None
    values = value if isinstance(value, list | tuple) else [value]
    return [str(v) for v in values]


def verdict(hit: bool, miss_note: str) -> Score:
    """1.0 and passed on a hit, else 0.0 with `miss_note`."""
    return Score(value=1.0 if hit else 0.0, passed=hit, notes="" if hit else miss_note)


def guarded(name: str, grader: Grader) -> Grader:
    """`grader` with any exception turned into a logged `None`, named `name`."""

    @functools.wraps(grader)
    async def guard(case: EvalCase, run: Any) -> Score | None:
        try:
            return await grader(case, run)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            log_failure(logger, logging.WARNING, exc, "%s grader failed on %s", name, case.id)
            return None

    guard.__name__ = guard.__qualname__ = name
    guard._unguarded = grader  # type: ignore[attr-defined]  # all_of tells errors from None
    return guard


def grader_name(grader: object) -> str:
    """A grader's display name: its `name`, else its `__name__`, else its class name."""
    name = getattr(grader, "name", None) or getattr(grader, "__name__", None)
    return str(name) if name else type(grader).__name__
