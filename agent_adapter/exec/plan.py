"""Plans: steps with dependencies, validated up front and run in Kahn waves."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import PlanInvalid, log_failure
from agent_adapter.core.types import ErrorInfo, ToolCall, ToolResult
from agent_adapter.exec.parallel import OnError

logger = logging.getLogger(__name__)

DEPENDENCY_FAILED = "dependency-failed"

StepArgs = Mapping[str, Any] | Callable[[Mapping[str, ToolResult]], Mapping[str, Any]]
RunWave = Callable[[list[ToolCall]], Awaitable[list[ToolResult]]]


@dataclass(frozen=True)
class Step:
    """One tool call of a plan; callable `args` receive the results of `depends_on`."""

    id: str
    tool: str
    args: StepArgs = field(default_factory=dict)
    depends_on: Sequence[str] = ()


def _plan_waves(steps: Sequence[Step]) -> list[list[Step]]:
    """Kahn layers in step order.

    Raises:
        PlanInvalid: duplicate ids, unknown dependencies or a cycle (all problems listed).
    """
    problems: list[str] = []
    ids: set[str] = set()
    for step in steps:
        if step.id in ids:
            problems.append(f"duplicate step id {step.id!r}")
        ids.add(step.id)
    for step in steps:
        for dep in step.depends_on:
            if dep not in ids:
                problems.append(f"step {step.id!r} depends on unknown step {dep!r}")
    if problems:
        raise PlanInvalid(problems)
    waves: list[list[Step]] = []
    done: set[str] = set()
    remaining = list(steps)
    while remaining:
        wave = [s for s in remaining if all(dep in done for dep in s.depends_on)]
        if not wave:
            cycle = ", ".join(repr(s.id) for s in remaining)
            raise PlanInvalid([f"dependency cycle among steps {cycle}"])
        waves.append(wave)
        done.update(s.id for s in wave)
        remaining = [s for s in remaining if s.id not in done]
    return waves


async def _run_waves(
    steps: Sequence[Step], run_wave: RunWave, *, on_error: OnError, run_id: str
) -> dict[str, ToolResult]:
    """Run each wave as one batch; a step whose dependency failed is not run.

    Step call ids are `f"{run_id}:{step.id}"`, so re-running a plan in the same run (after an
    approval was decided) resumes its steps instead of starting new calls.
    """
    results: dict[str, ToolResult] = {}
    for number, wave in enumerate(_plan_waves(steps), start=1):
        runnable: list[tuple[Step, ToolCall]] = []
        for step in wave:
            blocked = _blocked(step, results)
            if blocked is not None:
                results[step.id] = blocked
                continue
            call = _call(step, results, on_error, f"{run_id}:{step.id}")
            if isinstance(call, ToolResult):
                results[step.id] = call
            else:
                runnable.append((step, call))
        logger.debug("plan wave %d: %s", number, [step.id for step, _ in runnable])
        for (step, _), result in zip(
            runnable, await run_wave([call for _, call in runnable]), strict=True
        ):
            results[step.id] = result
    return {step.id: results[step.id] for step in steps}


def _blocked(step: Step, results: Mapping[str, ToolResult]) -> ToolResult | None:
    failed = [dep for dep in step.depends_on if not results[dep].ok]
    if not failed:
        return None
    message = f"step {step.id!r} not run: dependencies {failed} did not succeed"
    return ToolResult(
        call_id="",
        name=step.tool,
        status="error",
        error=ErrorInfo(kind=DEPENDENCY_FAILED, message=message, details={"dependencies": failed}),
    )


def _call(
    step: Step, results: Mapping[str, ToolResult], on_error: OnError, call_id: str
) -> ToolCall | ToolResult:
    """The step's ToolCall; a failing `args` callable becomes the step's error result."""
    if not callable(step.args):
        return ToolCall(name=step.tool, arguments=dict(step.args), id=call_id)
    deps = {dep: results[dep] for dep in step.depends_on}
    try:
        arguments = dict(step.args(deps))
    except Exception as exc:
        if is_control_flow(exc) or on_error == "fail_fast":
            raise
        log_failure(
            logger,
            logging.WARNING,
            exc,
            "plan step %s (tool %s): args callable failed",
            step.id,
            step.tool,
        )
        return ToolResult(
            call_id=call_id,
            name=step.tool,
            status="error",
            error=ErrorInfo.from_exception(exc),
            exception=exc,
        )
    return ToolCall(name=step.tool, arguments=arguments, id=call_id)
