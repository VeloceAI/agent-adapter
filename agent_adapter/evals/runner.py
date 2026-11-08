"""`run_eval`: run every case through an executor, grade it, and report."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from agent_adapter.core.callables import is_async
from agent_adapter.core.control import control_flow_member, is_control_flow
from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ErrorInfo, Event
from agent_adapter.cost.ledger import Ledger
from agent_adapter.evals._collect import Collector, case_cost, tools_of
from agent_adapter.evals._grading import Grader, grader_name
from agent_adapter.evals._text import as_text, clip, safe_text
from agent_adapter.evals.cases import load_cases
from agent_adapter.evals.graders import Check, code_check
from agent_adapter.evals.model import EvalCase, EvalRun, Score
from agent_adapter.evals.report import CaseResult, EvalReport, GradeResult
from agent_adapter.evals.store import save_report
from agent_adapter.evals.targets import Outcome, Target, as_target, is_observed
from agent_adapter.exec.executor import Executor
from agent_adapter.runtime.current import use_context
from agent_adapter.runtime.ids import new_id
from agent_adapter.runtime.sync import run_sync
from agent_adapter.tracing.events import EventType, emit, make_event
from agent_adapter.tracing.sinks import FanoutSink

logger = logging.getLogger(__name__)

CASE_EVENT = "custom.eval.case"  # one per case run: eval name, case id, verdict and score
_OUTPUT_CHARS = 4_000  # of each output kept in the report
Graders = Sequence[Grader] | Mapping[str, Grader]
Cases = Sequence[EvalCase] | str | os.PathLike[str]


async def run_eval(
    target: object,
    cases: Cases,
    graders: Graders,
    *,
    concurrency: int = 4,
    executor: Executor | None = None,
    name: str = "",
    timeout_s: float | None = 300.0,
    pass_at: float = 1.0,
    save_to: str | os.PathLike[str] | None = None,
    capture_content: bool = True,
) -> EvalReport:
    """Run each case on `target`, grade it with every grader, and return the report.

    Each case is its own run on `executor` (default: `eval_executor()`),
    so its hooks, approvals, budgets, ledger and sinks apply, and a recorder attached to it
    records the run. A grader without its own verdict passes at `score >= pass_at`; a grader
    that errors or does not grade fails the case and scores 0. A failed grade adds at most
    `pass_at` to the case score. `save_to` also writes the report into that folder
    (`DEFAULT_EVALS_FOLDER` is the one the viewer's Evals view reads). With
    `capture_content=False` the report keeps no input, expected value, output or judge notes.

    Raises:
        ConfigError: no cases, duplicate case ids, no graders, or a bad option.
        OSError: `save_to` cannot be written.
    """
    checked = (*_settings(concurrency, timeout_s, pass_at), bool(capture_content))
    if executor is not None and not isinstance(executor, Executor):
        raise ConfigError(f"executor must be an Executor, got {type(executor).__name__}")
    if save_to is not None and not isinstance(save_to, str | os.PathLike):
        raise ConfigError(f"save_to must be a folder path, got {type(save_to).__name__}")
    the_target = as_target(target)
    the_cases = _cases(cases)
    named = _named(graders)
    base = executor if executor is not None else eval_executor()
    collector = Collector()
    runner = base.replace(sink=FanoutSink(base.sink, collector))
    title = name or the_target.name
    logger.info("eval %s: %d cases, %d graders", title, len(the_cases), len(named))
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    gate = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> CaseResult:
        async with gate:
            return await _case(the_target, case, runner, collector, named, title, checked)

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(case)) for case in the_cases]
    except BaseExceptionGroup as failed:  # only control flow escapes a case
        raise control_flow_member(failed) or failed.exceptions[0] from None
    report = EvalReport(
        name=title,
        started_at=started_at,
        wall_s=time.perf_counter() - started,
        cases=tuple(task.result() for task in tasks),
    )
    logger.info(
        "eval %s finished: %d/%d passed, %d errors",
        title,
        report.n_passed,
        report.n_cases,
        report.n_errors,
    )
    if save_to is not None:
        save_report(report, save_to)
    return report


def eval_executor() -> Executor:
    """The default eval executor: an empty registry (no stray tools) and a fresh ledger."""
    return Executor(Registry(), ledger=Ledger())


def run_eval_sync(target: object, cases: Cases, graders: Graders, **options: Any) -> EvalReport:
    """`run_eval` for sync code (scripts, tests).

    Raises:
        ConfigError: as `run_eval`.
    """
    return run_sync(run_eval(target, cases, graders, **options))


def assert_eval(
    target: object,
    case: EvalCase,
    graders: Sequence[Grader],
    *,
    executor: Executor | None = None,
    timeout_s: float | None = 300.0,
    pass_at: float = 1.0,
) -> CaseResult:
    """Run one case (in a test, say); the result, or AssertionError with the reasons.

    Raises:
        AssertionError: the case errored or did not pass.
        ConfigError: a bad target, grader or option.
    """
    report = run_eval_sync(
        target,
        [case],
        list(graders),
        concurrency=1,
        executor=executor,
        timeout_s=timeout_s,
        pass_at=pass_at,
    )
    result = report.cases[0]
    if not result.passed:
        reasons = result.error or "; ".join(
            f"{g.grader}: {g.notes or 'failed'}" for g in result.grades if not g.passed
        )
        raise AssertionError(f"eval case {case.id} failed (run {result.run_id}): {reasons}")
    return result


async def _case(
    target: Target,
    case: EvalCase,
    executor: Executor,
    collector: Collector,
    graders: tuple[tuple[str, Grader], ...],
    title: str,
    settings: tuple[float | None, float, bool],
) -> CaseResult:
    timeout_s, pass_at, keep = settings
    ctx = executor.context(run_id=new_id("run_"))
    collector.open(ctx.run_id)
    emit(ctx.sink, make_event(EventType.RUN_STARTED, ctx))
    error = "interrupted"  # until the attempt returns: control flow leaves the run failed
    try:
        started = time.perf_counter()
        output, error = await _attempt(target, case, executor, ctx, timeout_s)
        latency = time.perf_counter() - started
        events = collector.close(ctx.run_id)
        tools, cost = _measured(target, executor, ctx.run_id, events)
        run = EvalRun(
            case_id=case.id,
            output=output,
            run_id=ctx.run_id,
            events=events,
            tool_calls=tools,
            cost_usd=cost,
            latency_s=latency,
        )
        grades = () if error else await _grades(graders, case, run, pass_at, keep)
        score = _score(grades, pass_at) if grades else None
        passed = not error and bool(grades) and all(g.passed for g in grades)
        emit(
            ctx.sink,
            make_event(CASE_EVENT, ctx, eval=title, case_id=case.id, passed=passed, score=score),
        )
    finally:
        collector.close(ctx.run_id)
        emit(ctx.sink, make_event(EventType.RUN_FINISHED, ctx, ok=not error))
        executor.end_run(ctx.run_id)
    return CaseResult(
        case_id=safe_text(case.id),
        tags=tuple(safe_text(tag) for tag in case.tags),
        run_id=ctx.run_id,
        passed=passed,
        score=score,
        grades=grades,
        input=_kept(case.input, keep),
        expected=_kept(case.expected, keep),
        output=_kept(output, keep),
        error=safe_text(error),
        cost_usd=cost or 0.0,
        latency_s=latency,
        tool_calls=tuple(safe_text(t) for t in tools or ()),
        measured=tools is not None and cost is not None,
    )


def _kept(value: Any, keep: bool) -> str:
    """The clipped, UTF-8-safe text of `value`; "" for None or when content is not kept."""
    if value is None or not keep:
        return ""
    return safe_text(clip(as_text(value), _OUTPUT_CHARS))


def _score(grades: tuple[GradeResult, ...], pass_at: float) -> float:
    """Mean over every grade: ungraded counts 0, a failed grade at most `pass_at`."""
    values = [
        0.0 if g.score is None else g.score if g.passed else min(g.score, pass_at) for g in grades
    ]
    return sum(values) / len(values)


def _measured(
    target: Target, executor: Executor, run_id: str, events: tuple[Event, ...]
) -> tuple[tuple[str, ...] | None, float | None]:
    """(tools called, cost) of the run; (None, None) when it was not observed or not measurable.

    A target that may work outside the case's run counts as observed only when the run saw
    its events (any beyond `run.*`) or the ledger charged the run.
    """
    try:
        tools = tools_of(events)
        cost = case_cost(executor.ledger, run_id, events)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        log_failure(logger, logging.WARNING, exc, "eval run %s: cannot measure it", run_id)
        return None, None
    seen = any(not e.type.startswith("run.") for e in events) or cost > 0
    return (tools, cost) if is_observed(target) or seen else (None, None)


async def _attempt(
    target: Target, case: EvalCase, executor: Executor, ctx: Any, timeout_s: float | None
) -> tuple[Any, str]:
    """(output, "") on success; (partial output or None, "<kind>: <message>") otherwise."""
    try:
        with use_context(ctx):
            async with asyncio.timeout(timeout_s):
                raw = await target.run_case(case, executor=executor, ctx=ctx)
    except TimeoutError:
        logger.warning("eval case %s timed out after %ss", case.id, timeout_s)
        return None, f"timeout: the case ran longer than {timeout_s}s"
    except Exception as exc:
        if is_control_flow(exc):
            raise
        info = ErrorInfo.from_exception(exc)
        log_failure(logger, logging.WARNING, exc, "eval case %s failed", case.id)
        return None, f"{info.kind}: {info.message}"
    if isinstance(raw, Outcome):
        if raw.unfinished:
            logger.warning("eval case %s did not finish", case.id)
            return raw.output, f"unfinished: {raw.unfinished}"
        return raw.output, ""
    return raw, ""


async def _grades(
    graders: tuple[tuple[str, Grader], ...],
    case: EvalCase,
    run: EvalRun,
    pass_at: float,
    keep: bool,
) -> tuple[GradeResult, ...]:
    results = []
    for name, grader in graders:
        try:
            score = await grader(case, run)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            log_failure(logger, logging.WARNING, exc, "grader %s failed on %s", name, case.id)
            score = None
        if score is not None and not isinstance(score, Score):
            logger.warning("grader %s returned %s on %s", name, type(score).__name__, case.id)
            score = None
        label = safe_text(name)
        if score is None:
            results.append(GradeResult(grader=label, score=None, passed=False, notes="not graded"))
            continue
        passed = score.passed if score.passed is not None else score.value >= pass_at
        quotes = getattr(grader, "quotes_output", False) and not keep
        notes = "" if quotes else safe_text(score.notes)
        results.append(GradeResult(grader=label, score=score.value, passed=passed, notes=notes))
    return tuple(results)


def _settings(
    concurrency: int, timeout_s: float | None, pass_at: float
) -> tuple[float | None, float]:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigError(f"concurrency must be an int >= 1, got {concurrency!r}")
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, int | float)
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ConfigError(f"timeout_s must be a number > 0 or None, got {timeout_s!r}")
    if isinstance(pass_at, bool) or not isinstance(pass_at, int | float) or not 0 <= pass_at <= 1:
        raise ConfigError(f"pass_at must be in [0, 1], got {pass_at!r}")
    return timeout_s, float(pass_at)


def _cases(cases: Cases) -> tuple[EvalCase, ...]:
    if isinstance(cases, str | os.PathLike):
        return load_cases(cases)
    found = tuple(cases)
    if not found:
        raise ConfigError("run_eval needs at least one case")
    bad = [type(c).__name__ for c in found if not isinstance(c, EvalCase)]
    if bad:
        raise ConfigError(f"cases must be EvalCase objects, got {sorted(set(bad))}")
    duplicates = sorted(i for i, n in Counter(case.id for case in found).items() if n > 1)
    if duplicates:
        raise ConfigError(f"duplicate case ids: {duplicates}")
    return found


def _named(graders: Graders) -> tuple[tuple[str, Grader], ...]:
    pairs = (
        [(str(k), g) for k, g in graders.items()]
        if isinstance(graders, Mapping)
        else [(grader_name(g), g) for g in graders]
    )
    if not pairs:
        raise ConfigError("run_eval needs at least one grader")
    bad = [name for name, grader in pairs if not callable(grader)]
    if bad:
        raise ConfigError(f"graders must be callable: {bad}")
    seen: dict[str, int] = {}
    named = []
    for name, grader in pairs:
        seen[name] = seen.get(name, 0) + 1
        usable = grader if is_async(grader) else code_check(cast("Check", grader))  # sync check
        named.append((name if seen[name] == 1 else f"{name}#{seen[name]}", usable))
    return tuple(named)
