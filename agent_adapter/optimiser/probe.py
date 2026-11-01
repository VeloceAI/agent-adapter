"""Probe: run one harness over every case, judge each run, and contrast it with a reference."""

from __future__ import annotations

import asyncio
import difflib
import functools
import logging
import math
from collections.abc import Sequence
from typing import cast

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import log_failure
from agent_adapter.core.ports import LedgerPort, LLMPort
from agent_adapter.core.types import LedgerEntry
from agent_adapter.exec.executor import Executor
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser.evidence import clip
from agent_adapter.optimiser.hosts import Host, failed_run
from agent_adapter.optimiser.model import (
    Candidate,
    CaseRun,
    Contrast,
    EvalCase,
    HarnessTree,
    ProbeReport,
    Score,
)
from agent_adapter.optimiser.surface import parse_path

logger = logging.getLogger(__name__)

HOST_COST_ENTRY = "host-reported-cost"  # ledger entry id of a host's own spend for a case run


def case_run_id(run_prefix: str, label: str, case_id: str) -> str:
    """The run id of one case under one harness (also the ledger run id)."""
    return f"{run_prefix}/{label}/{case_id}"


async def probe(
    host: Host,
    tree: HarnessTree,
    label: str,
    *,
    model: InstrumentedModel,
    run_prefix: str,
    ledger: LedgerPort,
    concurrency: int,
    cases: Sequence[EvalCase] | None = None,
    case_timeout_s: float | None = None,
) -> ProbeReport:
    """Run `tree` over every case once (never raises for a case); results keep case order.

    A case (or its judging) running past `case_timeout_s` is cancelled and counts as failed.
    """
    chosen = tuple(host.cases() if cases is None else cases)
    work = [
        functools.partial(
            _run_case, host, tree, label, case, model, run_prefix, ledger, case_timeout_s
        )
        for case in chosen
    ]
    runs = await Executor().parallel(*work, concurrency=max(concurrency, 1))
    return summarise(label, runs)


def summarise(label: str, runs: Sequence[CaseRun]) -> ProbeReport:
    """Totals of a list of case runs."""
    graded = [r.score for r in runs if r.score is not None]
    return ProbeReport(
        label=label,
        runs=list(runs),
        mean_score=sum(graded) / len(graded) if graded else None,
        n_passed=sum(1 for r in runs if r.passed),
        n_graded=len(graded),
        n_delivered=sum(1 for r in runs if r.delivered),
        cost_usd=sum(r.metrics.get("cost_usd", 0.0) for r in runs),
    )


async def _run_case(
    host: Host,
    tree: HarnessTree,
    label: str,
    case: EvalCase,
    model: InstrumentedModel,
    run_prefix: str,
    ledger: LedgerPort,
    timeout_s: float | None,
) -> CaseRun:
    run_id = case_run_id(run_prefix, label, case.id)
    llm = cast("LLMPort", model.bind(ToolContext(run_id=run_id)))
    deadline = asyncio.timeout(timeout_s)
    try:
        async with deadline:
            run = await host.run(tree, case, run_id=run_id, model=llm)
    except Exception as exc:  # a timeout, or the host broke its never-raise contract
        if is_control_flow(exc):
            raise
        run = _timed_out(case, timeout_s) if deadline.expired() else _raised(case, label, exc)
    metrics = {**run.metrics, "cost_usd": _case_cost(ledger, run_id, run)}
    run = run.model_copy(update={"case_id": case.id, "label": label, "metrics": metrics})
    score = await _judge(host, case, run, timeout_s)
    if score is None:
        return run
    return run.model_copy(
        update={"score": score.value, "passed": score.passed, "judge_notes": score.notes}
    )


def _case_cost(ledger: LedgerPort, run_id: str, run: CaseRun) -> float:
    """The run's spend: the host's `metrics["cost_usd"]` (its total) or the ledger's, if more.

    The part the ledger did not see is recorded, so `max_usd` and the budget hook count it.
    """
    seen = ledger.total_usd(run_id)
    reported = run.metrics.get("cost_usd", 0.0)
    if reported == 0:
        return seen
    if not math.isfinite(reported) or reported < 0:
        logger.warning("run %s: host reported cost_usd=%r; ignored", run_id, reported)
        return seen
    if reported <= seen:
        return seen
    ledger.record(
        LedgerEntry(
            run_id=run_id,
            call_id=HOST_COST_ENTRY,
            kind="api",
            stage="host",
            provider="host",
            model="reported",
            cost_usd=reported - seen,
        )
    )
    return reported


def _timed_out(case: EvalCase, timeout_s: float | None) -> CaseRun:
    logger.warning("case %s timed out after %s s", case.id, timeout_s)
    error = f"case-timeout: case {case.id} took longer than {timeout_s:g} s"
    return CaseRun(
        case_id=case.id,
        label="",
        ok=False,
        delivered=False,
        output="",
        transcript=f"[ERROR] {error}",
        error=error,
    )


def _raised(case: EvalCase, label: str, exc: Exception) -> CaseRun:
    log_failure(
        logger, logging.WARNING, exc, "host.run raised for case %s under %s", case.id, label
    )
    return failed_run(case, exc)


async def _judge(
    host: Host, case: EvalCase, run: CaseRun, timeout_s: float | None = None
) -> Score | None:
    """The host judge's score; a failing judge is `None` (excluded from means), never a zero."""
    if host.judge is None:
        return None
    try:
        async with asyncio.timeout(timeout_s):
            score = await host.judge(case, run)
    except Exception as exc:
        if is_control_flow(exc):
            raise
        log_failure(logger, logging.WARNING, exc, "judge failed on case %s", case.id)
        return None
    if score is not None and not isinstance(score, Score):
        logger.warning("judge returned %s for case %s, not a Score", type(score).__name__, case.id)
        return None
    return score


def contrast(
    ref: ProbeReport, cand: ProbeReport, candidate: Candidate, *, diff_chars: int = 2_000
) -> list[Contrast]:
    """Per case in both reports: deltas, delivery pair, exercise flag and transcript diff."""
    by_case = {r.case_id: r for r in cand.runs}
    rows: list[Contrast] = []
    for old in ref.runs:
        new = by_case.get(old.case_id)
        if new is None:
            continue
        both = old.score is not None and new.score is not None
        diff = "".join(
            difflib.unified_diff(
                old.transcript.splitlines(keepends=True),
                new.transcript.splitlines(keepends=True),
                fromfile=ref.label,
                tofile=cand.label,
                n=1,
            )
        )
        rows.append(
            Contrast(
                case_id=old.case_id,
                ref_label=ref.label,
                cand_label=cand.label,
                score_delta=(new.score or 0.0) - (old.score or 0.0) if both else None,
                delivered=(old.delivered, new.delivered),
                metric_deltas={
                    k: new.metrics[k] - old.metrics[k]
                    for k in sorted(set(old.metrics) & set(new.metrics))
                    if k != "cost_usd" and new.metrics[k] != old.metrics[k]
                },
                edit_exercised=_exercised(candidate, old, new),
                transcript_diff=clip(diff, diff_chars),
            )
        )
    return rows


def _exercised(candidate: Candidate, old: CaseRun, new: CaseRun) -> bool | None:
    """True if any edit visibly acted; False if every edit is observable and none did."""
    verdicts = [_edit_exercised(edit.path, old, new) for edit in candidate.edits]
    if any(v is True for v in verdicts):
        return True
    if verdicts and all(v is False for v in verdicts):
        return False
    return None


def _edit_exercised(path: str, old: CaseRun, new: CaseRun) -> bool | None:
    parsed = parse_path(path)
    if parsed.kind == "tool":
        key = f"tool_calls.{parsed.name}"
        if not ({key, "tool_calls"} & (set(old.metrics) | set(new.metrics))):
            return None  # the host does not report tool use
    elif parsed.kind == "hook":
        key = f"hook_fired.{parsed.name}"
        if key not in old.metrics and key not in new.metrics:
            return None
    else:
        return None  # prompt text has no observable trigger
    before, after = old.metrics.get(key, 0.0), new.metrics.get(key, 0.0)
    return after > 0 or before != after
