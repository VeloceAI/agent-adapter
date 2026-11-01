"""Assemble the `OptimiseResult` (decisions, errors, usage, report) from the run state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_adapter.cost.summary import LedgerSummary
from agent_adapter.optimiser._fatal import run_failures
from agent_adapter.optimiser._state import RunState
from agent_adapter.optimiser.model import (
    BudgetUsage,
    DecisionLine,
    HarnessTree,
    OptimiserConfig,
    OptimiseResult,
    ProbeReport,
)
from agent_adapter.optimiser.probe import summarise
from agent_adapter.optimiser.report import patch_for, render_report

_QUIET_REFLECTIONS = ("", "NO_ENTRY", "nothing was proposed")


def decisions_of(state: RunState) -> list[DecisionLine]:
    """Every compose stage's decision lines, prefixed with the stage label."""
    return [
        d.model_copy(update={"evidence": f"[{label}] {d.evidence}"})
        for label, record in state.composes.items()
        for d in record.decisions
    ]


def errors_of(
    state: RunState,
    decisions: Sequence[DecisionLine],
    stopped: str,
    probes: Mapping[str, ProbeReport],
    failure: str = "",
) -> list[str]:
    """Model failures per probe, fallbacks, failed merges and reflection, a stop or failure."""
    errors = [line for report in probes.values() if (line := run_failures(report))]
    errors += [
        f"{c.label}: fell back to {c.parent_label}: {c.error}"
        for c in state.candidates.values()
        if c.fell_back
    ]
    errors += [f"{d.path}: {d.evidence}" for d in decisions if "llm merge failed" in d.evidence]
    if state.reflection is not None and state.reflection.why not in _QUIET_REFLECTIONS:
        errors.append(f"reflection: no lesson ({state.reflection.why})")
    if stopped:
        errors.append(f"stopped early: {stopped}")
    if failure:
        errors.append(f"failed: {failure}")
    return errors


def build_result(
    state: RunState,
    *,
    probes: Mapping[str, ProbeReport],
    finals: Sequence[str],
    base_tree: HarnessTree,
    summary: LedgerSummary,
    usage: BudgetUsage,
    config: OptimiserConfig,
    domain: str,
    n_cases: int,
    failure: str = "",
) -> OptimiseResult:
    """The run's result with its rendered `report.md`."""
    chosen = state.selection.chosen if state.selection else None
    decisions = decisions_of(state)
    chosen_tree = state.candidates[chosen].tree if chosen else None
    result = OptimiseResult(
        base=probes.get("base") or summarise("base", []),
        candidates=dict(state.candidates),
        probes=dict(probes),
        finals=list(finals),
        chosen=chosen,
        decisions=decisions,
        patch=patch_for(base_tree, chosen_tree),
        report_md="",
        lessons_added=list(state.reflection.lessons) if state.reflection else [],
        ledger_summary=summary,
        budget=usage,
        errors=errors_of(state, decisions, usage.stopped, probes, failure),
    )
    why = state.selection.why if state.selection else "No final was selected."
    report = render_report(
        result,
        run_id=state.run_id,
        domain=domain,
        config=config,
        n_cases=n_cases,
        why=why,
        reflect_calls=state.reflection.calls if state.reflection else 0,
        select_calls=state.selection.calls if state.selection else 0,
    )
    return result.model_copy(update={"report_md": report})
