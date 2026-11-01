"""Run outputs: `harness.patch`, `report.md`, `result.json`, case files, `harness.after.json`."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from agent_adapter.cache.keys import digest
from agent_adapter.optimiser._files import write_atomic
from agent_adapter.optimiser.model import (
    Candidate,
    EvalCase,
    HarnessTree,
    OptimiserConfig,
    OptimiseResult,
    ProbeReport,
)

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(10)),
    *(f"LPT{i}" for i in range(10)),
}


def case_stems(case_ids: Iterable[str]) -> dict[str, str]:
    """A file stem per case id: safe on every OS, unique even on case-insensitive file systems."""
    stems: dict[str, str] = {}
    for case_id in dict.fromkeys(case_ids):
        stem = _UNSAFE.sub("_", case_id)[:80] or "case"
        if stem != case_id or stem.split(".")[0].upper() in _RESERVED:
            stem = f"{stem}-{digest(case_id)[:8]}"
        stems[case_id] = stem
    folded = Counter(stem.lower() for stem in stems.values())
    return {
        case_id: f"{stem}-{digest(case_id)[:8]}" if folded[stem.lower()] > 1 else stem
        for case_id, stem in stems.items()
    }


def case_file(label: str, stem: str) -> str:
    """Relative path of one case's transcript file, from its `case_stems` stem."""
    return f"runs/{label}/{stem}.md"


def _stems(result: OptimiseResult) -> dict[str, str]:
    reports = [result.base, *result.probes.values()]
    return case_stems(run.case_id for report in reports for run in report.runs)


def patch_for(base: HarnessTree, chosen: HarnessTree | None) -> str:
    """The reviewable patch of the chosen harness against the base ("" for KEEP_BASE)."""
    return "" if chosen is None else base.unified_diff(chosen, a="a", b="b")


def render_report(
    result: OptimiseResult,
    *,
    run_id: str,
    domain: str,
    config: OptimiserConfig,
    n_cases: int,
    why: str,
    reflect_calls: int = 0,
    select_calls: int = 0,
) -> str:
    """`report.md`: header, result, base and candidate tables, decisions, lessons, errors."""
    used, k = result.budget, config.k
    lines = [
        f"# Optimiser report: {domain}",
        "",
        f"Run `{run_id}`, mode {config.mode}, k={k}, compose {config.compose_mode}, "
        f"{n_cases} cases.",
        "",
        "| Budget | Used | Identity |",
        "|---|---|---|",
        f"| case runs | {used.case_runs} | (1 + 3k) x N = {(1 + 3 * k) * n_cases} |",
        f"| proposer calls | {used.proposer_calls} | 2k = {2 * k} |",
        f"| merge calls | {used.merge_calls} | counted separately |",
        f"| reflection calls | {reflect_calls} | at most 1 |",
        f"| selection calls | {select_calls} | at most 1, only without a judge |",
        f"| cost (USD) | {used.usd:.4f} | limit {_limit(config.budget.max_usd)} |",
        f"| wall (s) | {used.wall_s:.1f} | limit {_limit(config.budget.max_wall_s)} |",
        f"| response cache hit rate | {result.ledger_summary.response_cache_hit_rate:.0%} | |",
        "",
    ]
    if used.stopped:
        lines += [f"**Stopped early: {used.stopped}.** This report is partial.", ""]
    chosen = result.chosen or "KEEP_BASE"
    lines += [
        "## Result",
        "",
        f"Chosen: **{chosen}**. {why}",
        "",
        "Finals: " + (", ".join(result.finals) or "none"),
        "",
    ]
    lines += ["## Base", "", *_case_table(result.base), ""]
    lines += ["## Candidates", "", *_candidate_table(result.candidates, result.probes), ""]
    lines += ["## Decisions", ""]
    lines += [f"- {d.kind} -> `{d.path}` -> {d.evidence}" for d in result.decisions] or ["- none"]
    lines += ["", "## Lessons added", ""]
    lines += [f"- {x.cue}: {x.lesson} ({x.confidence})" for x in result.lessons_added] or ["- none"]
    lines += ["", "## Errors and fallbacks", ""]
    lines += [f"- {e}" for e in result.errors] or ["- none"]
    if any(": case-timeout" in e or ", case-timeout" in e for e in result.errors):
        lines += [
            "",
            "Some case runs passed `case_timeout_s`. If the host is slow but healthy, raise "
            "`case_timeout_s` and run again.",
        ]
    lines += ["", "The patch is in `harness.patch`; nothing was applied to the host.", ""]
    return "\n".join(lines)


def _limit(value: float | None) -> str:
    return "none" if value is None else f"{value:g}"


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _case_table(report: ProbeReport) -> list[str]:
    rows = ["| case | answered | score | cost (USD) | error |", "|---|---|---|---|---|"]
    rows += [
        f"| {_cell(r.case_id)} | {'yes' if r.delivered else 'no'} | {_score(r.score)} | "
        f"{r.metrics.get('cost_usd', 0.0):.4f} | {_cell(r.error)} |"
        for r in report.runs
    ]
    return rows


def _candidate_table(
    candidates: Mapping[str, Candidate], probes: Mapping[str, ProbeReport]
) -> list[str]:
    rows = [
        "| label | dim | parent | edits | mean | passed | answered | cost (USD) | attempts "
        "| notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, cand in candidates.items():
        probe = probes.get(label)
        edits = ", ".join(f"`{e.path}`" for e in cand.edits) or "KEEP_BASE"
        note = f"fell back: {cand.error}" if cand.fell_back else cand.notes
        measured = (
            f"{_score(probe.mean_score)} | {probe.n_passed} | {probe.n_delivered}/{len(probe.runs)}"
            if probe
            else "- | - | -"
        )
        cost = cand.cost_usd + (probe.cost_usd if probe else 0.0)
        rows.append(
            f"| {label} | {cand.dim} | {cand.parent_label or '-'} | {edits} | {measured} | "
            f"{cost:.4f} | {cand.attempts} | {_cell(note)} |"
        )
    return rows


def _cell(text: str) -> str:
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= 160 else flat[:157] + "..."


def result_json(result: OptimiseResult) -> str:
    """`result.json`: the result with each transcript replaced by its case file path."""
    data = result.model_dump(mode="json")
    stems = _stems(result)
    for report in [data["base"], *data["probes"].values()]:
        for run in report["runs"]:
            run["transcript"] = case_file(run["label"], stems[run["case_id"]])
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_outputs(
    out_dir: str | os.PathLike[str],
    result: OptimiseResult,
    *,
    run_id: str,
    chosen_tree: HarnessTree,
    cases: Mapping[str, EvalCase],
) -> None:
    """Write every output file atomically under `out_dir`; nothing is applied to the host."""
    root = Path(out_dir)
    stems = _stems(result)
    for report in {"base": result.base, **result.probes}.values():
        for run in report.runs:
            text = (
                f"# {run.label} / {run.case_id}\n\n"
                f"Answered: {'yes' if run.delivered else 'no'}; score: {_score(run.score)}"
                + (f"; error: {run.error}" if run.error else "")
                + f"\n\n## Answer\n\n{run.output}\n\n## Transcript\n\n{run.transcript}\n"
            )
            write_atomic(root / case_file(run.label, stems[run.case_id]), text)
    after = {
        "version": 1,
        "run": run_id,
        "label": result.chosen or "base",
        "files": chosen_tree.files,
    }
    write_atomic(
        root / "harness.after.json", json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    )
    write_atomic(root / "harness.patch", result.patch)
    write_atomic(root / "report.md", result.report_md)
    write_atomic(root / "result.json", result_json(result))
    logger.info("optimiser outputs written to %s (%d cases)", root, len(cases))
