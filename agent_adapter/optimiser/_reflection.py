"""The reflection call's evidence: base harness, every candidate's diff and fate, the outcome."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from agent_adapter.core.ports import LLMPort
from agent_adapter.optimiser._state import ComposeRecord, RunState
from agent_adapter.optimiser.evidence import clip, render_tree
from agent_adapter.optimiser.lessons import reflect
from agent_adapter.optimiser.model import Candidate, HarnessTree, Lesson

_DIFF_CHARS = 4_000


def reflection_evidence(
    *,
    base: HarnessTree,
    candidates: Mapping[str, Candidate],
    composes: Mapping[str, ComposeRecord],
    chosen: str | None,
    why: str,
    limit: int,
) -> str:
    """At most `limit` characters describing what was tried, what happened and what was kept."""
    assessed: dict[str, list[str]] = {}
    for stage, record in composes.items():
        for label, line in record.assessed.items():
            assessed.setdefault(label, []).append(f"{stage}: {line}")
    parts = ["# BASE HARNESS\n" + clip(render_tree(base), limit // 4), "# CANDIDATES"]
    for label, cand in candidates.items():
        if cand.dim == "base":
            continue
        parent = candidates.get(cand.parent_label)
        diff = parent.tree.unified_diff(cand.tree, a=parent.label, b=label) if parent else ""
        head = f"## {label} ({cand.dim}, from {cand.parent_label})"
        if cand.fell_back:
            head += f"\nfell back: {cand.error}"
        elif not cand.edits:
            head += "\nno change proposed"
        reasons = "".join(f"\n- {e.path}: {e.reason}" for e in cand.edits)
        measured = "".join(f"\n- measured in {line}" for line in assessed.get(label, []))
        parts.append(head + reasons + measured + ("\n" + clip(diff, _DIFF_CHARS) if diff else ""))
    parts.append("# DECISIONS")
    parts += [
        f"- [{stage}] {d.kind} {d.path}: {d.evidence}"
        for stage, record in composes.items()
        for d in record.decisions
    ] or ["- none"]
    if chosen is None:
        parts.append(f"# OUTCOME\nThe base harness was kept. {why}")
    else:
        final = candidates[chosen].tree
        parts.append(
            f"# OUTCOME\nChosen: {chosen}. {why}\n"
            + clip(base.unified_diff(final, a="base", b=chosen), _DIFF_CHARS)
        )
    return clip("\n\n".join(parts), limit)


async def reflect_on_run(
    state: RunState, composer: LLMPort, *, base: HarnessTree, domain: str, run: str, limit: int
) -> tuple[Lesson | None, str]:
    """One reflection call over the run so far; adopted paths are the chosen tree's changes.

    Raises:
        BudgetExceeded: a budget hook stopped the call.
        OptimiserError: a fatal model error.
    """
    chosen = state.selection.chosen if state.selection else None
    chosen_tree = state.candidates[chosen].tree if chosen else base
    evidence = reflection_evidence(
        base=base,
        candidates=state.candidates,
        composes=state.composes,
        chosen=chosen,
        why=state.selection.why if state.selection else "",
        limit=limit,
    )
    diff = base.diff(chosen_tree)
    return await reflect(
        composer,
        domain=domain,
        run=run,
        evidence=evidence,
        adopted_paths={*diff.added, *diff.removed, *diff.modified},
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
