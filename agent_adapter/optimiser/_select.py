"""Final choice: the best judged final that beats the base, or a blind LLM pick without a judge."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_adapter.core.errors import ModelError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import ErrorInfo
from agent_adapter.optimiser._fatal import raise_if_fatal
from agent_adapter.optimiser.evidence import clip, render_case
from agent_adapter.optimiser.model import Candidate, EvalCase, HarnessTree, ProbeReport
from agent_adapter.optimiser.prompts import SELECT_SCHEMA, select_system
from agent_adapter.optimiser.surface import always_on_chars

logger = logging.getLogger(__name__)

_ANSWER_CHARS = 3_000  # at most per case and set
_MIN_CASE_CHARS = 300  # fewer cases are shown rather than less than this per case


@dataclass(frozen=True)
class Choice:
    """The chosen final (None: KEEP_BASE), why, and how many selector calls it took."""

    chosen: str | None
    why: str
    calls: int = 0


class _Pick(BaseModel):
    model_config = ConfigDict(extra="ignore")

    why: str = ""
    choice: str


async def select_final(
    base: ProbeReport,
    base_tree: HarnessTree,
    finals: Sequence[tuple[Candidate, ProbeReport]],
    *,
    judged: bool,
    min_gain: float,
    seed: int,
    cases: Mapping[str, EvalCase],
    selector: LLMPort | None,
    budget: int = 60_000,
) -> Choice:
    """Pick among finals that change the harness and answer at least as many cases as the base.

    `budget` bounds the blind selector's prompt (characters).

    Raises:
        BudgetExceeded: a budget hook stopped the selector call.
        OptimiserError: a fatal model error in the selector call.
    """
    changed = [(c, r) for c, r in finals if c.tree != base_tree]
    safe = [(c, r) for c, r in changed if r.n_delivered >= base.n_delivered]
    unsafe = [c.label for c, r in changed if r.n_delivered < base.n_delivered]
    note = f"; left out (fewer answers than base): {', '.join(unsafe)}" if unsafe else ""
    if not safe:
        return Choice(None, "KEEP_BASE: no final changes the harness safely" + note)
    graded = base.n_graded > 0 or any(r.n_graded for _, r in safe)
    if judged and graded:
        return _by_score(base, safe, min_gain, note)
    if selector is None:
        return Choice(None, "KEEP_BASE: no judge scores and no selector" + note)
    return await _blind(base, safe, selector, seed, cases, note, budget)


def _by_score(
    base: ProbeReport,
    finals: Sequence[tuple[Candidate, ProbeReport]],
    min_gain: float,
    note: str,
) -> Choice:
    floor = base.mean_score or 0.0
    beating = [
        (c, r)
        for c, r in finals
        if r.mean_score is not None and r.mean_score > floor and r.mean_score - floor >= min_gain
    ]
    if not beating:
        return Choice(None, f"KEEP_BASE: no final beats the base mean {floor:.3f}" + note)
    best, report = min(
        beating,
        key=lambda cr: (-(cr[1].mean_score or 0.0), always_on_chars(cr[0].tree), cr[0].label),
    )
    return Choice(
        best.label,
        f"{best.label}: mean {report.mean_score:.3f} vs base {floor:.3f}, "
        f"answered {report.n_delivered}/{len(report.runs)}" + note,
    )


async def _blind(
    base: ProbeReport,
    finals: Sequence[tuple[Candidate, ProbeReport]],
    selector: LLMPort,
    seed: int,
    cases: Mapping[str, EvalCase],
    note: str,
    budget: int,
) -> Choice:
    """Anonymised answer sets (base included, seeded order); the selector picks a letter."""
    pool: list[tuple[str | None, ProbeReport]] = [(None, base), *((c.label, r) for c, r in finals)]
    random.Random(seed).shuffle(pool)
    letters = {chr(ord("A") + i): entry for i, entry in enumerate(pool)}
    request = LLMRequest(
        messages=(
            Message(role="system", content=select_system()),
            Message(role="user", content=_render_sets(letters, cases, budget)),
        ),
        output_schema=SELECT_SCHEMA,
        stage="select",
        seed=seed,
    )
    try:
        response = await selector.generate(request)
        pick = _Pick.model_validate(response.parsed)
    except ModelError as exc:
        raise_if_fatal(exc, "selection")
        info = ErrorInfo.from_exception(exc)
        logger.warning("blind selection failed (%s); keeping the base", info.kind)
        return Choice(None, f"KEEP_BASE: selector failed ({info.kind})" + note, calls=1)
    except ValidationError:
        logger.warning("blind selection returned an invalid reply; keeping the base")
        return Choice(None, "KEEP_BASE: selector failed (invalid reply)" + note, calls=1)
    entry = letters.get(pick.choice.strip().upper())
    if entry is None:
        logger.warning("blind selection chose unknown option %r; keeping the base", pick.choice)
        return Choice(None, f"KEEP_BASE: selector failed (unknown choice {pick.choice!r})", 1)
    label = entry[0]
    if label is None:
        return Choice(None, f"KEEP_BASE: the selector preferred the base: {pick.why}" + note, 1)
    return Choice(label, pick.why, calls=1)


def _render_sets(
    letters: Mapping[str, tuple[str | None, ProbeReport]],
    cases: Mapping[str, EvalCase],
    budget: int,
) -> str:
    """Every answer set within `budget`: the same cases for each set, each case its share."""
    reports = [report for _, report in letters.values()]
    case_ids = [run.case_id for run in reports[0].runs] if reports else []
    per_set = max(budget // max(len(reports), 1) - 40, 0)  # 40: the set heading
    shown = case_ids[: max(1, per_set // _MIN_CASE_CHARS)]
    per_case = min(_ANSWER_CHARS, max(per_set // max(len(shown), 1) - 2, 0))
    parts: list[str] = []
    if len(shown) < len(case_ids):
        parts.append(f"(The first {len(shown)} of {len(case_ids)} cases are shown.)")
    for letter, (_, report) in letters.items():
        parts.append(f"# Answer set {letter}")
        by_case = {run.case_id: run for run in report.runs}
        for case_id in shown:
            run = by_case.get(case_id)
            if run is None:
                continue
            blind = run.model_copy(update={"transcript": "", "judge_notes": "", "error": ""})
            text = render_case(
                cases.get(case_id), blind, obs_trunc=0, show_scores=False, limit=per_case
            )
            parts.append(clip(text, per_case))
    return clip("\n\n".join(parts), budget)
