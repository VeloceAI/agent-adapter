"""The proposer's evidence: harness, cases, transcripts, lessons and history, within a budget."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Mapping, Sequence

from agent_adapter.optimiser.model import CaseRun, EvalCase, HarnessTree, Lesson, ProbeReport

_BLOCK = re.compile(r"^\[[A-Z][A-Z ]*\]", re.MULTILINE)
MIN_CASE_CHARS = 800  # below this a case shows too little to be worth including


def clip(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters, with a marker saying how much was cut."""
    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit + 31} chars clipped]"
    marker = f"\n[... {len(text) - (limit - len(marker))} chars clipped]"
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)] + marker


def clip_observations(transcript: str, limit: int) -> str:
    """Each `[TAG]` block of a transcript cut to `limit` characters."""
    starts = [m.start() for m in _BLOCK.finditer(transcript)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    bounds = [*starts, len(transcript)]
    blocks = [transcript[a:b].rstrip("\n") for a, b in itertools.pairwise(bounds)]
    return "\n".join(clip(block, limit) for block in blocks if block)


def worst_first(runs: Sequence[CaseRun]) -> list[CaseRun]:
    """Undelivered runs first, then by ascending score (ungraded in the middle), then by id."""
    return sorted(
        runs,
        key=lambda r: (r.delivered, r.score if r.score is not None else 0.5, r.case_id),
    )


def render_tree(tree: HarnessTree) -> str:
    """Every path of the tree with its text."""
    parts = [
        f"## {path}\n{text}" if text else f"## {path}\n(empty)"
        for path, text in sorted(tree.files.items())
    ]
    return "\n\n".join(parts) or "(the harness is empty)"


def _input_text(case: EvalCase | None) -> str:
    if case is None:
        return "(unknown case)"
    if isinstance(case.input, str):
        return case.input
    return json.dumps(case.input, ensure_ascii=False, sort_keys=True, default=str)


def render_case(
    case: EvalCase | None, run: CaseRun, *, obs_trunc: int, show_scores: bool, limit: int
) -> str:
    """One case: request, delivery, remarks, answer and transcript; never the expected answer."""
    tags = f" [{', '.join(case.tags)}]" if case is not None and case.tags else ""
    head = [f"## Case {run.case_id}{tags}", f"Request: {clip(_input_text(case), 2_000)}"]
    status = "answered" if run.delivered else "NO ANSWER"
    head.append(f"Outcome: {status}" + (f"; error: {run.error}" if run.error else ""))
    if run.judge_notes:
        head.append(f"Reviewer: {clip(run.judge_notes, 1_000)}")
    if show_scores and run.score is not None:
        verdict = "passed" if run.passed else "not passed"
        head.append(f"Result: {run.score:.2f} ({verdict})")
    head.append("Answer:\n" + clip(run.output or "(none)", max(200, limit // 4)))
    text = "\n".join(head)
    room = limit - len(text) - len("\nTranscript:\n")
    transcript = clip(clip_observations(run.transcript, obs_trunc), max(room, 0))
    if transcript:
        text += "\nTranscript:\n" + transcript
    return clip(text, limit)


def render_lessons(lessons: Sequence[Lesson]) -> str:
    """Lessons from earlier runs as a short list; empty when there are none."""
    return "\n".join(
        f"- When {lesson.cue}: {lesson.lesson} (evidence: {lesson.evidence}) ({lesson.confidence})"
        for lesson in lessons
    )


def build_evidence(
    *,
    tree: HarnessTree,
    cases: Mapping[str, EvalCase],
    report: ProbeReport,
    obs_trunc: int,
    budget: int,
    show_scores: bool,
    lessons_text: str = "",
    history: str = "",
) -> str:
    """The proposer's user turn in at most `budget` characters, weakest cases first.

    When the cases cannot all get `MIN_CASE_CHARS`, only the weakest that fit are shown.
    """
    fixed = ["# CURRENT HARNESS\n" + clip(render_tree(tree), budget // 2)]
    if lessons_text:
        fixed.append(
            "# EARLIER RUNS IN THIS AREA\nNotes from earlier runs; check each against the "
            "transcripts below before relying on it.\n" + clip(lessons_text, budget // 8)
        )
    if history:
        fixed.append("# HISTORY\n" + clip(history, budget // 4))
    runs = worst_first(report.runs)
    room = budget - sum(len(part) + 2 for part in fixed) - 80  # 80: the cases header
    shown = runs[: max(1, room // MIN_CASE_CHARS)] if runs else []
    header = f"# CASES ({len(runs)}, weakest first"
    header += f"; the {len(shown)} weakest shown)" if len(shown) < len(runs) else ")"
    share = max(room // max(len(shown), 1) - 2, 0)
    rendered = [
        render_case(
            cases.get(r.case_id), r, obs_trunc=obs_trunc, show_scores=show_scores, limit=share
        )
        for r in shown
    ]
    return clip("\n\n".join([*fixed, header, *rendered]), budget)
