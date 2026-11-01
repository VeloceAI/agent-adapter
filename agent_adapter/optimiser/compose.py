"""Compose: admit candidates that helped safely, resolve conflicting paths, build one harness."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_adapter.core.errors import ModelError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import ErrorInfo
from agent_adapter.optimiser._fatal import raise_if_fatal
from agent_adapter.optimiser.model import (
    Candidate,
    Contrast,
    DecisionLine,
    Edit,
    ProbeReport,
    SurfaceCaps,
)
from agent_adapter.optimiser.probe import contrast
from agent_adapter.optimiser.prompts import MERGE_SCHEMA, merge_system
from agent_adapter.optimiser.surface import Validates, cap_key, validate_tree

logger = logging.getLogger(__name__)

ComposeMode = Literal["greedy", "llm"]


@dataclass(frozen=True)
class Assessment:
    """One candidate measured against the reference, and whether composition may use it."""

    candidate: Candidate
    contrasts: list[Contrast]
    gain: float | None  # mean score change over cases graded under both; None: no scores
    improved: int
    regressed: int
    undelivered: int
    exercised: int
    observable: int
    admitted: bool
    reason: str = ""

    def evidence(self) -> str:
        """The counts behind a decision, e.g. `improved 2/3, regressed 0/3, ...`."""
        n = len(self.contrasts)
        gain = "n/a" if self.gain is None else f"{self.gain:+.3f}"
        text = (
            f"improved {self.improved}/{n}, regressed {self.regressed}/{n}, "
            f"undelivered {self.undelivered}, gain {gain}"
        )
        if self.observable:
            text += f", exercised {self.exercised}/{self.observable}"
        return text


def assess(
    candidate: Candidate, report: ProbeReport | None, ref: ProbeReport, *, min_gain: float
) -> Assessment:
    """Measure a candidate; admitted only if it improved, lost no answer and was exercised."""
    rows = contrast(ref, report, candidate) if report is not None else []
    deltas = [r.score_delta for r in rows if r.score_delta is not None]
    gain = sum(deltas) / len(deltas) if deltas else None
    improved = sum(1 for r in rows if _better(r))
    regressed = sum(1 for r in rows if _worse(r))
    undelivered = sum(1 for r in rows if r.delivered == (True, False))
    observable = sum(1 for r in rows if r.edit_exercised is not None)
    exercised = sum(1 for r in rows if r.edit_exercised)
    measured = Assessment(
        candidate, rows, gain, improved, regressed, undelivered, exercised, observable, False
    )
    reason = _rejection(measured, report, min_gain)
    return Assessment(**{**measured.__dict__, "admitted": not reason, "reason": reason})


def _better(row: Contrast) -> bool:
    if row.score_delta is not None:
        return row.score_delta > 0
    return row.delivered == (False, True)


def _worse(row: Contrast) -> bool:
    if row.score_delta is not None:
        return row.score_delta < 0
    return row.delivered == (True, False)


def _rejection(a: Assessment, report: ProbeReport | None, min_gain: float) -> str:
    if a.candidate.fell_back:
        return f"fell back: {a.candidate.error}"
    if report is None:
        return "not probed"
    if a.undelivered:
        return f"unsafe: {a.undelivered} case(s) no longer answered"
    if a.observable and not a.exercised:
        return "never exercised"
    if a.improved == 0 and a.gain is not None:
        return "no case improved"
    if a.gain is not None and a.gain < min_gain:
        return f"gain {a.gain:+.3f} below {min_gain:g}"
    return ""


@dataclass(frozen=True)
class ComposeOutcome:
    """The composed candidate, its decision lines, the LLM merge calls made, and assessments."""

    candidate: Candidate
    decisions: list[DecisionLine]
    merge_calls: int = 0
    assessments: list[Assessment] = field(default_factory=list)


async def compose(
    label: str,
    parent: Candidate,
    ref: ProbeReport,
    pool: Sequence[tuple[Candidate, ProbeReport | None]],
    *,
    caps: SurfaceCaps,
    host: Validates,
    mode: ComposeMode,
    min_gain: float,
    seed: int,
    composer: LLMPort | None = None,
    drop: int = 0,
) -> ComposeOutcome:
    """One composed harness from the admitted candidates of `pool`, minus the `drop` weakest.

    Raises:
        BudgetExceeded: a budget hook stopped the merge call.
        OptimiserError: a fatal model error in the merge call.
    """
    assessments = [assess(c, r, ref, min_gain=min_gain) for c, r in pool if c.edits]
    admitted = _ranked([a for a in assessments if a.admitted], seed)
    used = admitted[: max(len(admitted) - drop, 1)]
    decisions = [
        DecisionLine(kind="dropped", path=e.path, evidence=f"{a.candidate.label}: {a.reason}")
        for a in assessments
        if not a.admitted
        for e in a.candidate.edits
    ]
    decisions += [
        DecisionLine(
            kind="dropped",
            path=e.path,
            evidence=f"{a.candidate.label}: left out of {label} (rank {rank})",
        )
        for rank, a in enumerate(admitted, 1)
        if a not in used
        for e in a.candidate.edits
    ]
    chosen, lines, conflicts = _greedy(used)
    calls = 0
    if conflicts and mode == "llm" and composer is not None:
        calls = 1
        merged = await _llm_merge(composer, label, parent, conflicts, caps)
        if isinstance(merged, str):
            lines = [_note(d, f"; llm merge failed: {merged}") for d in lines]
        elif problems := validate_tree(
            parent.tree.with_edits({**chosen, **{p: c for p, (c, _) in merged.items()}}),
            caps,
            host,
        ):
            logger.warning("%s: merged harness invalid; greedy choice kept", label)
            note = f"; merged harness invalid ({'; '.join(problems)}); greedy choice kept"
            lines = [_note(d, note) for d in lines]
        else:
            chosen.update({p: content for p, (content, _) in merged.items()})
            lines = [
                DecisionLine(
                    kind="revised", path=d.path, evidence=f"merged by LLM: {merged[d.path][1]}"
                )
                if d.kind == "revised" and d.path in merged
                else d
                for d in lines
            ]
    candidate, lines = _build(label, parent, chosen, lines, used, caps, host)
    return ComposeOutcome(candidate, [*lines, *decisions], calls, assessments)


def _ranked(admitted: list[Assessment], seed: int) -> list[Assessment]:
    """Best first: gain, then cases improved, then a seeded tie-break."""
    rng = random.Random(seed)
    ties = {
        a.candidate.label: rng.random() for a in sorted(admitted, key=lambda a: a.candidate.label)
    }
    return sorted(
        admitted,
        key=lambda a: (-(a.gain or 0.0), -a.improved, ties[a.candidate.label], a.candidate.label),
    )


Conflict = list[tuple[Assessment, Edit]]


def _greedy(
    used: Sequence[Assessment],
) -> tuple[dict[str, str | None], list[DecisionLine], dict[str, Conflict]]:
    """Single writers taken; a conflict takes the best-ranked writer (decision `revised`)."""
    writers: dict[str, Conflict] = {}
    for a in used:
        for edit in a.candidate.edits:
            writers.setdefault(edit.path, []).append((a, edit))
    chosen: dict[str, str | None] = {}
    lines: list[DecisionLine] = []
    conflicts: dict[str, Conflict] = {}
    for path, entries in writers.items():
        best, edit = entries[0]
        chosen[path] = edit.content
        versions = {e.content for _, e in entries}
        if len(versions) == 1:
            lines.append(
                DecisionLine(
                    kind="kept", path=path, evidence=f"{best.candidate.label}: {best.evidence()}"
                )
            )
            continue
        conflicts[path] = entries
        others = ", ".join(a.candidate.label for a, _ in entries[1:])
        lines.append(
            DecisionLine(
                kind="revised",
                path=path,
                evidence=f"conflict: took {best.candidate.label} over {others}; {best.evidence()}",
            )
        )
    return chosen, lines, conflicts


class _MergeItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    content: str
    decision: str = ""


class _MergeReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    merges: list[_MergeItem]


async def _llm_merge(
    composer: LLMPort,
    label: str,
    parent: Candidate,
    conflicts: dict[str, Conflict],
    caps: SurfaceCaps,
) -> dict[str, tuple[str, str]] | str:
    """`{path: (merged text, decision)}` for every conflict, or the reason the merge failed."""
    request = LLMRequest(
        messages=(
            Message(role="system", content=merge_system()),
            Message(role="user", content=_merge_evidence(parent, conflicts, caps)),
        ),
        output_schema=MERGE_SCHEMA,
        stage=f"merge:{label}",
    )
    try:
        response = await composer.generate(request)
        reply = _MergeReply.model_validate(response.parsed)
    except ModelError as exc:
        raise_if_fatal(exc, f"merge {label}")
        info = ErrorInfo.from_exception(exc)
        logger.warning("llm merge for %s failed (%s); greedy choice kept", label, info.kind)
        return info.kind
    except ValidationError:
        logger.warning("llm merge for %s returned an invalid reply; greedy choice kept", label)
        return "invalid reply"
    merged = {m.path: (m.content, m.decision) for m in reply.merges}
    problems = [f"{p}: not a conflicting path" for p in merged if p not in conflicts]
    problems += [f"{p}: no merged text" for p in conflicts if p not in merged]
    for path, (content, _) in merged.items():
        cap = caps.max_chars.get(cap_key(path)) if path in conflicts else None
        if cap is not None and len(content) > cap:
            problems.append(f"{path}: {len(content)} chars, cap is {cap}")
    if problems:
        logger.warning("llm merge for %s rejected: %s", label, "; ".join(problems))
        return "; ".join(problems)
    return merged


def _merge_evidence(parent: Candidate, conflicts: dict[str, Conflict], caps: SurfaceCaps) -> str:
    parts: list[str] = []
    for path, entries in conflicts.items():
        cap = caps.max_chars.get(cap_key(path))
        parts.append(f"# {path}" + (f" (at most {cap} chars)" if cap else ""))
        parts.append("## current text\n" + (parent.tree.files.get(path) or "(none)"))
        for a, edit in entries:
            text = edit.content if edit.content is not None else "(deleted)"
            parts.append(
                f"## version from {a.candidate.label} ({a.evidence()})\n"
                f"reason given: {edit.reason}\n{text}"
            )
    return "\n\n".join(parts)


def _note(line: DecisionLine, suffix: str) -> DecisionLine:
    if line.kind != "revised":
        return line
    return line.model_copy(update={"evidence": line.evidence + suffix})


def _build(
    label: str,
    parent: Candidate,
    chosen: dict[str, str | None],
    lines: list[DecisionLine],
    used: Sequence[Assessment],
    caps: SurfaceCaps,
    host: Validates,
) -> tuple[Candidate, list[DecisionLine]]:
    """The composed candidate; an invalid combination falls back to the best single candidate."""
    tree = parent.tree.with_edits(chosen)
    problems = validate_tree(tree, caps, host) if chosen else []
    if problems:
        best = used[0].candidate
        logger.warning("%s: combined edits invalid; adopting %s alone", label, best.label)
        chosen = {e.path: e.content for e in best.edits}
        tree = parent.tree.with_edits(chosen)
        note = f"combination invalid ({'; '.join(problems)}); adopted {best.label} alone"
        lines = [
            DecisionLine(kind="revised" if p in chosen else "dropped", path=p, evidence=note)
            for p in sorted({d.path for d in lines} | set(chosen))
        ]
    evidence = {d.path: d.evidence for d in lines}
    edits = tuple(
        Edit(path=p, content=c, reason=evidence.get(p, ""))
        for p, c in sorted(chosen.items())
        if parent.tree.files.get(p) != c
    )
    candidate = Candidate(
        label=label,
        dim="compose",
        edits=edits,
        tree=tree,
        parent_label=parent.label,
        notes=f"from {', '.join(a.candidate.label for a in used) or 'nothing (keep base)'}",
    )
    return candidate, lines
