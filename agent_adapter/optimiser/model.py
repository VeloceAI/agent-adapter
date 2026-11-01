"""Optimiser data model: the harness tree, candidates, case runs, probes, decisions, results."""

from __future__ import annotations

import contextlib
import difflib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_adapter.cost.summary import LedgerSummary

_FROZEN = ConfigDict(frozen=True)

# tool/<name>/description | tool/<name>/prompt_hint | hook/<name>/config.json
# | prompt/system | prompt/layout/<section> | flow/<flow>/<node>/prompt
ComponentPath = str


class DiffSummary(BaseModel):
    """Which paths two trees add, remove or change, and the total character change."""

    model_config = _FROZEN

    added: list[ComponentPath]
    removed: list[ComponentPath]
    modified: list[ComponentPath]
    char_delta: int


class HarnessTree(BaseModel):
    """Every editable harness component as path -> text."""

    model_config = _FROZEN

    files: dict[ComponentPath, str]

    def with_edits(self, edits: Mapping[ComponentPath, str | None]) -> HarnessTree:
        """A copy with `edits` applied: str sets a path, None deletes it, omitted is inherited."""
        files = dict(self.files)
        for path, content in edits.items():
            if content is None:
                files.pop(path, None)
            else:
                files[path] = content
        return HarnessTree(files=files)

    def diff(self, other: HarnessTree) -> DiffSummary:
        """What `other` adds, removes and modifies relative to this tree."""
        mine, theirs = self.files, other.files
        return DiffSummary(
            added=sorted(set(theirs) - set(mine)),
            removed=sorted(set(mine) - set(theirs)),
            modified=sorted(p for p in set(mine) & set(theirs) if mine[p] != theirs[p]),
            char_delta=sum(map(len, theirs.values())) - sum(map(len, mine.values())),
        )

    def unified_diff(self, other: HarnessTree, *, a: str = "base", b: str = "candidate") -> str:
        """A unified diff per changed path (`--- a/<path>`, `+++ b/<path>`); JSON pretty-printed."""
        hunks: list[str] = []
        for path in sorted(set(self.files) | set(other.files)):
            old, new = self.files.get(path), other.files.get(path)
            if old == new:
                continue
            lines = difflib.unified_diff(
                _diff_lines(path, old),
                _diff_lines(path, new),
                fromfile=f"{a}/{path}",
                tofile=f"{b}/{path}",
            )
            hunks.append("".join(lines))
        return "".join(hunks)


def _diff_lines(path: ComponentPath, text: str | None) -> list[str]:
    if text is None:
        return []
    if path.endswith(".json"):
        with contextlib.suppress(ValueError):  # invalid JSON is diffed as written
            text = json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False)
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


class SurfaceCaps(BaseModel):
    """Character caps per path pattern, the editable kinds, and JSON schemas of hook configs."""

    model_config = _FROZEN

    max_chars: dict[str, int]  # keyed by pattern, e.g. "tool/*/description", "prompt/system"
    kinds: frozenset[str]
    hook_schemas: dict[str, dict[str, Any]] = {}


class EvalCase(BaseModel):
    """One evaluation input; `expected` is for the host's judge only."""

    model_config = _FROZEN

    id: str
    input: Any
    tags: tuple[str, ...] = ()
    expected: Any = None


class Score(BaseModel):
    """A judge's verdict on one case run; `value` is in [0, 1]."""

    model_config = _FROZEN

    value: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    notes: str = ""


class Edit(BaseModel):
    """One proposed change: `content` None deletes the path; `reason` cites the evidence."""

    model_config = _FROZEN

    path: ComponentPath
    content: str | None
    reason: str


class Candidate(BaseModel):
    """A harness proposal (or the base): its edits, the tree they produce and how it was made."""

    model_config = _FROZEN

    label: str
    dim: Literal["procedure", "knowledge", "compose", "base"]
    edits: tuple[Edit, ...]  # () == KEEP_BASE
    tree: HarnessTree
    parent_label: str
    notes: str = ""
    fell_back: bool = False
    error: str = ""
    cost_usd: float = 0.0
    attempts: int = 0


class CaseRun(BaseModel):
    """One eval case run under one harness: output, transcript, metrics and the judge's view."""

    model_config = _FROZEN

    case_id: str
    label: str
    ok: bool
    delivered: bool
    output: str
    transcript: str
    metrics: dict[str, float] = {}
    score: float | None = None  # None: ungraded or the judge failed
    passed: bool | None = None
    judge_notes: str = ""
    error: str = ""


class ProbeReport(BaseModel):
    """Every case run of one harness, with score and delivery totals."""

    model_config = _FROZEN

    label: str
    runs: list[CaseRun]
    mean_score: float | None
    n_passed: int
    n_graded: int
    n_delivered: int
    cost_usd: float


class Contrast(BaseModel):
    """One case under a candidate against a reference harness."""

    model_config = _FROZEN

    case_id: str
    ref_label: str
    cand_label: str
    score_delta: float | None
    delivered: tuple[bool, bool]  # (reference, candidate)
    metric_deltas: dict[str, float]
    edit_exercised: bool | None  # None: not observable (prompt edits)
    transcript_diff: str


class DecisionLine(BaseModel):
    """What composition did with one path, and the counts behind it."""

    model_config = _FROZEN

    kind: Literal["kept", "revised", "dropped"]
    path: ComponentPath
    evidence: str


class Lesson(BaseModel):
    """A transferable cue from one run, stored in `lessons.json` for later runs."""

    model_config = _FROZEN

    id: str
    domain: str
    created_at: str
    cue: str
    lesson: str
    evidence: str
    confidence: Literal["low", "medium", "high"]
    component_paths: list[ComponentPath]
    run: str
    adopted: bool


class OptimiserBudget(BaseModel):
    """Limits for one optimiser run; None means unlimited."""

    model_config = _FROZEN

    max_usd: float | None = None
    max_case_runs: int | None = None
    max_wall_s: float | None = None


class OptimiserConfig(BaseModel):
    """Settings of one optimiser run (`k` is the main cost knob: `(1 + 3k) * N` case runs)."""

    model_config = _FROZEN

    k: int = 1
    mode: Literal["parallel", "sequential"] = "parallel"
    compose_mode: Literal["greedy", "llm"] = "greedy"
    min_gain: float = 0.0
    rollout_concurrency: int = 4
    case_timeout_s: float | None = 300.0  # per case run (and its judging); None: no limit
    evidence_chars: int = 60_000
    obs_trunc: int = 4_000
    show_scores_to_proposer: bool = False
    max_lessons: int = 8
    lessons_path: str = ".agent_adapter/lessons.json"  # "" disables lessons
    out_dir: str = "runs/opt"
    namespace: str = "aa-opt"
    seed: int = 0
    budget: OptimiserBudget = OptimiserBudget()


class BudgetUsage(BaseModel):
    """What a run spent; `stopped` names the limit that ended it early ("" when none did)."""

    model_config = _FROZEN

    usd: float
    case_runs: int
    wall_s: float
    proposer_calls: int
    merge_calls: int
    stopped: str = ""


class OptimiseResult(BaseModel):
    """Everything one optimiser run produced; nothing in it has been applied to the host."""

    model_config = _FROZEN

    base: ProbeReport
    candidates: dict[str, Candidate]
    probes: dict[str, ProbeReport]
    finals: list[str]
    chosen: str | None  # None: KEEP_BASE
    decisions: list[DecisionLine]
    patch: str
    report_md: str
    lessons_added: list[Lesson]
    ledger_summary: LedgerSummary
    budget: BudgetUsage
    errors: list[str]
