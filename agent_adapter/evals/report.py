"""`EvalReport`: per-case results, totals, per-tag and per-grader breakdowns, and comparisons."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, computed_field

from agent_adapter.evals._render import junit, markdown

_FROZEN = ConfigDict(frozen=True)


class GradeResult(BaseModel):
    """One grader's verdict on one case; `score` None means it did not grade."""

    model_config = _FROZEN

    grader: str
    score: float | None
    passed: bool
    notes: str = ""


class CaseResult(BaseModel):
    """One case: its run id (open it in the viewer), verdict, grades, cost and time."""

    model_config = _FROZEN

    case_id: str
    tags: tuple[str, ...] = ()
    run_id: str = ""
    passed: bool
    score: float | None  # mean of every grade (ungraded counts 0); None when it errored
    grades: tuple[GradeResult, ...] = ()
    input: str = ""  # clipped text of the case input
    expected: str = ""  # clipped text of the expected value; "" when the case has none
    output: str = ""  # clipped text of the output
    error: str = ""  # "<kind>: <message>" when the run failed or did not finish
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tool_calls: tuple[str, ...] = ()
    measured: bool = True  # False: cost and tool calls were not observed (shown as 0)


class GroupSummary(BaseModel):
    """Totals of a group of cases (one tag, or one grader)."""

    model_config = _FROZEN

    n: int
    n_passed: int
    mean_score: float


class CaseChange(BaseModel):
    """One case that got worse or better between two reports."""

    model_config = _FROZEN

    case_id: str
    before_passed: bool
    after_passed: bool
    before_score: float
    after_score: float


class EvalComparison(BaseModel):
    """How a report differs from a baseline: regressions, improvements, added and removed cases."""

    model_config = _FROZEN

    score_delta: float
    regressions: tuple[CaseChange, ...]
    improvements: tuple[CaseChange, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def regressed(self) -> bool:
        """True when a shared case got worse, or a baseline case is missing now."""
        return bool(self.regressions or self.removed)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _value(case: CaseResult) -> float:
    """A case's score for totals: errors and ungraded cases count as 0."""
    return 0.0 if case.error or case.score is None else case.score


class EvalReport(BaseModel):
    """Every case result of one eval, with totals; render it as JUnit, Markdown or JSON."""

    model_config = _FROZEN

    name: str
    started_at: str  # ISO 8601, UTC
    wall_s: float
    cases: tuple[CaseResult, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_cases(self) -> int:
        """How many cases ran."""
        return len(self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_passed(self) -> int:
        """How many cases passed."""
        return sum(case.passed for case in self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_errors(self) -> int:
        """How many cases failed to run or did not finish."""
        return sum(bool(case.error) for case in self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_score(self) -> float:
        """Mean case score; errors and ungraded cases count as 0."""
        return _mean([_value(case) for case in self.cases])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        """Share of cases that passed."""
        return self.n_passed / self.n_cases if self.cases else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_usd(self) -> float:
        """Total cost of every case run."""
        return sum(case.cost_usd for case in self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def by_tag(self) -> dict[str, GroupSummary]:
        """Totals per tag; a case counts once under each of its tags."""
        groups: dict[str, list[CaseResult]] = defaultdict(list)
        for case in self.cases:
            for tag in case.tags:
                groups[tag].append(case)
        return {tag: _summary(group) for tag, group in sorted(groups.items())}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def by_grader(self) -> dict[str, GroupSummary]:
        """Totals per grader over the cases it saw; an ungraded case counts as 0."""
        grades: dict[str, list[GradeResult]] = defaultdict(list)
        for case in self.cases:
            for grade in case.grades:
                grades[grade.grader].append(grade)
        return {
            name: GroupSummary(
                n=len(group),
                n_passed=sum(g.passed for g in group),
                mean_score=_mean([g.score or 0.0 for g in group]),
            )
            for name, group in sorted(grades.items())
        }

    @property
    def failed(self) -> tuple[CaseResult, ...]:
        """The cases that did not pass."""
        return tuple(case for case in self.cases if not case.passed)

    def compare(self, baseline: EvalReport, *, tolerance: float = 0.0) -> EvalComparison:
        """This report against `baseline`: a case regresses when it stops passing or loses
        more than `tolerance` of score."""
        before = {case.case_id: case for case in baseline.cases}
        after = {case.case_id: case for case in self.cases}
        regressions: list[CaseChange] = []
        improvements: list[CaseChange] = []
        for case_id in sorted(before.keys() & after.keys()):
            old, new = before[case_id], after[case_id]
            change = CaseChange(
                case_id=case_id,
                before_passed=old.passed,
                after_passed=new.passed,
                before_score=_value(old),
                after_score=_value(new),
            )
            drop = change.before_score - change.after_score
            if (old.passed and not new.passed) or drop > tolerance:
                regressions.append(change)
            elif (new.passed and not old.passed) or -drop > tolerance:
                improvements.append(change)
        return EvalComparison(
            score_delta=self.mean_score - baseline.mean_score,
            regressions=tuple(regressions),
            improvements=tuple(improvements),
            added=tuple(sorted(after.keys() - before.keys())),
            removed=tuple(sorted(before.keys() - after.keys())),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """The report, with its totals, as JSON (`EvalReport.from_json` reads it back)."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str | bytes) -> EvalReport:
        """A report from `to_json` output; the totals are recomputed.

        Raises:
            pydantic.ValidationError: not a report.
        """
        return cls.model_validate_json(text)

    def to_markdown(self) -> str:
        """A Markdown summary: totals, per-tag table and the failing cases."""
        return markdown(self)

    def to_junit(self) -> str:
        """JUnit XML: one test case per eval case, failures and errors with their reasons."""
        return junit(self)


def _summary(cases: list[CaseResult]) -> GroupSummary:
    return GroupSummary(
        n=len(cases),
        n_passed=sum(case.passed for case in cases),
        mean_score=_mean([_value(case) for case in cases]),
    )
