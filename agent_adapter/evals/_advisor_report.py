"""The cost advisor's report: one trial per (step, candidate), advice per step, a mapping."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, computed_field

_FROZEN = ConfigDict(frozen=True)


class StepUse(BaseModel):
    """What one step's model calls cost in one eval run, summed over its cases."""

    model_config = _FROZEN

    calls: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    unpriced: int = 0  # calls the price table could not price


class Trial(BaseModel):
    """One eval run: `model` answers `step` and every other step stays on the baseline."""

    model_config = _FROZEN

    step: str  # "" for the baseline run, "*" for every step at once
    model: str
    score: float  # the eval's mean score
    n_passed: int
    n_errors: int
    cost_usd: float  # every priced ledger entry of the run: models, tools, APIs, judges
    model_cost_usd: float = 0.0  # only the model calls the advisor routed (the steps)
    unpriced: int = 0  # model or API entries of the run without a price
    latency_s: float  # time of the routed model calls, from the ledger
    steps: dict[str, StepUse] = {}
    complete: bool = True  # False: the spend cap stopped it
    keeps_score: bool = False  # score >= min_score_ratio x baseline and no new errors


class StepAdvice(BaseModel):
    """The advice for one step: the cheapest candidate that keeps the score, if any."""

    model_config = _FROZEN

    step: str
    baseline: StepUse
    model: str = ""  # "" keeps the baseline
    score: float | None = None
    cost_delta_usd: float = 0.0  # this step's cost with `model` minus with the baseline
    run_cost_delta_usd: float = 0.0  # the whole run's priced cost with `model` minus baseline
    latency_delta_s: float = 0.0
    reason: str = ""


class AdvisorReport(BaseModel):
    """Per-step model advice from evals and the ledger; apply `mapping` yourself, never auto."""

    model_config = _FROZEN

    name: str
    baseline: str
    min_score_ratio: float
    baseline_trial: Trial | None
    steps: tuple[StepAdvice, ...] = ()
    trials: tuple[Trial, ...] = ()
    combined: Trial | None = None  # every recommended step at once, when it was verified
    max_spend_usd: float | None = None
    spent_usd: float = 0.0
    stopped: str = ""  # "budget": the spend cap stopped the advisor early
    verified: bool = False  # the advice ran all together and kept the score
    note: str = ""  # why `mapping` is empty although steps were advised

    @computed_field  # type: ignore[prop-decorator]
    @property
    def advised(self) -> dict[str, str]:
        """`{step: candidate}` per step on its own, verified together or not."""
        return {advice.step: advice.model for advice in self.steps if advice.model}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mapping(self) -> dict[str, str]:
        """The advice to apply: `advised` once verified together, else empty (see `note`)."""
        return self.advised if self.verified else {}

    def to_json(self, *, indent: int | None = 2) -> str:
        """The report as JSON (`AdvisorReport.from_json` reads it back)."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str | bytes) -> AdvisorReport:
        """A report from `to_json` output.

        Raises:
            pydantic.ValidationError: not an advisor report.
        """
        return cls.model_validate_json(text)

    def to_markdown(self) -> str:
        """A Markdown summary: the baseline, one row per step, and the mapping to apply."""
        return _markdown(self)


def _usd(value: float) -> str:
    return f"{value:+.6f}" if value else "0"


def _markdown(report: AdvisorReport) -> str:
    base = report.baseline_trial
    lines = [f"# Cost advice: {report.name}", ""]
    if base is None:
        lines.append(f"No baseline run ({report.stopped or 'not run'}).")
        return "\n".join(lines) + "\n"
    lines += [
        f"Baseline `{report.baseline}`: score {base.score:.3f}, run cost ${base.cost_usd:.6f} "
        f"(models ${base.model_cost_usd:.6f}), "
        f"{base.n_passed} passed, {base.n_errors} errors. Keep score >= "
        f"{report.min_score_ratio:.2f} x baseline.",
        "",
        "| Step | Calls | Baseline $ | Advice | Score | Step $ delta | Latency delta s | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for advice in report.steps:
        score = "" if advice.score is None else f"{advice.score:.3f}"
        lines.append(
            f"| `{advice.step}` | {advice.baseline.calls} | {advice.baseline.cost_usd:.6f} | "
            f"{advice.model or 'keep'} | {score} | {_usd(advice.cost_delta_usd)} | "
            f"{advice.latency_delta_s:+.3f} | {advice.reason} |"
        )
    lines += ["", f"Mapping: `{report.mapping}`"]
    if report.note:
        lines.append(f"Advised one step at a time: `{report.advised}`; {report.note}.")
    if report.combined is not None:
        held = "keeps" if report.combined.keeps_score else "does NOT keep"
        lines.append(
            f"All together: score {report.combined.score:.3f}, run cost "
            f"${report.combined.cost_usd:.6f} ({held} the score)."
        )
    if report.max_spend_usd is not None:
        lines.append(f"Spent ${report.spent_usd:.6f} of ${report.max_spend_usd:.6f}.")
    if report.stopped:
        lines.append(f"Stopped early: {report.stopped}.")
    return "\n".join(lines) + "\n"
