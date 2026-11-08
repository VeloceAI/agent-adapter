"""Cost advisor: find a cheaper model per flow node or stage that keeps the eval score."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import Usage
from agent_adapter.cost.ledger import Ledger
from agent_adapter.cost.prices import Prices
from agent_adapter.evals._advisor_report import AdvisorReport, StepAdvice, StepUse, Trial
from agent_adapter.evals._spend import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    CappedModel,
    SpendCap,
    cap_graders,
)
from agent_adapter.evals._step_models import (
    ALL_STEPS,
    StepModels,
    apply_mapping,
    current_step,
    entry_step,
    step_of,
)
from agent_adapter.evals.cases import load_cases
from agent_adapter.evals.model import EvalCase
from agent_adapter.evals.runner import Graders, run_eval
from agent_adapter.exec.executor import Executor
from agent_adapter.runtime.current import current_context
from agent_adapter.runtime.sync import run_sync

logger = logging.getLogger(__name__)

TargetFactory = Callable[[LLMPort], object]
COMBINED = "combined"  # the step label of the run with every recommended model at once
_SIZE = Usage(input_tokens=1000, output_tokens=1000)  # orders candidates cheapest first
_TOLERANCE = 1e-9
_BILLED = frozenset({"llm", "api"})  # entry kinds that should carry a price


class _Recording(StepModels):
    """A `StepModels` that remembers which ledger entries (and steps) it answered."""

    def __init__(self, baseline: LLMPort, steps: Mapping[str, LLMPort]) -> None:
        super().__init__(baseline, steps)
        self.entry_ids: set[str] = set()

    def _route(self, request: LLMRequest) -> tuple[LLMPort, LLMRequest]:
        ctx = current_context()
        if ctx is not None and ctx.call_id:
            self.entry_ids.add(ctx.call_id)
        return super()._route(request)


class _Advisor:
    """One advisor session: shared cases, executor, prices and spend cap."""

    def __init__(
        self,
        factory: TargetFactory,
        cases: tuple[EvalCase, ...],
        graders: Graders,
        ports: Mapping[str, LLMPort],
        executor: Executor,
        table: Prices,
        cap: SpendCap,
        options: dict[str, Any],
    ) -> None:
        self.factory = factory
        self.cases = cases
        self.graders = graders
        self.ports = ports
        self.executor = executor
        self.table = table
        self.cap = cap
        self.options = options

    async def trial(self, step: str, model: str, mapping: Mapping[str, str]) -> Trial:
        """One eval run with `mapping` applied and the baseline everywhere else."""
        baseline = self.options["baseline"]
        router = _Recording(self.ports[baseline], {s: self.ports[m] for s, m in mapping.items()})
        ledger = Ledger(prices=self.table, max_runs=max(1000, 2 * len(self.cases)))
        report = await run_eval(
            self.factory(router),
            self.cases,
            self.graders,
            executor=self.executor.replace(ledger=ledger),
            concurrency=self.options["concurrency"],
            pass_at=self.options["pass_at"],
            name=self.options["name"],
            capture_content=False,
        )
        steps: dict[str, StepUse] = {}
        run_cost, unpriced = 0.0, 0
        for case in report.cases:
            for entry in ledger.entries(case.run_id):
                run_cost += entry.cost_usd or 0.0
                unpriced += entry.cost_usd is None and entry.kind in _BILLED
                if entry.kind != "llm" or entry.entry_id not in router.entry_ids:
                    continue
                key = entry_step(entry)
                used = steps.get(key, StepUse())
                steps[key] = StepUse(
                    calls=used.calls + 1,
                    cost_usd=used.cost_usd + (entry.cost_usd or 0.0),
                    latency_s=used.latency_s + entry.latency_ms / 1000,
                    unpriced=used.unpriced + (entry.cost_usd is None),
                )
        logger.info("advisor trial %s=%s: %d cases", step or "baseline", model, report.n_cases)
        return Trial(
            step=step,
            model=model,
            score=report.mean_score,
            n_passed=report.n_passed,
            n_errors=report.n_errors,
            cost_usd=run_cost,
            model_cost_usd=sum(use.cost_usd for use in steps.values()),
            unpriced=unpriced,
            latency_s=sum(use.latency_s for use in steps.values()),
            steps=dict(sorted(steps.items())),
            complete=not self.cap.stopped,
        )

    def keeps(self, trial: Trial, base: Trial) -> Trial:
        """`trial` with `keeps_score` set against the baseline."""
        floor = self.options["min_score_ratio"] * base.score - _TOLERANCE
        kept = trial.complete and trial.score >= floor and trial.n_errors <= base.n_errors
        return trial.model_copy(update={"keeps_score": kept})

    def order(self) -> list[str]:
        """Candidate names other than the baseline, cheapest list price first."""
        names = [name for name in self.ports if name != self.options["baseline"]]

        def price(name: str) -> float:
            cost = self.table.cost(self.ports[name].model, _SIZE)
            return math.inf if cost is None else cost

        return sorted(names, key=lambda name: (price(name), name))


def _use(trial: Trial, step: str) -> StepUse:
    """What `step` cost in `trial`; for `"*"` the whole run."""
    if step != ALL_STEPS:
        return trial.steps.get(step, StepUse())
    uses = trial.steps.values()
    return StepUse(
        calls=sum(u.calls for u in uses),
        cost_usd=trial.model_cost_usd,
        latency_s=trial.latency_s,
        unpriced=sum(u.unpriced for u in uses),
    )


def _advice(step: str, base: Trial, tried: Sequence[Trial], stopped: bool) -> StepAdvice:
    """The cheapest tried candidate for `step` that keeps the score and lowers the cost."""
    used = _use(base, step)
    if base.score <= 0:
        return StepAdvice(step=step, baseline=used, reason="baseline scores 0: no advice")
    if used.unpriced or base.unpriced:
        return StepAdvice(step=step, baseline=used, reason="baseline calls are unpriced")
    if base.cost_usd <= 0:
        return StepAdvice(step=step, baseline=used, reason="baseline costs 0: nothing to save")
    good = [
        t
        for t in tried
        if t.keeps_score and not t.unpriced and t.cost_usd < base.cost_usd - _TOLERANCE
    ]
    if not good:
        why = "stopped by the spend cap" if stopped else "no cheaper candidate keeps the score"
        return StepAdvice(step=step, baseline=used, reason=why)
    best = min(good, key=lambda t: (t.cost_usd, -t.score, t.model))
    mine = _use(best, step)
    return StepAdvice(
        step=step,
        baseline=used,
        model=best.model,
        score=best.score,
        cost_delta_usd=mine.cost_usd - used.cost_usd,
        run_cost_delta_usd=best.cost_usd - base.cost_usd,
        latency_delta_s=mine.latency_s - used.latency_s,
        reason="cheapest candidate that keeps the score",
    )


async def advise(
    target_factory: TargetFactory,
    cases: Sequence[EvalCase] | str | os.PathLike[str],
    graders: Graders,
    *,
    candidates: Mapping[str, LLMPort],
    baseline: str,
    per_step: bool = True,
    min_score_ratio: float = 0.98,
    prices: Prices | Mapping[str, Any] | None = None,
    max_spend: float | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    executor: Executor | None = None,
    concurrency: int = 4,
    pass_at: float = 1.0,
    verify: bool = True,
    name: str = "",
) -> AdvisorReport:
    """Run the evals on `baseline`, then each cheaper candidate per step; report the advice.

    `target_factory(model)` builds the eval target around the given port (pass it on as
    `FlowRunner(model=...)`, `ToolLoopTarget(model, ...)`, ...). Steps are the flow nodes and
    stages that called it. Each trial swaps one step's model and keeps the rest on the
    baseline; with `per_step=False` a candidate replaces every step at once. `prices` is a
    `Prices` or a table mapping (default: the executor ledger's, else the built-in table).
    `max_spend` (USD) caps every model call the advisor makes, `LLMJudge` graders included:
    each holds its worst-case cost (output limited to `max_output_tokens`) before it runs and
    settles the real cost after; the advisor stops early when a hold does not fit. `verify`
    reruns the evals with all advice at once; `report.mapping` holds the advice only when that
    run kept the score (else it is empty and `report.note` says why). Nothing is applied: pass
    `report.mapping` to `apply_mapping` yourself.

    Raises:
        ConfigError: a bad option, an unknown baseline, a non-port candidate, or, with
            `max_spend`, a candidate or judge model without a token price.
    """
    ports = _check(target_factory, candidates, baseline, min_score_ratio, max_spend)
    table = _prices(prices, executor)
    _check_tokens(max_output_tokens)
    cap = SpendCap(max_spend, table, max_output_tokens=max_output_tokens)
    graders, judges = cap_graders(graders, cap)
    cap.check_priced({**ports, **judges})
    capped = {key: CappedModel(port, cap) for key, port in ports.items()}
    base_exec = executor if executor is not None else Executor(Registry())
    options = {
        "baseline": baseline,
        "min_score_ratio": float(min_score_ratio),
        "concurrency": concurrency,
        "pass_at": pass_at,
        "name": name or "advisor",
    }
    session = _Advisor(
        target_factory,
        _load(cases),
        graders,
        capped,
        base_exec.replace(cache=None),  # a cached reply would hide a candidate's real cost
        table,
        cap,
        options,
    )
    return await _run(session, per_step=bool(per_step), verify=bool(verify), max_spend=max_spend)


async def _run(
    session: _Advisor, *, per_step: bool, verify: bool, max_spend: float | None
) -> AdvisorReport:
    options, cap = session.options, session.cap
    head: dict[str, Any] = {
        "name": options["name"],
        "baseline": options["baseline"],
        "min_score_ratio": options["min_score_ratio"],
        "max_spend_usd": max_spend,
    }
    base = await session.trial("", options["baseline"], {})
    if not base.complete:
        return AdvisorReport(**head, baseline_trial=None, spent_usd=cap.spent_usd, stopped="budget")
    steps = list(base.steps) if per_step else [ALL_STEPS]
    trials: list[Trial] = []
    for step in steps:
        for model in session.order():
            if cap.stopped:
                break
            trials.append(session.keeps(await session.trial(step, model, {step: model}), base))
    advice = tuple(
        _advice(step, base, [t for t in trials if t.step == step], cap.stopped) for step in steps
    )
    mapping = {a.step: a.model for a in advice if a.model}
    combined = await _combined(session, mapping, trials, base, verify)
    return AdvisorReport(
        **head,
        baseline_trial=base,
        steps=advice,
        trials=tuple(trials),
        combined=combined,
        verified=bool(mapping) and combined is not None and combined.keeps_score,
        note=_note(mapping, combined, verify),
        spent_usd=cap.spent_usd,
        stopped="budget" if cap.stopped else "",
    )


async def _combined(
    session: _Advisor, mapping: dict[str, str], trials: list[Trial], base: Trial, verify: bool
) -> Trial | None:
    """The run with every recommended model at once; the trial itself for a single step."""
    if not mapping:
        return None
    if len(mapping) == 1:
        ((step, model),) = mapping.items()
        return next(t for t in trials if t.step == step and t.model == model)
    if not verify or session.cap.stopped:
        return None
    return session.keeps(await session.trial(COMBINED, COMBINED, mapping), base)


def _note(mapping: dict[str, str], combined: Trial | None, verify: bool) -> str:
    """Why `mapping` is empty although steps were advised; "" otherwise."""
    if not mapping or (combined is not None and combined.keeps_score):
        return ""
    if combined is not None:
        return "not verified: the advice together did not keep the score"
    if not verify:
        return "not verified: verify=False"
    return "not verified: the spend cap stopped before the combined run"


def advise_sync(
    target_factory: TargetFactory, cases: Any, graders: Graders, **options: Any
) -> AdvisorReport:
    """`advise` for sync code (scripts, tests).

    Raises:
        ConfigError: as `advise`.
    """
    report: AdvisorReport = run_sync(advise(target_factory, cases, graders, **options))
    return report


def _check(
    factory: object,
    candidates: object,
    baseline: object,
    ratio: object,
    max_spend: object,
) -> Mapping[str, LLMPort]:
    if not callable(factory):
        raise ConfigError("target_factory must be a function of the model")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ConfigError("candidates must be a non-empty mapping of name -> model port")
    for key, port in candidates.items():
        if not isinstance(key, str) or not key:
            raise ConfigError("candidate names must be non-empty strings")
        if not callable(getattr(port, "generate", None)) or not isinstance(
            getattr(port, "model", None), str
        ):
            raise ConfigError(f"candidate {key!r} is not an LLMPort: {type(port).__name__}")
    if not isinstance(baseline, str) or baseline not in candidates:
        raise ConfigError(f"baseline must name a candidate, got {baseline!r}")
    if isinstance(ratio, bool) or not isinstance(ratio, int | float) or not 0 < ratio <= 1:
        raise ConfigError(f"min_score_ratio must be in (0, 1], got {ratio!r}")
    if max_spend is not None and (
        isinstance(max_spend, bool)
        or not isinstance(max_spend, int | float)
        or not math.isfinite(max_spend)
        or max_spend <= 0
    ):
        raise ConfigError(f"max_spend must be a number > 0 or None, got {max_spend!r}")
    return candidates


def _check_tokens(tokens: object) -> None:
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ConfigError(f"max_output_tokens must be an int >= 1, got {tokens!r}")


def _prices(prices: Prices | Mapping[str, Any] | None, executor: Executor | None) -> Prices:
    if isinstance(prices, Prices):
        return prices
    if isinstance(prices, Mapping):
        return Prices.empty("advisor").merged(prices)
    if prices is not None:
        raise ConfigError(f"prices must be a Prices or a mapping, got {type(prices).__name__}")
    found = getattr(executor.ledger, "prices", None) if executor is not None else None
    return found if isinstance(found, Prices) else Prices.default()


def _load(cases: Sequence[EvalCase] | str | os.PathLike[str]) -> tuple[EvalCase, ...]:
    if isinstance(cases, str | os.PathLike):
        return load_cases(cases)
    return tuple(cases)


__all__ = [
    "COMBINED",
    "AdvisorReport",
    "StepAdvice",
    "StepModels",
    "StepUse",
    "Trial",
    "advise",
    "advise_sync",
    "apply_mapping",
    "current_step",
    "entry_step",
    "step_of",
]
