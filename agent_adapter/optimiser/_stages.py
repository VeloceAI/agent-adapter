"""The run's propose and probe stages, shared by both loop modes (a base of `loop._Run`)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar, cast

from agent_adapter.cache.keys import digest
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import BudgetExceeded, OptimiserError
from agent_adapter.core.ports import LLMPort
from agent_adapter.cost.ledger import Ledger
from agent_adapter.exec.executor import Executor
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser._budget import Meter, Stopped
from agent_adapter.optimiser._fatal import check_probe, check_proposers, check_stage
from agent_adapter.optimiser._state import RunState, save_state
from agent_adapter.optimiser.evidence import build_evidence
from agent_adapter.optimiser.hosts import Host
from agent_adapter.optimiser.model import (
    Candidate,
    EvalCase,
    OptimiserConfig,
    ProbeReport,
    SurfaceCaps,
)
from agent_adapter.optimiser.probe import probe
from agent_adapter.optimiser.prompts import propose_system
from agent_adapter.optimiser.propose import propose

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StageRunner:
    """Proposing and probing with the run's state, budget and resume rules."""

    host: Host
    config: OptimiserConfig
    caps: SurfaceCaps
    cases: tuple[EvalCase, ...]
    case_map: dict[str, EvalCase]
    out_dir: Path
    state: RunState
    run_id: str
    probes: dict[str, ProbeReport]
    ledger: Ledger
    meter: Meter
    agent: InstrumentedModel
    proposer: InstrumentedModel
    composer: InstrumentedModel

    def _bound(self, model: InstrumentedModel, stage_run: str) -> LLMPort:
        """`model` bound to the stage's own run id (its ledger entries and cost)."""
        return cast("LLMPort", model.bind(ToolContext(run_id=stage_run)))

    def _gate(self, case_runs: int = 0) -> None:
        self.meter.gate(done_case_runs=self.state.case_runs, case_runs=case_runs)

    def _save(self) -> None:
        self.state.usd, self.state.wall_s = self.meter.spent(), self.meter.wall()
        save_state(self.out_dir, self.state)

    def _seed(self, label: str) -> int:
        return int(digest([self.config.seed, label])[:8], 16)

    async def _proposed(
        self, stages: Sequence[Callable[[], Awaitable[Candidate]]]
    ) -> list[Candidate]:
        """Run proposer stages; if every one failed on a model error, forget them and stop."""
        candidates = await self._many(stages)
        try:
            check_proposers(candidates)
        except OptimiserError:
            for candidate in candidates:  # so a resume proposes again
                if self.state.candidates.pop(candidate.label, None) is not None:
                    self.state.proposer_calls -= 1
            self._save()
            raise
        return candidates

    async def _probe_stage(self, candidates: Sequence[Candidate], stage: str) -> None:
        """Probe each candidate; if every case run failed on a model error, forget and stop."""
        for candidate in candidates:
            await self._probe(candidate)
        try:
            check_stage([self.probes[c.label] for c in candidates], stage)
        except OptimiserError:
            for candidate in candidates:  # so a resume probes again
                if self.state.probes.pop(candidate.label, None) is not None:
                    self.state.case_runs -= len(self.cases)
            self._save()
            raise

    async def _many(self, stages: Sequence[Callable[[], Awaitable[T]]]) -> list[T]:
        return await Executor().parallel(*stages, concurrency=self.config.rollout_concurrency)

    # --- stages --------------------------------------------------------------------------

    async def _probe(self, candidate: Candidate) -> ProbeReport:
        label = candidate.label
        if label in self.state.probes:
            return self.state.probes[label]
        self._gate(len(self.cases))
        logger.info("optimiser %s: probing %s", self.run_id, label)
        report = await probe(
            self.host,
            candidate.tree,
            label,
            model=self.agent,
            run_prefix=self.run_id,
            ledger=self.ledger,
            concurrency=self.config.rollout_concurrency,
            case_timeout_s=self.config.case_timeout_s,
        )
        self.probes[label] = report
        cut = any(r.error.startswith("budget-exceeded") for r in report.runs)
        if cut or self.meter.over_usd():
            self.state.case_runs += len(self.cases)
            self._save()  # an interrupted probe is reported but never restored
            raise Stopped("max_usd")
        check_probe(report)  # a fatal probe is reported, not counted, and redone on resume
        self.state.case_runs += len(self.cases)
        self.state.probes[label] = report
        self._save()
        return report

    async def _propose(
        self,
        label: str,
        dim: Literal["procedure", "knowledge"],
        parent: Candidate,
        parent_report: ProbeReport,
        index: int,
        lessons: str,
        history: str,
    ) -> Candidate:
        if label in self.state.candidates:
            return self.state.candidates[label]
        self._gate()
        evidence = build_evidence(
            tree=parent.tree,
            cases=self.case_map,
            report=parent_report,
            obs_trunc=self.config.obs_trunc,
            budget=self.config.evidence_chars,
            show_scores=self.config.show_scores_to_proposer,
            lessons_text=lessons,
            history=history,
        )
        system = propose_system(
            dim,
            caps=self.caps,
            index=index,
            k=self.config.k,
            sequential=self.config.mode == "sequential",
        )
        stage_run = f"{self.run_id}/{label}"
        try:
            candidate = await propose(
                self._bound(self.proposer, stage_run),
                label=label,
                dim=dim,
                parent=parent,
                caps=self.caps,
                host=self.host,
                system=system,
                evidence=evidence,
                seed=self._seed(label),
            )
        except BudgetExceeded as exc:
            raise Stopped("max_usd") from exc
        candidate = candidate.model_copy(update={"cost_usd": self.ledger.total_usd(stage_run)})
        self.state.candidates[label] = candidate
        self.state.proposer_calls += 1
        self._save()
        return candidate
