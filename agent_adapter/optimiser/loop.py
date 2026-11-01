"""The optimiser loop: probe the base, propose, probe, compose, probe finals, select, reflect."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

from agent_adapter.cache.memory import MemoryCache
from agent_adapter.core.errors import BudgetExceeded, OptimiserError
from agent_adapter.core.ports import CachePort, LLMPort
from agent_adapter.cost.ledger import Ledger
from agent_adapter.cost.sinks import CallbackSink
from agent_adapter.hooks.base import Hook
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser._budget import Meter, Stopped, check_config, check_host
from agent_adapter.optimiser._outcome import build_result
from agent_adapter.optimiser._reflection import reflect_on_run
from agent_adapter.optimiser._select import select_final
from agent_adapter.optimiser._stages import StageRunner
from agent_adapter.optimiser._state import (
    ComposeRecord,
    ReflectionRecord,
    RunState,
    SelectionRecord,
    fingerprint,
    load_state,
)
from agent_adapter.optimiser._transcript import TranscriptHook
from agent_adapter.optimiser.compose import compose
from agent_adapter.optimiser.hosts import Host
from agent_adapter.optimiser.lessons import LessonsStore, lessons_text
from agent_adapter.optimiser.model import (
    BudgetUsage,
    Candidate,
    OptimiserConfig,
    OptimiseResult,
    ProbeReport,
)
from agent_adapter.optimiser.report import write_outputs
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

T = TypeVar("T")
_MAX_RUNS = 1_000_000  # ledger runs kept in memory: one per case run and stage
_DIMS: tuple[tuple[str, Literal["procedure", "knowledge"]], ...] = (
    ("proc", "procedure"),
    ("know", "knowledge"),
)


async def optimise(
    host: Host,
    config: OptimiserConfig,
    *,
    proposer: LLMPort,
    model: LLMPort,
    composer: LLMPort | None = None,
    cache: CachePort | None = None,
    ledger: Ledger | None = None,
) -> OptimiseResult:
    """Run Propose -> Probe -> Compose over the host's cases and write a reviewable patch.

    `model` is the host's agent model; `proposer` writes edits; `composer` (default: the
    proposer) merges conflicts, reflects and selects without a judge. Every call goes through
    an `InstrumentedModel` with the response cache (namespace `config.namespace`), the
    ledger and a budget hook. Outputs land in `config.out_dir`; nothing is applied to the
    host. A run with the same inputs and `out_dir` resumes from `state.json`.

    Raises:
        ConfigError: invalid config values.
        OptimiserError: the host has no cases, duplicate case ids or an invalid surface; the
            lessons file is invalid; a model error is fatal (bad key, rejected request); no base
            case ran; or every proposer of a stage failed on a model error.
    """
    check_config(config)
    run = _Run(
        host, config, proposer=proposer, model=model, composer=composer, cache=cache, ledger=ledger
    )
    return await run.execute()


class _Run(StageRunner):
    """State of one `optimise` call."""

    def __init__(
        self,
        host: Host,
        config: OptimiserConfig,
        *,
        proposer: LLMPort,
        model: LLMPort,
        composer: LLMPort | None,
        cache: CachePort | None,
        ledger: Ledger | None,
    ) -> None:
        self.host, self.config = host, config
        self.base_tree, self.caps = host.surface(), host.caps()
        self.cases = tuple(host.cases())
        self.case_map = {case.id: case for case in self.cases}
        check_host(host, self.base_tree, self.caps, self.cases)
        self.out_dir = Path(config.out_dir)
        writer = composer or proposer
        models = {
            name: f"{port.provider}:{port.model}"
            for name, port in (("model", model), ("proposer", proposer), ("composer", writer))
        }
        key = fingerprint(
            tree=self.base_tree,
            caps=self.caps,
            cases=list(self.cases),
            config=config,
            domain=host.domain,
            models=models,
        )
        self.state = load_state(self.out_dir, key) or RunState(
            fingerprint=key, run_id=new_id("opt_")
        )
        self.run_id = self.state.run_id
        self.probes: dict[str, ProbeReport] = dict(self.state.probes)
        self.finals: list[str] = []
        self.stopped = ""
        sinks = [CallbackSink(ledger.record)] if ledger is not None else []
        self.ledger = Ledger(
            prices=ledger.prices if ledger else None, sinks=sinks, max_runs=_MAX_RUNS
        )
        self.cache = cache if cache is not None else MemoryCache()
        self.meter = Meter(
            config.budget,
            self.ledger,
            restored_usd=self.state.usd,
            restored_wall=self.state.wall_s,
        )
        guards = self.meter.guards()
        self.agent = self._wrap(model, [*guards, TranscriptHook()])
        self.proposer = self._wrap(proposer, guards)
        self.composer = self._wrap(writer, guards)

    def _wrap(self, port: LLMPort, hooks: Sequence[Hook]) -> InstrumentedModel:
        return InstrumentedModel(
            port, hooks=hooks, ledger=self.ledger, cache=self.cache, namespace=self.config.namespace
        )

    # --- the run -------------------------------------------------------------------------

    async def execute(self) -> OptimiseResult:
        lessons = lessons_text(
            self.config.lessons_path,
            domain=self.host.domain,
            tree=self.base_tree,
            limit=self.config.max_lessons,
        )
        config = self.config
        logger.info("optimiser %s: %d cases, k=%d", self.run_id, len(self.cases), config.k)
        deadline = asyncio.timeout(self.meter.wall_left())  # cancels a stage in flight
        failure: OptimiserError | None = None
        try:
            async with deadline:
                await self._stages(lessons)
        except Stopped as stop:
            self.stopped = stop.reason
        except OptimiserError as exc:
            failure = exc  # reported in the outputs, then raised
        except TimeoutError:
            if not deadline.expired():
                raise
            self.stopped = "max_wall_s"
        if self.stopped:
            logger.warning("optimiser %s stopped early: %s", self.run_id, self.stopped)
        result = self._result(str(failure) if failure else "")
        self._save()
        chosen = self.state.candidates[result.chosen].tree if result.chosen else self.base_tree
        write_outputs(
            self.out_dir, result, run_id=self.run_id, chosen_tree=chosen, cases=self.case_map
        )
        if failure is not None:
            raise failure
        logger.info("optimiser %s finished: chosen %s", self.run_id, result.chosen or "KEEP_BASE")
        return result

    async def _stages(self, lessons_text: str) -> None:
        base = self._base_candidate()
        base_report = await self._probe(base)
        if self.config.mode == "parallel":
            await self._parallel(base, base_report, lessons_text)
        else:
            await self._sequential(base, base_report, lessons_text)
        await self._select(base_report)
        await self._reflect()

    def _base_candidate(self) -> Candidate:
        base = Candidate(label="base", dim="base", edits=(), tree=self.base_tree, parent_label="")
        self.state.candidates.setdefault("base", base)
        return base

    async def _parallel(self, base: Candidate, base_report: ProbeReport, lessons: str) -> None:
        k = self.config.k
        stages = [
            functools.partial(self._propose, f"{prefix}{i}", dim, base, base_report, i, lessons, "")
            for prefix, dim in _DIMS
            for i in range(1, k + 1)
        ]
        candidates = await self._proposed(stages)
        await self._probe_stage(candidates, "candidates")
        pool = [(c, self.probes[c.label]) for c in candidates]
        finals = []
        for i in range(1, k + 1):
            finals.append(await self._compose(f"agg{i}", base, base_report, pool, drop=i - 1))
            self.finals.append(finals[-1].label)
        await self._probe_stage(finals, "finals")

    async def _sequential(self, base: Candidate, base_report: ProbeReport, lessons: str) -> None:
        parent, ref, history = base, base_report, ""
        for t in range(1, self.config.k + 1):
            stages = [
                functools.partial(
                    self._propose, f"{prefix}{t}", dim, parent, ref, t, lessons, history
                )
                for prefix, dim in _DIMS
            ]
            candidates = await self._proposed(stages)
            await self._probe_stage(candidates, f"round {t}")
            pool = [(c, self.probes[c.label]) for c in candidates]
            checkpoint = await self._compose(f"c{t}", parent, ref, pool, drop=0)
            self.finals.append(checkpoint.label)
            await self._probe_stage([checkpoint], f"checkpoint {checkpoint.label}")
            ref, parent = self.probes[checkpoint.label], checkpoint
            lines = self.state.composes[checkpoint.label].decisions
            history += f"Round {t} -> {checkpoint.label}:\n" + "".join(
                f"- {d.kind} {d.path}: {d.evidence}\n" for d in lines
            )

    async def _compose(
        self,
        label: str,
        parent: Candidate,
        ref: ProbeReport,
        pool: Sequence[tuple[Candidate, ProbeReport]],
        *,
        drop: int,
    ) -> Candidate:
        if label in self.state.candidates and label in self.state.composes:
            return self.state.candidates[label]
        self._gate()
        stage_run = f"{self.run_id}/{label}"
        try:
            outcome = await compose(
                label,
                parent,
                ref,
                pool,
                caps=self.caps,
                host=self.host,
                mode=self.config.compose_mode,
                min_gain=self.config.min_gain,
                seed=self._seed(label),
                composer=self._bound(self.composer, stage_run),
                drop=drop,
            )
        except BudgetExceeded as exc:
            raise Stopped("max_usd") from exc
        candidate = outcome.candidate.model_copy(
            update={"cost_usd": self.ledger.total_usd(stage_run)}
        )
        assessed = {
            a.candidate.label: a.evidence()
            + (f"; dropped: {a.reason}" if a.reason else "; admitted")
            for a in outcome.assessments
        }
        self.state.candidates[label] = candidate
        self.state.composes[label] = ComposeRecord(
            decisions=outcome.decisions, merge_calls=outcome.merge_calls, assessed=assessed
        )
        self.state.merge_calls += outcome.merge_calls
        self._save()
        return candidate

    async def _select(self, base_report: ProbeReport) -> None:
        if self.state.selection is not None:
            return
        self._gate()
        finals = [(self.state.candidates[label], self.state.probes[label]) for label in self.finals]
        try:
            choice = await select_final(
                base_report,
                self.base_tree,
                finals,
                judged=self.host.judge is not None,
                min_gain=self.config.min_gain,
                seed=self._seed("select"),
                cases=self.case_map,
                selector=self._bound(self.composer, f"{self.run_id}/select"),
                budget=self.config.evidence_chars,
            )
        except BudgetExceeded as exc:
            raise Stopped("max_usd") from exc
        self.state.selection = SelectionRecord(
            chosen=choice.chosen, why=choice.why, calls=choice.calls
        )
        self._save()

    async def _reflect(self) -> None:
        if not self.config.lessons_path:
            return
        if self.state.reflection is not None:
            # Saved before the store write, so a crash in between stores it now (ids dedupe).
            if self.state.reflection.lessons:
                LessonsStore(self.config.lessons_path).add(self.state.reflection.lessons)
            return
        proposed = [
            c for c in self.state.candidates.values() if c.dim in ("procedure", "knowledge")
        ]
        if not any(c.edits for c in proposed):
            self.state.reflection = ReflectionRecord(lessons=[], why="nothing was proposed")
            self._save()
            return
        self._gate()
        try:
            lesson, why = await reflect_on_run(
                self.state,
                self._bound(self.composer, f"{self.run_id}/reflect"),
                base=self.base_tree,
                domain=self.host.domain,
                run=str(self.out_dir),
                limit=self.config.evidence_chars,
            )
        except BudgetExceeded as exc:
            raise Stopped("max_usd") from exc
        self.state.reflection = ReflectionRecord(
            lessons=[lesson] if lesson else [], why=why, calls=1
        )
        self._save()
        if lesson is not None:
            LessonsStore(self.config.lessons_path).add([lesson])

    # --- result --------------------------------------------------------------------------

    def _result(self, failure: str) -> OptimiseResult:
        usage = BudgetUsage(
            usd=self.meter.spent(),
            case_runs=self.state.case_runs,
            wall_s=self.meter.wall(),
            proposer_calls=self.state.proposer_calls,
            merge_calls=self.state.merge_calls,
            stopped=self.stopped,
        )
        return build_result(
            self.state,
            probes=self.probes,
            finals=self.finals,
            base_tree=self.base_tree,
            summary=self.ledger.summary(),
            usage=usage,
            config=self.config,
            domain=self.host.domain,
            n_cases=len(self.cases),
            failure=failure,
        )
