"""`CallableHost` optimises any agent you can call; `NullModel` stands in for its own model."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, cast

from agent_adapter.core.context import ToolContext
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import ConfigError, OptimiserError
from agent_adapter.core.messages import LLMRequest, LLMResponse, StreamDone, StreamEvent
from agent_adapter.core.ports import LLMPort
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser._transcript import TRANSCRIPT_KEY, TranscriptRecorder
from agent_adapter.optimiser.hosts import Judge, _unique_cases, failed_run, user_text
from agent_adapter.optimiser.model import CaseRun, EvalCase, HarnessTree, SurfaceCaps
from agent_adapter.optimiser.surface import DEFAULT_CAPS, parse_path

logger = logging.getLogger(__name__)

RunCase = Callable[[HarnessTree, EvalCase, LLMPort], Awaitable[str | CaseRun]]
_NO_MODEL = "the host's agent runs without --model, but it called the model it was given"


class NullModel:
    """The agent model for hosts whose agent brings its own: every call raises `ConfigError`."""

    provider = "none"
    model = "none"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Never answers.

        Raises:
            ConfigError: always.
        """
        raise ConfigError(_NO_MODEL)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Never streams.

        Raises:
            ConfigError: always.
        """
        yield StreamDone(response=await self.generate(request))

    def cache_key_fields(self) -> dict[str, Any]:
        return {}


class _Recorder(TranscriptRecorder):
    """Also counts the tool calls the model asked for, so tool edits are observable."""

    def add_reply(self, stage: str, prompt: str, response: LLMResponse) -> None:
        super().add_reply(stage, prompt, response)
        with self._lock:
            for call in response.tool_calls:
                self.counts[f"tool_calls.{call.name}"] += 1


class CallableHost:
    """Optimise any agent you can call: `run_case` reads the tunable texts from `tree`.

    `surface` maps component paths to their base texts; only those paths can be edited.
    `required` paths may not be deleted or emptied. Model calls made through the `model`
    passed to `run_case` are metered and recorded; `withhold` names tools whose call
    arguments transcripts leave out (pass your confirm tools). Report `metrics["cost_usd"]`
    in a returned `CaseRun` when your agent pays for its own model.
    """

    def __init__(
        self,
        run_case: RunCase,
        *,
        surface: Mapping[str, str],
        cases: Sequence[EvalCase],
        judge: Judge | None = None,
        domain: str = "default",
        caps: Mapping[str, int] | None = None,
        required: Iterable[str] = (),
        validate: Callable[[HarnessTree], list[str]] | None = None,
        withhold: Iterable[str] = (),
    ) -> None:
        if not callable(run_case):
            raise ConfigError("run_case must be callable")
        self._surface = _checked_surface(surface)
        self._max_chars = _checked_caps(caps)
        self.required = tuple(required)
        missing = [p for p in self.required if p not in self._surface]
        if missing:
            raise ConfigError(f"required paths not on the surface: {missing}")
        self.run_case = run_case
        self.judge = judge
        self.domain = domain
        self.withhold = frozenset(withhold)
        self._validate = validate
        self._cases = _unique_cases(cases)

    def surface(self) -> HarnessTree:
        return HarnessTree(files=dict(self._surface))

    def caps(self) -> SurfaceCaps:
        kinds = frozenset(parse_path(path).kind for path in self._surface)
        return SurfaceCaps(max_chars=dict(self._max_chars), kinds=kinds)

    def validate(self, tree: HarnessTree) -> list[str]:
        problems = [
            f"{p}: not on this host's surface" for p in tree.files if p not in self._surface
        ]
        for path in self.required:
            if path not in tree.files:
                problems.append(f"{path}: a required path cannot be deleted")
            elif not tree.files[path].strip():
                problems.append(f"{path}: a required path cannot be empty")
        if self._validate is not None:
            problems += [str(p) for p in self._validate(tree)]
        return sorted(problems)

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    async def run(
        self, tree: HarnessTree, case: EvalCase, *, run_id: str, model: LLMPort
    ) -> CaseRun:
        recorder = _Recorder(withhold=self.withhold)
        try:
            out = await self.run_case(tree, case, _recording(model, recorder, run_id))
        except Exception as exc:
            if is_control_flow(exc):
                raise
            failed = failed_run(case, exc)
            transcript = "\n".join(filter(None, (recorder.text(), failed.transcript)))
            return failed.model_copy(update={"transcript": transcript})
        if isinstance(out, CaseRun):
            return _with_recorded(out, recorder)
        if not isinstance(out, str):
            return failed_run(case, ConfigError(f"run_case returned {type(out).__name__}"))
        lines = [f"[USER] {user_text(case)}", recorder.text(), f"[OUTPUT] {out}"]
        return CaseRun(
            case_id=case.id,
            label="",
            ok=True,
            delivered=bool(out.strip()),
            output=out,
            transcript="\n".join(filter(None, lines)),
            metrics={key: float(n) for key, n in recorder.counts.items()},
        )


def _with_recorded(run: CaseRun, recorder: _Recorder) -> CaseRun:
    """`run` plus the recorded counts (its own metrics win) and, if it has none, the transcript."""
    recorded = recorder.text()
    if not recorder.counts and not recorded:
        return run
    metrics = {**{key: float(n) for key, n in recorder.counts.items()}, **run.metrics}
    transcript = run.transcript or recorded
    return run.model_copy(update={"metrics": metrics, "transcript": transcript})


def _recording(model: LLMPort, recorder: TranscriptRecorder, run_id: str) -> LLMPort:
    """`model` with the recorder in its context, so each model reply joins the transcript."""
    if not isinstance(model, InstrumentedModel):
        return model
    ctx = model.ctx or ToolContext(run_id=run_id)
    ctx = dataclasses.replace(ctx, extras={**ctx.extras, TRANSCRIPT_KEY: recorder})
    return cast("LLMPort", model.bind(ctx))


def _checked_surface(surface: Mapping[str, str]) -> dict[str, str]:
    if not surface:
        raise ConfigError("surface is empty: name at least one tunable text")
    for path, text in surface.items():
        try:
            parse_path(path)
        except OptimiserError as exc:
            raise ConfigError(f"surface: {exc}") from exc
        if not isinstance(text, str):
            raise ConfigError(f"surface: {path} must be text, got {type(text).__name__}")
    return dict(surface)


def _checked_caps(caps: Mapping[str, int] | None) -> dict[str, int]:
    merged = dict(DEFAULT_CAPS)
    for pattern, cap in (caps or {}).items():
        if pattern not in DEFAULT_CAPS:
            raise ConfigError(
                f"caps: unknown pattern {pattern!r}; use one of {sorted(DEFAULT_CAPS)}"
            )
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ConfigError(f"caps: {pattern} must be an int >= 1, got {cap!r}")
        merged[pattern] = cap
    return merged
