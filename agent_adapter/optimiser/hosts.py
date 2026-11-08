"""The host contract and the two built-in hosts: a tool loop over a registry, and a flow."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_adapter.cache.layout import PromptLayout
from agent_adapter.core.control import is_control_flow
from agent_adapter.core.errors import FlowDefinitionError, OptimiserError
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ErrorInfo
from agent_adapter.evals._text import user_text as user_text
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.tool_loop import run_tool_loop
from agent_adapter.flow.runner import FlowRunner
from agent_adapter.flow.spec import Flow
from agent_adapter.hooks.base import Hook
from agent_adapter.optimiser._hook_surface import HookSurface, parsed_or_none
from agent_adapter.optimiser._transcript import (
    TRANSCRIPT_KEY,
    TranscriptRecorder,
    hook_metrics,
    loop_metrics,
    render_messages,
)
from agent_adapter.optimiser.apply import layout_with_hints
from agent_adapter.optimiser.model import CaseRun, EvalCase, HarnessTree, Score, SurfaceCaps
from agent_adapter.optimiser.surface import DEFAULT_CAPS, parse_path
from agent_adapter.tracing.sinks import FanoutSink

logger = logging.getLogger(__name__)

Judge = Callable[[EvalCase, CaseRun], Awaitable[Score | None]]


@runtime_checkable
class Host(Protocol):
    """What the optimiser needs from an application: its harness, its cases, and a way to run."""

    domain: str
    judge: Judge | None

    def surface(self) -> HarnessTree: ...
    def caps(self) -> SurfaceCaps: ...
    def validate(self, tree: HarnessTree) -> list[str]: ...  # [] = ok
    def cases(self) -> Sequence[EvalCase]: ...
    async def run(
        self, tree: HarnessTree, case: EvalCase, *, run_id: str, model: LLMPort
    ) -> CaseRun: ...  # applies the tree in memory; errors become CaseRun(ok=False, error=...)


def apply_tree(registry: Registry, tree: HarnessTree) -> Registry:
    """A copy of `registry` with the tree's descriptions and prompt hints; tools without paths stay.

    Raises:
        OptimiserError: a tool path names a tool the registry does not hold.
    """
    edits: dict[str, dict[str, str]] = {}
    for path, text in tree.files.items():
        if path.startswith("tool/"):
            parsed = parse_path(path)
            if parsed.name not in registry:
                raise OptimiserError(f"{path}: no tool named {parsed.name!r}")
            edits.setdefault(parsed.name, {})[parsed.field] = text
    copy = Registry()
    specs = []
    for spec in registry:
        fields = edits.get(spec.name)
        if fields is not None:
            spec = dataclasses.replace(
                spec,
                description=fields.get("description", spec.description),
                prompt_hint=fields.get("prompt_hint") or None,
            )
        specs.append(spec)
    copy.add_many(specs)
    return copy


def _unique_cases(cases: Sequence[EvalCase]) -> tuple[EvalCase, ...]:
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise OptimiserError(f"duplicate case ids: {duplicates}")
    return tuple(cases)


def failed_run(case: EvalCase, exc: BaseException) -> CaseRun:
    """A `CaseRun` for a run that raised; only the error kind and safe message are kept."""
    info = ErrorInfo.from_exception(exc)
    logger.warning("case %s failed: %s", case.id, info.kind)
    error = f"{info.kind}: {info.message}"
    return CaseRun(
        case_id=case.id,
        label="",
        ok=False,
        delivered=False,
        output="",
        transcript=f"[ERROR] {error}",
        error=error,
    )


class RegistryHost:
    """Runs `run_tool_loop` over a registry; edits descriptions, hints, prompts and hook configs.

    `hooks` run on tool calls after the executor's own hooks; tunable ones (with a
    `config_schema()`) are on the surface, safety hooks only with `allow_safety_tuning=True`.
    Put guardrails on `executor=`, which is never tuned. Tool hints are rendered into the
    system prompt.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        layout: PromptLayout,
        cases: Sequence[EvalCase],
        judge: Judge | None = None,
        hooks: Sequence[Hook] = (),
        domain: str = "default",
        max_steps: int = 8,
        executor: Executor | None = None,
        allow_safety_tuning: bool = False,
    ) -> None:
        self.registry = registry
        self.layout = layout
        self.judge = judge
        self.hooks = tuple(hooks)
        self.domain = domain
        self.max_steps = max_steps
        self.executor = executor
        self._cases = _unique_cases(cases)
        self._hooks = HookSurface(self.hooks, allow_safety_tuning=allow_safety_tuning)

    def surface(self) -> HarnessTree:
        files = {"prompt/system": self.layout.static}
        files.update({f"prompt/layout/{k}": v for k, v in self.layout.sections.items()})
        for spec in self.registry:
            files[f"tool/{spec.name}/description"] = spec.description
            if spec.prompt_hint:
                files[f"tool/{spec.name}/prompt_hint"] = spec.prompt_hint
        files.update(self._hooks.files())
        return HarnessTree(files=files)

    def caps(self) -> SurfaceCaps:
        return SurfaceCaps(
            max_chars=dict(DEFAULT_CAPS),
            kinds=frozenset({"tool", "hook", "prompt"}),
            hook_schemas=self._hooks.schemas(),
        )

    def validate(self, tree: HarnessTree) -> list[str]:
        problems = self._hooks.problems(tree)
        for spec in self.registry:
            path = f"tool/{spec.name}/description"
            if path not in tree.files:
                problems.append(f"{path}: a tool description cannot be deleted")
            elif not tree.files[path].strip():
                problems.append(f"{path}: a tool description cannot be empty")
        for path in tree.files:
            parsed = parsed_or_none(path)
            if parsed is not None and parsed.kind == "tool" and parsed.name not in self.registry:
                problems.append(f"{path}: no tool named {parsed.name!r}")
        return sorted(problems)

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    def layout_for(self, tree: HarnessTree, registry: Registry) -> PromptLayout:
        """The prompt layout the tree describes, with tool hints as a final section."""
        hints = [(s.name, s.prompt_hint) for s in registry if s.prompt_hint]
        return layout_with_hints(self.layout, tree, hints)

    async def run(
        self, tree: HarnessTree, case: EvalCase, *, run_id: str, model: LLMPort
    ) -> CaseRun:
        executor: Executor | None = None
        try:
            registry = apply_tree(self.registry, tree)
            hooks = self._hooks.applied(tree)
            base = self.executor or Executor(registry)
            executor = base.replace(registry=registry, hooks=(*base.hooks, *hooks))
            messages = self.layout_for(tree, registry).messages(user_text(case))
            result = await run_tool_loop(
                model,
                registry,
                messages,
                max_steps=self.max_steps,
                executor=executor,
                ctx=executor.context(run_id=run_id),
            )
        except Exception as exc:
            if is_control_flow(exc):
                raise
            return failed_run(case, exc)
        finally:
            if executor is not None:
                executor.end_run(run_id)
        withhold = {s.name for s in registry if s.confirm is not False}
        output = result.final.text
        pending = result.finish == "pending"
        return CaseRun(
            case_id=case.id,
            label="",
            ok=not pending,
            delivered=result.finish == "stop" and bool(output.strip()),
            output=output,
            transcript=render_messages(result.messages, withhold=withhold),
            metrics={**loop_metrics(result), **hook_metrics(hooks)},
            error="approval-pending: the run paused for approval" if pending else "",
        )


class FlowHost:
    """Runs a flow with `FlowRunner`; edits node prompts (`Flow.with_prompts`) and hook configs.

    Nodes read their prompt with `node_prompt(ctx)` and the model with `node_model(ctx)`;
    hooks are tunable as in `RegistryHost`.
    """

    def __init__(
        self,
        flow: Flow[Any],
        *,
        initial_state: Callable[[EvalCase], BaseModel],
        output: Callable[[BaseModel], str],
        cases: Sequence[EvalCase],
        judge: Judge | None = None,
        hooks: Sequence[Hook] = (),
        domain: str = "default",
        executor: Executor | None = None,
        allow_safety_tuning: bool = False,
    ) -> None:
        self.flow = flow
        self.initial_state = initial_state
        self.output = output
        self.judge = judge
        self.hooks = tuple(hooks)
        self.domain = domain
        self.executor = executor
        self._cases = _unique_cases(cases)
        self._hooks = HookSurface(self.hooks, allow_safety_tuning=allow_safety_tuning)

    def _path(self, node: str) -> str:
        return f"flow/{self.flow.name}/{node}/prompt"

    def surface(self) -> HarnessTree:
        files = {self._path(node): text for node, text in self.flow.prompts().items()}
        files.update(self._hooks.files())
        return HarnessTree(files=files)

    def caps(self) -> SurfaceCaps:
        return SurfaceCaps(
            max_chars=dict(DEFAULT_CAPS),
            kinds=frozenset({"flow", "hook"}),
            hook_schemas=self._hooks.schemas(),
        )

    def validate(self, tree: HarnessTree) -> list[str]:
        problems = self._hooks.problems(tree)
        prompts = self.flow.prompts()
        for node in prompts:
            if self._path(node) not in tree.files:
                problems.append(f"{self._path(node)}: a node prompt cannot be deleted")
        for path in tree.files:
            parsed = parsed_or_none(path)
            if parsed is None or parsed.kind != "flow":
                continue
            if parsed.name != self.flow.name:
                problems.append(f"{path}: no flow named {parsed.name!r}")
            elif parsed.node not in prompts:
                problems.append(f"{path}: node {parsed.node!r} has no prompt to edit")
        return sorted(problems)

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    async def run(
        self, tree: HarnessTree, case: EvalCase, *, run_id: str, model: LLMPort
    ) -> CaseRun:
        executor: Executor | None = None
        recorder = TranscriptRecorder()
        try:
            base = self.executor or Executor()
            recorder.withhold_tools(self._confirm_tools(base))
            prompts = {node: tree.files[self._path(node)] for node in self.flow.prompts()}
            flow = self.flow.with_prompts(prompts)
            hooks = self._hooks.applied(tree)
            sinks = [s for s in (base.sink, recorder) if s is not None]
            executor = base.replace(hooks=(*base.hooks, *hooks), sink=FanoutSink(*sinks))
            ctx = executor.context(run_id=run_id, extras={TRANSCRIPT_KEY: recorder})
            result = await FlowRunner(executor, model=model).run(
                flow, self.initial_state(case), ctx
            )
            output = self.output(result.state)
        except Exception as exc:
            if is_control_flow(exc):
                raise
            failed = failed_run(case, exc)
            transcript = "\n".join(filter(None, (recorder.text(), failed.transcript)))
            return failed.model_copy(update={"transcript": transcript})
        finally:
            if executor is not None:
                executor.end_run(run_id)
        return CaseRun(
            case_id=case.id,
            label="",
            ok=result.status == "done",
            delivered=result.status == "done" and bool(output.strip()),
            output=output,
            transcript=recorder.text(),
            metrics={**self._metrics(recorder, result.steps, result.visits), **hook_metrics(hooks)},
            error="" if result.status == "done" else f"flow {result.status}",
        )

    def _confirm_tools(self, executor: Executor) -> set[str]:
        """Tools whose arguments transcripts withhold: the executor's and the flow's own."""
        specs = list(executor.registry or ())
        try:
            nodes = list(self.flow.validate().nodes.values())
        except FlowDefinitionError:
            nodes = []  # the run itself reports the definition problems
        for node in nodes:
            tool = getattr(node.action, "tool", None)
            if isinstance(tool, ToolSpec):
                specs.append(tool)
        return {spec.name for spec in specs if spec.confirm is not False}

    @staticmethod
    def _metrics(
        recorder: TranscriptRecorder, steps: int, visits: Mapping[str, int]
    ) -> dict[str, float]:
        counts = recorder.counts
        metrics = {
            "steps": float(steps),
            "nodes": float(sum(visits.values())),
            "llm_calls": float(counts["llm_calls"]),
            "tool_calls": float(counts["tool.finished"] + counts["tool.failed"]),
            "tool_errors": float(counts["tool.failed"]),
        }
        metrics.update(
            {key: float(n) for key, n in counts.items() if key.startswith("tool_calls.")}
        )
        return metrics
