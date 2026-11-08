"""`run_tool_loop`: a minimal tool-calling loop over any LLMPort, tools run by the executor."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.context.run import (
    check_policy,
    context_window,
)
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import (
    LLMRequest,
    Message,
    OutputSchema,
    ToolDef,
)
from agent_adapter.core.ports import LedgerPort, LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.exec._durable import durable_key
from agent_adapter.exec._durable_loop import LoopCheckpoints, Saved, begin, claim, load
from agent_adapter.exec._durable_seal import check_identity
from agent_adapter.exec._loop_history import _history_digest as _history_digest
from agent_adapter.exec._loop_history import _trailing_turn as _trailing_turn
from agent_adapter.exec._loop_history import _unanswered as _unanswered
from agent_adapter.exec._loop_steps import EventCallback as EventCallback
from agent_adapter.exec._loop_steps import LoopFinish as LoopFinish
from agent_adapter.exec._loop_steps import TextCallback as TextCallback
from agent_adapter.exec._loop_steps import ToolLoopResult as ToolLoopResult
from agent_adapter.exec._loop_steps import _Loop
from agent_adapter.exec._loop_steps import _tool_content as _tool_content
from agent_adapter.exec.checkpoints import CheckpointStore
from agent_adapter.exec.executor import Executor, bind_model
from agent_adapter.models.entries import entry_prefix, seed_entry_numbers
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

_RESERVED = frozenset({"messages", "tools", "tool_choice", "output_schema", "stage", "compression"})


async def run_tool_loop(
    model: LLMPort,
    registry: Registry,
    messages: Sequence[Message],
    *,
    max_steps: int = 8,
    executor: Executor | None = None,
    ctx: ToolContext | None = None,
    output_schema: OutputSchema | None = None,
    tool_choice: Literal["auto", "none", "required"] | str = "auto",
    concurrency: int | None = None,
    stage: str = "tool_loop",
    request_options: Mapping[str, Any] | None = None,
    compression: CompressionPolicy | None = None,
    on_text: TextCallback | None = None,
    on_event: EventCallback | None = None,
    run_id: str | None = None,
    checkpoints: CheckpointStore | None = None,
    checkpoint_key: bytes | None = None,
) -> ToolLoopResult:
    """Call the model, run its tool calls, feed the results back, until it answers.

    `messages` ending in an assistant turn whose tool calls lack results (a loop that stopped
    on `pending` or `max_steps`) resumes by running those calls first; pass the same `ctx`
    (or one with the same `run_id`) so approvals and run guards carry over.

    With `compression`, the history is compressed before each model call and the loop goes
    on with the shorter history (also in the result); a context overflow runs the policy's
    `if_still_too_long` once and retries the call once.

    With `on_text`, each model call streams through the port's `stream()` and every text
    chunk is passed to `on_text` (sync or async) as it arrives. `on_event` (also streaming)
    gets every other stream event: `ToolCallDelta`, `PartialOutput`, and the `StreamDone`
    that ends each model call.

    With `checkpoints` (and a `run_id`, or a `ctx` with one), the history is saved signed
    after each step and every tool call is logged; `recover_tool_loop` continues the run
    after a crash without re-running a finished call. A paused run resumes here.

    Raises:
        ToolUsageError: `max_steps < 1`, or bad `request_options` (reserved, unknown or
            invalid keys, or `metadata["call_id"]`), before anything runs.
        ConfigError: `compression` is not a `CompressionPolicy`, or no context window is
            known for the model, before anything runs; a durable store without a key.
        CheckpointConflict: `checkpoints` holds this run as running or ended.
        ModelError: the model call failed.
    """
    setup = _Setup.made(
        model,
        registry,
        max_steps,
        executor,
        ctx,
        run_id,
        request_options,
        tool_choice,
        output_schema,
        stage,
        compression,
        checkpoints,
    )
    durable = None
    if checkpoints is not None:
        key = durable_key(checkpoints, checkpoint_key, "run_tool_loop")
        durable = await begin(checkpoints, key, stage, setup.ctx)
    return await setup.run(
        list(messages),
        max_steps,
        concurrency=concurrency,
        on_text=on_text,
        on_event=on_event,
        durable=durable,
    )


async def recover_tool_loop(
    model: LLMPort,
    registry: Registry,
    run_id: str,
    *,
    checkpoints: CheckpointStore,
    checkpoint_key: bytes | None = None,
    rerun_in_doubt: bool = False,
    max_steps: int = 8,
    executor: Executor | None = None,
    ctx: ToolContext | None = None,
    output_schema: OutputSchema | None = None,
    tool_choice: Literal["auto", "none", "required"] | str = "auto",
    concurrency: int | None = None,
    stage: str = "tool_loop",
    request_options: Mapping[str, Any] | None = None,
    compression: CompressionPolicy | None = None,
    on_text: TextCallback | None = None,
    on_event: EventCallback | None = None,
) -> ToolLoopResult:
    """Continue a durable tool loop from its last checkpoint; finished tool calls replay.

    A call that started but logged no result runs again only if its tool is
    `idempotent=True`, or with `rerun_in_doubt=True` after you checked it. `max_steps`
    counts the model calls of the whole run, before the crash too; `tool_results` and
    `usage` cover this call only.

    Raises:
        CheckpointNotFound: nothing is stored for `run_id`.
        CheckpointConflict: another worker took the run, it ended, or it is not this loop.
        CheckpointStoreCorrupt: the stored checkpoint or its signature does not check out.
        RecoveryNeedsDecision: a call may have run and its tool is not idempotent.
        ToolUsageError, ConfigError, ModelError: as `run_tool_loop`.
    """
    key = durable_key(checkpoints, checkpoint_key, "recover_tool_loop")
    saved: Saved = await load(checkpoints, key, run_id, stage)
    if ctx is not None and ctx.run_id not in ("", run_id):
        raise ToolUsageError(f"ctx.run_id {ctx.run_id!r} is not the recovered run {run_id!r}")
    check_identity(run_id, saved.identity, ctx)
    owned = ctx is None or not ctx.run_id
    executor = executor if executor is not None else Executor(registry)
    if ctx is None:
        identity = saved.identity
        ctx = executor.context(
            run_id=run_id, user_id=identity["user_id"], thread_id=identity["thread_id"]
        )
    setup = _Setup.made(
        model,
        registry,
        max_steps,
        executor,
        ctx,
        run_id,
        request_options,
        tool_choice,
        output_schema,
        stage,
        compression,
        checkpoints,
        owned=owned,
    )
    durable = await claim(saved, checkpoints, key, setup.ctx, registry, rerun=rerun_in_doubt)
    return await setup.run(
        saved.messages,
        max_steps,
        concurrency=concurrency,
        on_text=on_text,
        on_event=on_event,
        durable=durable,
        steps=saved.steps,
    )


@dataclass
class _Setup:
    """A tool loop's checked options, bound model and run context, before it runs."""

    llm: InstrumentedModel
    registry: Registry
    executor: Executor
    ctx: ToolContext
    owned: bool
    tools: tuple[ToolDef, ...]
    request: dict[str, Any]
    stage: str
    compression: CompressionPolicy | None

    @classmethod
    def made(
        cls,
        model: LLMPort,
        registry: Registry,
        max_steps: int,
        executor: Executor | None,
        ctx: ToolContext | None,
        run_id: str | None,
        request_options: Mapping[str, Any] | None,
        tool_choice: str,
        output_schema: OutputSchema | None,
        stage: str,
        compression: CompressionPolicy | None,
        checkpoints: CheckpointStore | None,
        *,
        owned: bool | None = None,
    ) -> _Setup:
        """Check the options and bind the run.

        Raises:
            ToolUsageError: bad options, or a run id that disagrees with `ctx` (or none
                for a durable loop).
            ConfigError: a bad `compression`.
        """
        if max_steps < 1:
            raise ToolUsageError(f"max_steps must be >= 1, got {max_steps}")
        tools = tuple(spec.tool_def() for spec in registry)
        request = _request_fields(
            request_options,
            tools,
            tool_choice=tool_choice,
            output_schema=output_schema,
            stage=stage,
        )
        if compression is not None:
            context_window(check_policy(compression), request.get("model") or model.model)
        executor = executor if executor is not None else Executor(registry)
        if run_id is not None and ctx is not None and ctx.run_id not in ("", run_id):
            raise ToolUsageError(f"run_id {run_id!r} is not ctx.run_id {ctx.run_id!r}")
        if checkpoints is not None and not run_id and (ctx is None or not ctx.run_id):
            raise ToolUsageError("a tool loop with checkpoints= needs run_id= (or a ctx with one)")
        if owned is None:
            owned = ctx is None or not ctx.run_id
        if ctx is None:
            ctx = executor.context(run_id=run_id or None)
        elif not ctx.run_id:
            ctx = dataclasses.replace(ctx, run_id=run_id or new_id("run_"))
        llm = bind_model(executor, model, ctx)
        return cls(llm, registry, executor, ctx, owned, tools, request, stage, compression)

    async def run(
        self,
        messages: list[Message],
        max_steps: int,
        *,
        concurrency: int | None,
        on_text: TextCallback | None,
        durable: LoopCheckpoints | None,
        on_event: EventCallback | None = None,
        steps: int = 0,
    ) -> ToolLoopResult:
        """Run the loop on `messages`; the run ends here when this call made it."""
        ctx, stage = self.ctx, self.stage
        # A fresh context resuming a run must not reuse the entry ids of the turns already
        # paid; a compressed history has fewer turns than were paid, so the ledger is asked.
        turns = sum(1 for m in messages if m.role == "assistant")
        if self.compression is not None:
            turns = max(turns, _paid_turns(ctx, self.llm.ledger or ctx.ledger, stage))
        seed_entry_numbers(ctx, stage, turns)
        loop = _Loop(
            self.llm,
            self.registry,
            self.executor,
            ctx,
            tools=self.tools,
            request=self.request,
            concurrency=concurrency,
            stage=stage,
            compression=self.compression,
            on_text=on_text,
            durable=durable,
            on_event=on_event,
        )
        loop.steps = steps
        try:
            return await loop.run(messages, max_steps)
        finally:
            if self.owned:
                self.executor.end_run(ctx.run_id)


def _request_fields(
    options: Mapping[str, Any] | None, tools: tuple[ToolDef, ...], **fields: Any
) -> dict[str, Any]:
    """The loop's LLMRequest fields, validated once before anything runs."""
    options = dict(options or {})
    allowed = set(LLMRequest.model_fields) - _RESERVED
    bad = sorted(set(options) - allowed)
    if bad:
        raise ToolUsageError(f"request_options may not set {bad}; allowed: {sorted(allowed)}")
    metadata = options.get("metadata")
    if isinstance(metadata, Mapping) and "call_id" in metadata:
        # One fixed ledger entry id for every step would overwrite the steps' entries.
        raise ToolUsageError("request_options metadata may not set call_id in a tool loop")
    request = {**fields, **options}
    try:
        LLMRequest(messages=(), tools=tools, **request)
    except ValidationError as exc:
        raise ToolUsageError(f"invalid tool loop request options: {exc}") from exc
    return request


def _paid_turns(ctx: ToolContext, ledger: LedgerPort | None, stage: str) -> int:
    """One past the highest model-turn entry number the ledger holds for this loop's run."""
    if ledger is None:
        return 0
    prefix = f"{entry_prefix(ctx, stage)}:"
    numbers = [
        int(suffix)
        for entry in ledger.entries(ctx.run_id)
        if entry.entry_id.startswith(prefix) and (suffix := entry.entry_id[len(prefix) :]).isdigit()
    ]
    return max(numbers) + 1 if numbers else 0
