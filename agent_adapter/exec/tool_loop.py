"""`run_tool_loop`: a minimal tool-calling loop over any LLMPort, tools run by the executor."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from agent_adapter.context.model import marked
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.context.run import (
    after_overflow,
    check_policy,
    compress_for_model,
    context_window,
)
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelContextOverflow, ModelOutputInvalid, ToolUsageError
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    OutputSchema,
    StreamDone,
    TextDelta,
    ToolDef,
)
from agent_adapter.core.ports import LedgerPort, LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, ToolResult, Usage
from agent_adapter.exec._loop_history import _history_digest, _trailing_turn, _unanswered
from agent_adapter.exec.executor import Executor, bind_model
from agent_adapter.models._port import closing
from agent_adapter.models.entries import entry_prefix, seed_entry_numbers
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.runtime.ids import new_id

logger = logging.getLogger(__name__)

LoopFinish = Literal["stop", "max_steps", "length", "content_filter", "pending"]
TextCallback = Callable[[str], Awaitable[None] | None]

_RESERVED = frozenset({"messages", "tools", "tool_choice", "output_schema", "stage", "compression"})


@dataclass(frozen=True)
class ToolLoopResult:
    """Where a tool loop ended: the full history, the last reply and every tool result."""

    messages: tuple[Message, ...]
    final: LLMResponse
    steps: int  # model calls made by this run of the loop
    finish: LoopFinish
    tool_results: tuple[ToolResult, ...]
    usage: Usage


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
) -> ToolLoopResult:
    """Call the model, run its tool calls, feed the results back, until it answers.

    `messages` ending in an assistant turn whose tool calls lack results (a loop that stopped
    on `pending` or `max_steps`) resumes by running those calls first; pass the same `ctx`
    (or one with the same `run_id`) so approvals and run guards carry over.

    With `compression`, the history is compressed before each model call and the loop goes
    on with the shorter history (also in the result); a context overflow runs the policy's
    `if_still_too_long` once and retries the call once.

    With `on_text`, each model call streams through the port's `stream()` and every text
    chunk is passed to `on_text` (sync or async) as it arrives.

    Raises:
        ToolUsageError: `max_steps < 1`, or bad `request_options` (reserved, unknown or
            invalid keys, or `metadata["call_id"]`), before anything runs.
        ConfigError: `compression` is not a `CompressionPolicy`, or no context window is
            known for the model, before anything runs.
        ModelError: the model call failed.
    """
    if max_steps < 1:
        raise ToolUsageError(f"max_steps must be >= 1, got {max_steps}")
    tools = tuple(spec.tool_def() for spec in registry)
    request = _request_fields(
        request_options, tools, tool_choice=tool_choice, output_schema=output_schema, stage=stage
    )
    if compression is not None:
        context_window(check_policy(compression), request.get("model") or model.model)
    executor = executor if executor is not None else Executor(registry)
    owned = ctx is None or not ctx.run_id
    if ctx is None:
        ctx = executor.context()
    elif not ctx.run_id:
        ctx = dataclasses.replace(ctx, run_id=new_id("run_"))
    # A fresh context resuming a run must not reuse the entry ids of the turns already paid;
    # a compressed history has fewer turns than were paid, so the ledger is asked too.
    llm = bind_model(executor, model, ctx)
    turns = sum(1 for m in messages if m.role == "assistant")
    if compression is not None:
        turns = max(turns, _paid_turns(ctx, llm.ledger or ctx.ledger, stage))
    seed_entry_numbers(ctx, stage, turns)
    loop = _Loop(
        llm,
        registry,
        executor,
        ctx,
        tools=tools,
        request=request,
        concurrency=concurrency,
        stage=stage,
        compression=compression,
        on_text=on_text,
    )
    try:
        return await loop.run(list(messages), max_steps)
    finally:
        if owned:
            executor.end_run(ctx.run_id)


class _Loop:
    """State of one `run_tool_loop` call."""

    def __init__(
        self,
        llm: InstrumentedModel,
        registry: Registry,
        executor: Executor,
        ctx: ToolContext,
        *,
        tools: tuple[ToolDef, ...],
        request: dict[str, Any],
        concurrency: int | None,
        stage: str,
        compression: CompressionPolicy | None,
        on_text: TextCallback | None = None,
    ) -> None:
        self.llm = llm
        self.compression = compression
        self.on_text = on_text
        self.registry = registry
        self.executor = executor
        self.ctx = ctx
        self.tools = tools
        self.request = request
        self.concurrency = concurrency
        self.id_prefix = f"{ctx.call_id or ctx.run_id}:{stage}"
        self.history: list[Message] = []
        self.results: list[ToolResult] = []
        self.usage = Usage()
        self.steps = 0

    async def run(self, messages: list[Message], max_steps: int) -> ToolLoopResult:
        self.history = messages
        trailing = _trailing_turn(messages)
        if trailing is not None and await self._run_calls(*trailing):
            index = trailing[0]
            return self._result("pending", self._replayed(messages[index]))
        while True:
            final = await self._generate()
            self.steps += 1
            self.usage += final.usage
            self.history.append(final.message)
            if final.finish_reason in ("length", "content_filter"):
                logger.info("tool loop stopped: %s (run %s)", final.finish_reason, self.ctx.run_id)
                stop: LoopFinish = "length" if final.finish_reason == "length" else "content_filter"
                return self._result(stop, final)
            if not final.tool_calls:
                if final.finish_reason == "error":
                    logger.warning(
                        "tool loop: the model ended with an error and no tool calls "
                        "(raw %r, run %s); finishing as stop",
                        final.raw_finish_reason,
                        self.ctx.run_id,
                    )
                return self._result("stop", final)
            if self.steps >= max_steps:
                logger.info("tool loop hit max_steps=%d (run %s)", max_steps, self.ctx.run_id)
                return self._result("max_steps", final)
            if await self._run_calls(len(self.history) - 1, final.tool_calls):
                return self._result("pending", final)

    async def _generate(self) -> LLMResponse:
        """One model call on the history, compressed first when the loop has a policy."""
        policy = self.compression
        if policy is None:
            return await self._call()
        where = self._where()
        self.history, _ = await compress_for_model(self.history, policy, **where)
        try:
            return await self._call()
        except ModelContextOverflow:
            smaller = await after_overflow(self.history, policy, **where)
            if smaller is None:
                raise
            self.history = smaller
        return await self._call()

    async def _call(self) -> LLMResponse:
        request = LLMRequest(messages=tuple(self.history), tools=self.tools, **self.request)
        # The loop compressed and retries itself: a CompressingModel inside must not again.
        request = marked(request) if self.compression else request
        if self.on_text is None:
            return await self.llm.generate(request)
        return await self._streamed(request, self.on_text)

    async def _streamed(self, request: LLMRequest, on_text: TextCallback) -> LLMResponse:
        """One streamed model call: text chunks to `on_text`, the `StreamDone` response back."""
        async with closing(self.llm.stream(request)) as events:
            async for event in events:
                if isinstance(event, TextDelta) and event.text:
                    answer = on_text(event.text)
                    if inspect.isawaitable(answer):
                        await answer
                elif isinstance(event, StreamDone):
                    return event.response
        # InstrumentedModel ends every stream with StreamDone or raises; kept as a guard.
        raise ModelOutputInvalid(  # pragma: no cover
            f"{self.llm.provider} {self.llm.model}: stream ended without StreamDone",
            provider=self.llm.provider,
            model=self.llm.model,
        )

    def _where(self) -> dict[str, Any]:
        model = self.request.get("model") or self.llm.model
        return {"provider": self.llm.provider, "model": model, "ctx": self.ctx}

    async def _run_calls(self, index: int, calls: Sequence[ToolCall]) -> bool:
        """Run the unanswered `calls` of the assistant turn at `index`; True if any is pending."""
        identified = self._with_ids(index, calls)
        todo = _unanswered(identified, self.history[index + 1 :])
        if not todo:
            return False
        results = await self.executor.run_many(
            todo, self.ctx, concurrency=self.concurrency, registry=self.registry
        )
        self.results.extend(results)
        pending = False
        for call, result in zip(todo, results, strict=True):
            if result.status == "pending":
                pending = True
                continue
            self.history.append(
                Message(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=_tool_content(result),
                )
            )
        if pending:
            logger.info("tool loop paused for approval (run %s)", self.ctx.run_id)
        return pending

    def _with_ids(self, index: int, calls: Sequence[ToolCall]) -> list[tuple[ToolCall, bool]]:
        """Each call with an id, and whether the model gave it; the same on every resume."""
        if all(call.id for call in calls):
            return [(call, True) for call in calls]
        # The conversation digest keeps ids distinct across loops sharing a run.
        conversation = _history_digest(self.history[: index + 1])
        return [
            (call, True)
            if call.id
            else (
                call.model_copy(update={"id": f"{self.id_prefix}:{conversation}:{index}:{n}"}),
                False,
            )
            for n, call in enumerate(calls)
        ]

    def _replayed(self, message: Message) -> LLMResponse:
        """The paused assistant turn as a response, when no model call happened."""
        return LLMResponse(
            provider=self.llm.provider,
            model=self.llm.model,
            text=message.content,
            tool_calls=message.tool_calls,
            finish_reason="tool_calls",
            message=message,
        )

    def _result(self, finish: LoopFinish, final: LLMResponse) -> ToolLoopResult:
        return ToolLoopResult(
            messages=tuple(self.history),
            final=final,
            steps=self.steps,
            finish=finish,
            tool_results=tuple(self.results),
            usage=self.usage,
        )


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


def _tool_content(result: ToolResult) -> str:
    """A tool message body: the value as text, or `{"error": {"kind", "message"}}` JSON."""
    if result.ok:
        value = result.value
        return (
            value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        )
    error = result.error
    kind, message = (error.kind, error.message) if error is not None else ("internal-error", "")
    return json.dumps({"error": {"kind": kind, "message": message}}, ensure_ascii=False)
