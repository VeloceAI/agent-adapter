"""The tool loop's steps: ask the model, run its tool calls, keep the history."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

from agent_adapter.context.model import marked
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.context.run import after_overflow, compress_for_model
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ModelContextOverflow, ModelOutputInvalid
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolDef,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, ToolResult, Usage
from agent_adapter.exec._durable_loop import LoopCheckpoints
from agent_adapter.exec._loop_history import _history_digest, _trailing_turn, _unanswered
from agent_adapter.exec.checkpoints import CheckpointStatus
from agent_adapter.exec.executor import Executor
from agent_adapter.models._port import closing
from agent_adapter.models.instrument import InstrumentedModel

logger = logging.getLogger(__name__)

LoopFinish = Literal["stop", "max_steps", "length", "content_filter", "pending"]
TextCallback = Callable[[str], Awaitable[None] | None]
EventCallback = Callable[[StreamEvent], Awaitable[None] | None]


@dataclass(frozen=True)
class ToolLoopResult:
    """Where a tool loop ended: the full history, the last reply and every tool result."""

    messages: tuple[Message, ...]
    final: LLMResponse
    steps: int  # model calls made by this run of the loop
    finish: LoopFinish
    tool_results: tuple[ToolResult, ...]
    usage: Usage


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
        durable: LoopCheckpoints | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.durable = durable
        self.llm = llm
        self.compression = compression
        self.on_text = on_text
        self.on_event = on_event
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
        if self.durable is not None and self.durable.unsaved:
            await self._save("running")  # a new run's step 0, or a paused run going on
        if trailing is not None and await self._run_calls(*trailing):
            index = trailing[0]
            return await self._end("pending", self._replayed(messages[index]))
        while True:
            final = await self._generate()
            self.steps += 1
            self.usage += final.usage
            self.history.append(final.message)
            if final.finish_reason in ("length", "content_filter"):
                logger.info("tool loop stopped: %s (run %s)", final.finish_reason, self.ctx.run_id)
                stop: LoopFinish = "length" if final.finish_reason == "length" else "content_filter"
                return await self._end(stop, final)
            if not final.tool_calls:
                if final.finish_reason == "error":
                    logger.warning(
                        "tool loop: the model ended with an error and no tool calls "
                        "(raw %r, run %s); finishing as stop",
                        final.raw_finish_reason,
                        self.ctx.run_id,
                    )
                return await self._end("stop", final)
            if self.steps >= max_steps:
                logger.info("tool loop hit max_steps=%d (run %s)", max_steps, self.ctx.run_id)
                return await self._end("max_steps", final)
            await self._save("running")  # the calls are pinned before any of them runs
            if await self._run_calls(len(self.history) - 1, final.tool_calls):
                return await self._end("pending", final)
            await self._save("running")  # the results are kept before the next model call

    async def _save(self, status: CheckpointStatus) -> None:
        if self.durable is not None:
            await self.durable.save(self.history, self.steps, status)

    async def _end(self, finish: LoopFinish, final: LLMResponse) -> ToolLoopResult:
        await self._save("paused" if finish == "pending" else "done")
        return self._result(finish, final)

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
        if self.on_text is None and self.on_event is None:
            return await self.llm.generate(request)
        return await self._streamed(request)

    async def _streamed(self, request: LLMRequest) -> LLMResponse:
        """One streamed model call: text to `on_text`, other events to `on_event`."""
        async with closing(self.llm.stream(request)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    if event.text and self.on_text is not None:
                        await _call_back(self.on_text, event.text)
                elif self.on_event is not None:
                    await _call_back(self.on_event, event)
                if isinstance(event, StreamDone):
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
        logged = self.durable.calls(index) if self.durable is not None else nullcontext()
        with logged:
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


async def _call_back(callback: Callable[[Any], Awaitable[None] | None], value: Any) -> None:
    answer = callback(value)
    if inspect.isawaitable(answer):
        await answer
