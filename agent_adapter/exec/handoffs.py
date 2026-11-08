"""Handoffs: a tool that passes the conversation's task to another agent, whose answer ends it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    StreamDone,
    StreamEvent,
    TextDelta,
)
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.agents import DEFAULT_MAX_DEPTH, LoopAgent, TaskAgent, agent_tool
from agent_adapter.exec.executor import Executor
from agent_adapter.models._port import closing

HANDOFF_TAG = "handoff"


def handoff(
    to: LoopAgent | TaskAgent,
    *,
    name: str,
    description: str,
    executor: Executor | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    timeout: float | None = None,
) -> ToolSpec:
    """An agent tool tagged "handoff": under a `HandoffModel`, its answer is the loop's answer."""
    return agent_tool(
        to,
        name=name,
        description=description,
        executor=executor,
        max_depth=max_depth,
        timeout=timeout,
        tags=(HANDOFF_TAG,),
    )


class HandoffModel:
    """Wraps a model: once a handoff tool has answered, the loop ends with that answer."""

    def __init__(self, model: LLMPort, handoffs: Registry | Iterable[str]) -> None:
        if isinstance(handoffs, Registry):
            names = {spec.name for spec in handoffs if HANDOFF_TAG in spec.tags}
        else:
            names = set(handoffs)
        if not names:
            raise ConfigError("HandoffModel needs at least one handoff tool")
        self.inner = model
        self.provider = model.provider
        self.model = model.model
        self.names = frozenset(names)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """The handoff's answer if one just arrived, else the wrapped model's reply."""
        answer = handed_off(request.messages, self.names)
        if answer is None:
            return await self.inner.generate(request)
        return self._closing(answer)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """The handoff's answer as one chunk if one just arrived, else the wrapped stream."""
        answer = handed_off(request.messages, self.names)
        if answer is None:
            async with closing(self.inner.stream(request)) as events:
                async for event in events:
                    yield event
            return
        yield TextDelta(text=answer)
        yield StreamDone(response=self._closing(answer))

    def cache_key_fields(self) -> dict[str, Any]:
        """The wrapped model's cache key fields."""
        return self.inner.cache_key_fields()

    def _closing(self, answer: str) -> LLMResponse:
        return LLMResponse(
            provider=self.provider,
            model=self.model,
            text=answer,
            finish_reason="stop",
            message=Message(role="assistant", content=answer),
        )


def handed_off(messages: Sequence[Message], names: Iterable[str]) -> str | None:
    """The answer of a handoff tool in the latest assistant turn, if it succeeded; else None."""
    wanted = set(names)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role != "assistant":
            continue
        for reply in messages[index + 1 :]:  # the results of that turn's calls
            if reply.role == "tool" and reply.name in wanted and not _failed(reply):
                return reply.content
        return None
    return None


def _failed(reply: Message) -> bool:
    """True for the `{"error": {...}}` body the loop writes for a failed call."""
    if not reply.content.startswith('{"error"'):
        return False
    try:
        body = json.loads(reply.content)
    except ValueError:
        return False
    return isinstance(body, dict) and set(body) == {"error"}
