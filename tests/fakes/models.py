"""Model fakes: a scripted CallableModel, response helpers and a fake prompt-cache plug-in."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from agent_adapter.core.messages import FinishReason, LLMRequest, LLMResponse, Message
from agent_adapter.core.types import ToolCall, Usage
from agent_adapter.models.callable import CallableModel

Scripted = LLMResponse | str | BaseException


def text(
    content: str,
    *,
    provider: str = "scripted",
    model: str = "scripted",
    usage: Usage | None = None,
    finish_reason: FinishReason = "stop",
    parsed: Any = None,
) -> LLMResponse:
    """A plain text reply."""
    return LLMResponse(
        provider=provider,
        model=model,
        text=content,
        parsed=parsed,
        finish_reason=finish_reason,
        usage=usage or Usage(),
        message=Message(role="assistant", content=content),
    )


def calls(
    *tool_calls: ToolCall | tuple[str, dict[str, Any]],
    provider: str = "scripted",
    model: str = "scripted",
    usage: Usage | None = None,
    content: str = "",
) -> LLMResponse:
    """A reply asking for tool calls; tuples are `(name, arguments)` with ids `call_<n>`."""
    made = tuple(
        call
        if isinstance(call, ToolCall)
        else ToolCall(name=call[0], arguments=call[1], id=f"call_{n}")
        for n, call in enumerate(tool_calls)
    )
    return LLMResponse(
        provider=provider,
        model=model,
        text=content,
        tool_calls=made,
        finish_reason="tool_calls",
        usage=usage or Usage(),
        message=Message(role="assistant", content=content, tool_calls=made),
    )


class ScriptedModel(CallableModel):
    """Answers with `responses` in order (an exception item is raised); records requests."""

    def __init__(
        self,
        responses: Sequence[Scripted],
        *,
        provider: str = "scripted",
        model: str = "scripted",
    ) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []
        super().__init__(self._next, provider=provider, model=model)

    async def _next(self, request: LLMRequest) -> LLMResponse | str:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"ScriptedModel {self.model!r} has no response left")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakePromptCachePlugin:
    """PromptCachePlugin that merges fixed `sets` into the SDK kwargs and records each call."""

    name = "fake"

    def __init__(
        self,
        sets: dict[str, Any],
        *,
        providers: Iterable[str] = ("openai", "gemini", "anthropic"),
        edit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.sets = sets
        self.providers = frozenset(providers)
        self.edit = edit
        self.calls: list[dict[str, Any]] = []

    def key_fields(self) -> dict[str, Any]:
        return {"name": self.name, "sets": sorted(self.sets)}

    async def apply(
        self, request: LLMRequest, kwargs: dict[str, Any], *, model: str, client: Any
    ) -> dict[str, Any]:
        self.calls.append(
            {"request": request, "kwargs": copy.deepcopy(kwargs), "model": model, "client": client}
        )
        out = {**copy.deepcopy(kwargs), **self.sets}
        if self.edit is not None:
            self.edit(out)
        return out
