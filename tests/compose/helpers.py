"""A scripted model for compose tests: answers by blank (the request's ledger stage)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import agent_adapter as aa

Answer = Any  # a str, a response mapping, or a list of them (one per call, the last repeats)


class Script:
    """Replies per blank; records every request and how many ran at once."""

    def __init__(self, answers: Mapping[str, Answer], *, delay: float = 0.0) -> None:
        self.answers = dict(answers)
        self.delay = delay
        self.requests: list[aa.LLMRequest] = []
        self.running = 0
        self.most = 0
        self.model = aa.CallableModel(self._reply, provider="local", model="scripted")

    def calls(self, part: str) -> list[aa.LLMRequest]:
        return [r for r in self.requests if r.stage.rsplit(".", 1)[-1] == part]

    def order(self) -> list[str]:
        return [r.stage.rsplit(".", 1)[-1] for r in self.requests]

    async def _reply(self, request: aa.LLMRequest) -> Any:
        self.requests.append(request)
        self.running += 1
        self.most = max(self.most, self.running)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            part = request.stage.rsplit(".", 1)[-1]
            answer = self.answers[part]
            if callable(answer):
                return answer(request)
            if isinstance(answer, list):
                index = min(len(self.calls(part)) - 1, len(answer) - 1)
                answer = answer[index]
            return answer
        finally:
            self.running -= 1


def parsed(value: Any, usage: Mapping[str, int] | None = None) -> dict[str, Any]:
    """A structured reply."""
    reply: dict[str, Any] = {"text": "", "parsed": value}
    if usage:
        reply["usage"] = dict(usage)
    return reply


def executor(**options: Any) -> aa.Executor:
    return aa.Executor(aa.Registry(), **options)


def prompt_of(request: aa.LLMRequest) -> str:
    return request.messages[-1].content


Reply = Callable[[aa.LLMRequest], Any]
