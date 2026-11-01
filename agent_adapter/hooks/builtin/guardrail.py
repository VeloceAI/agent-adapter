"""GuardrailHook: run a check on tool args/results and LLM requests/responses; block on failure."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Literal, get_args

from agent_adapter.core.errors import ConfigError, GuardrailBlocked, ToolUsageError
from agent_adapter.core.messages import LLMResponse
from agent_adapter.core.types import ToolResult
from agent_adapter.hooks.base import Hook, LLMInvocation, ToolInvocation

logger = logging.getLogger(__name__)

Stage = Literal["args", "result", "llm_request", "llm_response"]
GuardrailCheck = Callable[[str, Any], bool | str | Awaitable[bool | str]]

DEFAULT_REASON = "check failed"


class GuardrailHook(Hook):
    """`check(stage, payload)`: True passes; False or a reason string raises GuardrailBlocked.

    Payloads: `args` the validated arguments, `result` the value of an ok result (failed
    results are not checked), `llm_request` the LLMRequest, `llm_response` the LLMResponse.
    """

    name = "guardrail"

    def __init__(self, check: GuardrailCheck, *, on: Iterable[Stage] | Stage = ("args",)) -> None:
        if not callable(check):
            raise ConfigError("GuardrailHook check must be callable")
        stages = frozenset([on] if isinstance(on, str) else on)
        allowed = frozenset(get_args(Stage))
        if not stages or not stages <= allowed:
            raise ConfigError(
                f"GuardrailHook stages must be a non-empty subset of {sorted(allowed)}, "
                f"got {sorted(stages)}"
            )
        self.check = check
        self.stages: frozenset[str] = stages

    async def before_tool(self, inv: ToolInvocation) -> None:
        if "args" in self.stages:
            await self._run("args", inv.call.arguments, tool=inv.spec.name)

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> None:
        if "result" in self.stages and result.ok:
            await self._run("result", result.value, tool=inv.spec.name)

    async def before_llm(self, inv: LLMInvocation) -> None:
        if "llm_request" in self.stages:
            await self._run("llm_request", inv.request)

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> None:
        if "llm_response" in self.stages:
            await self._run("llm_response", response)

    async def _run(self, stage: str, payload: Any, *, tool: str = "") -> None:
        verdict = self.check(stage, payload)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        if verdict is True:
            return
        if verdict is False:
            reason = DEFAULT_REASON
        elif isinstance(verdict, str):
            reason = verdict or DEFAULT_REASON
        else:
            raise ToolUsageError(
                f"guardrail check must return a bool or a reason string, "
                f"got {type(verdict).__name__}"
            )
        where = f" for tool {tool!r}" if tool else ""
        # The reason may quote the payload (an amount, a matched secret): DEBUG only.
        logger.warning("guardrail blocked %s%s", stage, where)
        logger.debug("guardrail reason for %s%s: %s", stage, where, reason)
        raise GuardrailBlocked(reason, tool=tool, stage=stage)
