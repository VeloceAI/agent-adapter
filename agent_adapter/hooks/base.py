"""Hook base class and the invocation records hooks receive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.messages import LLMRequest, LLMResponse
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import LedgerEntry, ToolCall, ToolResult


@dataclass(frozen=True)
class ShortCircuit:
    """Returned by a `before_*` / `on_*_error` hook: skip the call and use `value`."""

    value: Any


@dataclass(frozen=True)
class ToolInvocation:
    """A tool call as hooks see it; `call.arguments` are validated, LLM-facing, JSON mode."""

    spec: ToolSpec
    call: ToolCall
    ctx: ToolContext


@dataclass(frozen=True)
class LLMInvocation:
    """A model call as hooks see it."""

    request: LLMRequest
    provider: str
    model: str
    ctx: ToolContext


@dataclass(frozen=True)
class NodeInvocation:
    """A flow node run as hooks see it."""

    flow: str
    node: str
    state: BaseModel
    ctx: ToolContext


class Hook:
    """Override any phase; the defaults do nothing (return None)."""

    name: ClassVar[str] = ""  # "" -> class name

    async def before_tool(self, inv: ToolInvocation) -> ShortCircuit | None:
        return None

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> ToolResult | None:
        return None

    async def on_tool_error(self, inv: ToolInvocation, error: BaseException) -> ShortCircuit | None:
        return None

    async def before_llm(self, inv: LLMInvocation) -> ShortCircuit | None:
        """A ShortCircuit's value must be an LLMResponse."""
        return None

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> LLMResponse | None:
        return None

    async def on_llm_error(self, inv: LLMInvocation, error: BaseException) -> ShortCircuit | None:
        return None

    async def on_llm_end(self, inv: LLMInvocation, billed: LedgerEntry | None) -> None:
        """Runs once when the call ends, however it ends; `billed` is its ledger entry or None."""
        return None

    async def before_node(self, inv: NodeInvocation) -> ShortCircuit | None:
        """A ShortCircuit's value must be the node's state update mapping."""
        return None

    async def after_node(
        self, inv: NodeInvocation, update: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        return None

    async def on_node_error(self, inv: NodeInvocation, error: BaseException) -> ShortCircuit | None:
        return None

    def config(self) -> dict[str, Any]:
        """Tunable settings (read by the optimiser); empty when the hook has none."""
        return {}

    @classmethod
    def config_schema(cls) -> dict[str, Any] | None:
        """JSON schema of `config()`, or None when the hook is not tunable."""
        return None

    def with_config(self, config: Mapping[str, Any]) -> Hook:
        """A copy with `config` applied.

        Raises:
            ToolUsageError: the hook is not tunable.
        """
        raise ToolUsageError(f"hook {hook_name(self)!r} does not support with_config()")


def hook_name(hook: Hook) -> str:
    """The hook's `name`, or its class name."""
    return getattr(hook, "name", "") or type(hook).__name__
