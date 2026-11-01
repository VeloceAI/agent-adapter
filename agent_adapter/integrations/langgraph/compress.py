"""`langgraph_compressor`: compress a LangGraph state's messages inside your own agent node."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_adapter.context.formats import Pinned, compress_for, compress_for_sync
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.context.run import check_policy
from agent_adapter.core.context import ToolContext


class LangGraphCompressor:
    """Compresses LangChain messages before your node calls its model; the state is untouched.

    Tool-call ids stay paired, pinned and approval messages stay verbatim, and messages that
    are kept come back as the same objects.
    """

    def __init__(
        self,
        policy: CompressionPolicy,
        *,
        pinned: Pinned | None = None,
        model_max_tokens: int | None = None,
    ) -> None:
        self.policy = check_policy(policy)
        self.pinned = pinned
        self.model_max_tokens = model_max_tokens

    async def __call__(self, messages: Sequence[Any], ctx: ToolContext | None = None) -> list[Any]:
        """The compressed messages, for an async node.

        Raises:
            ConfigError: an item is not a LangChain message, or the policy cannot run.
            MissingDependency: langchain-core is missing and a step created a message.
        """
        smaller, _ = await compress_for(
            messages,
            self.policy,
            format="langchain",
            pinned=self.pinned,
            model_max_tokens=self.model_max_tokens,
            ctx=ctx,
        )
        return smaller

    def sync(self, messages: Sequence[Any], ctx: ToolContext | None = None) -> list[Any]:
        """The compressed messages, for a sync node.

        Raises:
            ConfigError: as `__call__`.
            MissingDependency: as `__call__`.
        """
        smaller, _ = compress_for_sync(
            messages,
            self.policy,
            format="langchain",
            pinned=self.pinned,
            model_max_tokens=self.model_max_tokens,
            ctx=ctx,
        )
        return smaller


def langgraph_compressor(
    policy: CompressionPolicy,
    *,
    pinned: Pinned | None = None,
    model_max_tokens: int | None = None,
) -> LangGraphCompressor:
    """A compressor for `state["messages"]`: `await c(messages)` or `c.sync(messages)`.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy`.
    """
    return LangGraphCompressor(policy, pinned=pinned, model_max_tokens=model_max_tokens)
