"""Shared steps of the native approval bridges (Agents SDK, Pydantic AI)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import ApprovalDecision, ToolCall
from agent_adapter.exec.args import dump_args
from agent_adapter.integrations._native_gate import TokenTable

logger = logging.getLogger(__name__)

# One resolve may hold at most this share of a bridge's process-wide token table.
MAX_TOKENS_PER_RESOLVE = 256


def validated_call(spec: ToolSpec, arguments: Any, call_id: str) -> ToolCall:
    """The call a `confirm` rule sees: validated arguments, LLM-facing names, JSON mode.

    Raises:
        ToolArgsInvalid: the arguments fail validation.
    """
    return ToolCall(name=spec.name, arguments=dump_args(spec.validate_args(arguments)), id=call_id)


@dataclass(frozen=True)
class Grant:
    """One approved call to issue a token for, bound to `run_key`."""

    tool: str
    arguments: dict[str, Any]
    decision: ApprovalDecision
    call_id: str
    run_key: str


def issue_tokens(tokens: TokenTable, grants: Sequence[Grant], *, integration: str) -> None:
    """Issue a token per grant, at most `MAX_TOKENS_PER_RESOLVE`; warn on overflow and eviction.

    A grant left without a token is not lost: the executor asks its own port when it runs.
    """
    if len(grants) > MAX_TOKENS_PER_RESOLVE:
        logger.warning(
            "%s: %d approved calls in one resolve; only the first %d get a token, "
            "the executor asks again for the rest",
            integration,
            len(grants),
            MAX_TOKENS_PER_RESOLVE,
        )
    before = tokens.evicted
    for grant in grants[:MAX_TOKENS_PER_RESOLVE]:
        tokens.issue(
            grant.tool, grant.arguments, grant.decision, call_id=grant.call_id, run_id=grant.run_key
        )
    evicted = tokens.evicted - before
    if evicted:
        logger.warning(
            "%s: this resolve evicted %d older approval token(s); those calls go back to "
            "the executor's port",
            integration,
            evicted,
        )
