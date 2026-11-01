"""Ledger entry ids of instrumented model calls: `{call or run id}:llm:{stage}:{n}`."""

from __future__ import annotations

from collections.abc import Mapping

from agent_adapter.core.context import ToolContext
from agent_adapter.core.messages import LLMRequest

SEQ_KEY = "aa.llm_seq"  # ctx.extras: {"<call or run id>:llm:<stage>": next number}


def entry_prefix(ctx: ToolContext, stage: str) -> str:
    """`{call or run id}:llm:{stage or "llm"}`: the entry ids of one context and stage."""
    return f"{ctx.call_id or ctx.run_id}:llm:{stage or 'llm'}"


def seed_entry_numbers(ctx: ToolContext, stage: str, at_least: int) -> None:
    """Make the next entry number of (ctx, stage) at least `at_least`."""
    prefix = entry_prefix(ctx, stage)
    seq = _sequence(ctx)
    seq[prefix] = max(seq.get(prefix, 0), at_least)
    ctx.extras[SEQ_KEY] = seq


def next_entry_id(ctx: ToolContext, request: LLMRequest) -> str:
    """`metadata["call_id"]`, else `{prefix}:{n}` numbered per context and stage."""
    explicit = request.metadata.get("call_id")
    if explicit:
        return explicit
    prefix = entry_prefix(ctx, request.stage)
    seq = _sequence(ctx)
    number = seq.get(prefix, 0)
    seq[prefix] = number + 1
    ctx.extras[SEQ_KEY] = seq
    return f"{prefix}:{number}"


def _sequence(ctx: ToolContext) -> dict[str, int]:
    # Copy on write: child contexts share `extras` values shallowly, so never mutate in place.
    current = ctx.extras.get(SEQ_KEY)
    return dict(current) if isinstance(current, Mapping) else {}
