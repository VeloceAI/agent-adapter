"""Compression before one model call: the context window, events, and the overflow retry."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Literal

from agent_adapter.context.compress import CompressionReport, compress
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.messages import Message
from agent_adapter.core.ports import CapabilityLookup
from agent_adapter.cost.prices import Prices
from agent_adapter.runtime.ids import new_id
from agent_adapter.tracing.events import EventType, emit, make_event

logger = logging.getLogger(__name__)

Trigger = Literal["threshold", "overflow"]
_SKIPPED = frozenset({"under_threshold", "under_target"})


def check_policy(policy: object) -> CompressionPolicy:
    """`policy`, checked.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy`.
    """
    if not isinstance(policy, CompressionPolicy):
        raise ConfigError(f"compression must be a CompressionPolicy, got {type(policy).__name__}")
    return policy


def context_window(
    policy: CompressionPolicy, model: str, caps: CapabilityLookup | None = None
) -> int | None:
    """The token window a percent policy applies to: its own, else the model's prompt window.

    The prompt window is the caps' `context_window` less `max_output_tokens` (room for the
    reply). Returns None for a policy given in tokens (`start_when_tokens_over`).

    Raises:
        ConfigError: a percent policy without `model_max_tokens` for a model whose caps
            carry no `context_window`.
    """
    if policy.model_max_tokens is not None or policy.start_when_tokens_over is not None:
        return policy.model_max_tokens
    lookup = caps if caps is not None else Prices.default().caps
    window = lookup(model).prompt_window()
    if window is None:
        raise ConfigError(
            f"no context window is known for model {model!r}: set "
            "CompressionPolicy(model_max_tokens=...), add context_window to the model's caps "
            "in your price table, or use start_when_tokens_over and shrink_to_tokens"
        )
    return window


async def compress_for_model(
    messages: Sequence[Message],
    policy: CompressionPolicy,
    *,
    provider: str,
    model: str,
    ctx: ToolContext,
    caps: CapabilityLookup | None = None,
    trigger: Trigger = "threshold",
) -> tuple[list[Message], CompressionReport]:
    """Compress the messages of one call to `model`; emit `context.compressed` when it ran.

    Each step skipped for an error (`on_step_error="skip"`) emits `context.compress_failed`.

    `trigger="overflow"` compresses even under the start threshold (the provider said the
    request is too long).

    Raises:
        ConfigError: as `compress`, or no context window is known (see `context_window`).
    """
    started = time.perf_counter()
    ids = {"provider": provider, "model": model, "trigger": trigger}
    try:
        window = context_window(policy, model, caps)
        result, report = await compress(
            messages, policy, model_max_tokens=window, force=trigger == "overflow", ctx=ctx
        )
    except Exception as exc:
        log_failure(
            logger,
            logging.WARNING,
            exc,
            "context compression failed for %s %s (run %s)",
            provider,
            model,
            ctx.run_id,
        )
        _emit(
            ctx,
            EventType.CONTEXT_COMPRESS_FAILED,
            **ids,
            latency_ms=_ms(started),
            error_kind=str(getattr(exc, "kind", "internal-error")),
        )
        raise
    for failure in report.steps_failed:  # skipped steps; the call goes on
        _emit(
            ctx,
            EventType.CONTEXT_COMPRESS_FAILED,
            **ids,
            latency_ms=failure.latency_ms,
            error_kind=failure.error_kind,
            step=failure.step,
        )
    if report.skipped_reason not in _SKIPPED:
        _emit(
            ctx,
            EventType.CONTEXT_COMPRESSED,
            **ids,
            latency_ms=_ms(started),
            tokens_before=report.tokens_before,
            tokens_after=report.tokens_after,
            target_tokens=report.target_tokens,
            messages_before=report.messages_before,
            messages_after=report.messages_after,
            steps=list(report.steps_used),
            summary_cost_usd=report.summary_cost_usd,
        )
    return result, report


async def after_overflow(
    messages: Sequence[Message],
    policy: CompressionPolicy,
    *,
    provider: str,
    model: str,
    ctx: ToolContext,
    caps: CapabilityLookup | None = None,
) -> list[Message] | None:
    """The messages to retry with after a context overflow, or None to give up.

    Runs `policy.if_still_too_long` once, even under its start threshold. None when the
    policy has none or it changed nothing.

    Raises:
        ConfigError: as `compress_for_model`.
    """
    stronger = policy.if_still_too_long
    if stronger is None:
        logger.info(
            "%s %s: context overflow and no if_still_too_long policy (run %s)",
            provider,
            model,
            ctx.run_id,
        )
        return None
    logger.warning(
        "%s %s: context overflow; compressing with if_still_too_long and retrying once (run %s)",
        provider,
        model,
        ctx.run_id,
    )
    current = list(messages)
    smaller, _ = await compress_for_model(
        current, stronger, provider=provider, model=model, ctx=ctx, caps=caps, trigger="overflow"
    )
    unchanged = len(smaller) == len(current) and all(
        a is b for a, b in zip(smaller, current, strict=True)
    )
    return None if unchanged else smaller


def _emit(ctx: ToolContext, type: str, **data: object) -> None:
    """One closing event under its own call id; the enclosing call is its parent."""
    emit(ctx.sink, make_event(type, ctx, call_id=new_id("cmp_"), **data))


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
