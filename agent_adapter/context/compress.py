"""compress(): run a policy's steps over a conversation under the invariant guard."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from agent_adapter.context.conversation import Protection
from agent_adapter.context.guard import enforce_invariants
from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.context.step import CompressionStep, StepContext, shortens_system_prompt
from agent_adapter.context.tokens import TokenMeter
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, log_failure
from agent_adapter.core.messages import Message
from agent_adapter.core.types import ErrorInfo

logger = logging.getLogger(__name__)

SkipReason = Literal["", "under_threshold", "under_target", "nothing_to_compress"]


@dataclass(frozen=True)
class StepFailure:
    """A step that raised and was skipped (`on_step_error="skip"`); no exception text."""

    step: str
    error_kind: str
    latency_ms: float


@dataclass(frozen=True)
class CompressionReport:
    """What one compression did; counts only, never message text."""

    tokens_before: int
    tokens_after: int
    steps_used: tuple[str, ...]  # steps that changed something, in order
    summary_cost_usd: float
    skipped_reason: SkipReason  # "" when the steps ran and changed something
    target_tokens: int
    messages_before: int
    messages_after: int
    steps_failed: tuple[StepFailure, ...] = ()  # skipped because they raised

    @property
    def reached_target(self) -> bool:
        """True when the result is at or under the target."""
        return self.tokens_after <= self.target_tokens


async def compress(
    messages: Sequence[Message],
    policy: CompressionPolicy,
    *,
    model_max_tokens: int | None = None,
    force: bool = False,
    ctx: ToolContext | None = None,
) -> tuple[list[Message], CompressionReport]:
    """The compressed conversation and a report; the input is returned as is when not needed.

    Args:
        model_max_tokens: The model's size, for a percent policy without its own.
        force: Compress even under the start threshold (e.g. after a context overflow).
        ctx: The run this compression belongs to; AI steps attribute their cost to it.

    Raises:
        ConfigError: the policy, the messages, the counter or a step's output is invalid.
    """
    if not isinstance(policy, CompressionPolicy):
        raise ConfigError("compress needs a CompressionPolicy")
    original = list(messages)
    if not all(isinstance(m, Message) for m in original):
        raise ConfigError("compress needs a list of Message")
    limits = policy.limits(model_max_tokens)
    meter = TokenMeter(policy.count_tokens)
    before = meter.total(original)

    def report(
        result: list[Message],
        tokens: int,
        used: tuple[str, ...],
        reason: SkipReason,
        cost: float,
        failed: tuple[StepFailure, ...] = (),
    ) -> CompressionReport:
        return CompressionReport(
            tokens_before=before,
            tokens_after=tokens,
            steps_used=used,
            summary_cost_usd=cost,
            skipped_reason=reason,
            target_tokens=limits.shrink_to_tokens,
            messages_before=len(original),
            messages_after=len(result),
            steps_failed=failed,
        )

    if not force and before <= limits.start_when_tokens_over:
        return original, report(original, before, (), "under_threshold", 0.0)
    if before <= limits.shrink_to_tokens:
        return original, report(original, before, (), "under_target", 0.0)
    context = StepContext(
        limits.shrink_to_tokens, meter, Protection(original), ctx, cache=policy.cache
    )
    run = _StepRun(context, skip_failures=policy.on_step_error == "skip")
    current, tokens, used = await run.steps(original, policy.resolved_steps(), before)
    reason: SkipReason = "" if used else "nothing_to_compress"
    result = report(current, tokens, used, reason, context.summary_cost_usd, tuple(run.failed))
    logger.info(
        "context compressed: %d -> %d tokens (target %d), %d -> %d messages, steps %s",
        before,
        tokens,
        limits.shrink_to_tokens,
        len(original),
        len(current),
        ",".join(used) or "none",
    )
    if not result.reached_target:
        logger.warning(
            "context still over target after compression: %d tokens, target %d",
            tokens,
            limits.shrink_to_tokens,
        )
    return current, result


class _StepRun:
    """Runs steps in order until under the target, skipping failed ones when asked."""

    def __init__(self, context: StepContext, *, skip_failures: bool) -> None:
        self.context = context
        self.skip_failures = skip_failures
        self.failed: list[StepFailure] = []

    async def steps(
        self, original: list[Message], steps: Sequence[CompressionStep], tokens: int
    ) -> tuple[list[Message], int, tuple[str, ...]]:
        """The result, its tokens and the steps that changed something.

        Raises:
            ConfigError: a step's output or setting is invalid (never skipped).
            Exception: a failing step's error, with `on_step_error="raise"`.
        """
        context = self.context
        current = original
        used: list[str] = []
        for step in steps:
            guarded = await self._one(step, current)
            if guarded is not None and not _same(current, guarded):
                used.append(step.name)
                current = guarded
                tokens = context.tokens(current)
            logger.debug("compression step %s: %d tokens", step.name, tokens)
            if context.under_target(tokens):
                break
        return current, tokens, tuple(used)

    async def _one(self, step: CompressionStep, current: list[Message]) -> list[Message] | None:
        """The guarded output of one step; None when it failed and was skipped."""
        started = time.perf_counter()
        try:
            proposed = await step.apply(list(current), self.context)
            return enforce_invariants(
                current,
                proposed,
                self.context.protection,
                step=step.name,
                allow_system_prompt_shortening=shortens_system_prompt(step),
            )
        except ConfigError as exc:
            log_failure(logger, logging.WARNING, exc, "compression step %s failed", step.name)
            raise
        except Exception as exc:
            if not self.skip_failures:
                log_failure(logger, logging.WARNING, exc, "compression step %s failed", step.name)
                raise
            log_failure(
                logger, logging.WARNING, exc, "compression step %s failed; skipped", step.name
            )
            ms = (time.perf_counter() - started) * 1000
            self.failed.append(StepFailure(step.name, ErrorInfo.from_exception(exc).kind, ms))
            return None


def _same(old: Sequence[Message], new: Sequence[Message]) -> bool:
    return len(old) == len(new) and all(a is b for a, b in zip(old, new, strict=True))
