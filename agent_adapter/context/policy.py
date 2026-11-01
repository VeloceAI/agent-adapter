"""CompressionPolicy: when to compress, how far, with which steps and token counter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from agent_adapter.context.defaults import default_steps
from agent_adapter.context.step import CompressionStep
from agent_adapter.context.tokens import TokenCounter
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.ports import CachePort, LLMPort

DEFAULT_START_WHEN_FULL_PERCENT = 90
DEFAULT_SHRINK_TO_PERCENT = 60
StepErrorMode = Literal["skip", "raise"]


@dataclass(frozen=True)
class CompressionLimits:
    """The two token counts one compression works with."""

    start_when_tokens_over: int  # compress only when the conversation has more tokens
    shrink_to_tokens: int  # then stop as soon as it has this many or fewer


@dataclass(frozen=True, kw_only=True)
class CompressionPolicy:
    """When and how far to compress, and with which steps; give percents or token counts.

    Raises:
        ConfigError: a setting is out of range, or percents and token counts are mixed.
    """

    model_max_tokens: int | None = None  # None: the caller supplies it (e.g. price-table caps)
    start_when_full_percent: int | None = None  # default 90
    shrink_to_percent: int | None = None  # default 60
    start_when_tokens_over: int | None = None
    shrink_to_tokens: int | None = None
    steps: Sequence[CompressionStep] = field(default=())  # () or []: the default steps
    count_tokens: TokenCounter | None = None  # None: chars/4 estimate
    if_still_too_long: CompressionPolicy | None = None  # used once on a context overflow
    summary_model: LLMPort | None = None  # adds Summarize to the default steps
    cache: CachePort | None = None  # summaries; a step's own cache wins over it
    # "skip": a failing step is logged and skipped, and the call goes on; "raise": it fails.
    on_step_error: StepErrorMode = "skip"

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        _positive("model_max_tokens", self.model_max_tokens)
        if self.start_when_tokens_over is not None or self.shrink_to_tokens is not None:
            self._check_tokens()
        else:
            self._check_percents()
        for step in self.steps:
            if not isinstance(step, CompressionStep):
                raise ConfigError(f"steps must be compression steps, got {type(step).__name__}")
        if self.count_tokens is not None and not callable(self.count_tokens):
            raise ConfigError("count_tokens must be a callable(list[Message]) -> int")
        if self.if_still_too_long is not None and not isinstance(
            self.if_still_too_long, CompressionPolicy
        ):
            raise ConfigError("if_still_too_long must be a CompressionPolicy")
        if self.cache is not None and not isinstance(self.cache, CachePort):
            raise ConfigError("cache must be a CachePort, e.g. MemoryCache()")
        if self.on_step_error not in ("skip", "raise"):
            raise ConfigError(
                f"on_step_error must be 'skip' or 'raise', got {self.on_step_error!r}"
            )

    def _check_tokens(self) -> None:
        if self.start_when_full_percent is not None or self.shrink_to_percent is not None:
            raise ConfigError(
                "give either start_when_full_percent/shrink_to_percent or "
                "start_when_tokens_over/shrink_to_tokens, not both"
            )
        if self.start_when_tokens_over is None or self.shrink_to_tokens is None:
            raise ConfigError("start_when_tokens_over and shrink_to_tokens go together")
        _positive("start_when_tokens_over", self.start_when_tokens_over)
        _positive("shrink_to_tokens", self.shrink_to_tokens)
        if self.shrink_to_tokens >= self.start_when_tokens_over:
            raise ConfigError("shrink_to_tokens must be less than start_when_tokens_over")
        if (
            self.model_max_tokens is not None
            and self.start_when_tokens_over > self.model_max_tokens
        ):
            raise ConfigError("start_when_tokens_over must not exceed model_max_tokens")

    def _check_percents(self) -> None:
        start = self.start_when_full_percent
        shrink = self.shrink_to_percent
        _percent("start_when_full_percent", start)
        _percent("shrink_to_percent", shrink)
        start = DEFAULT_START_WHEN_FULL_PERCENT if start is None else start
        shrink = DEFAULT_SHRINK_TO_PERCENT if shrink is None else shrink
        if shrink >= start:
            raise ConfigError(
                f"shrink_to_percent ({shrink}) must be less than start_when_full_percent ({start})"
            )

    def limits(self, model_max_tokens: int | None = None) -> CompressionLimits:
        """The token limits; percents apply to the policy's `model_max_tokens`, else this one.

        Raises:
            ConfigError: percents are used and no model size is known, or it is not positive.
        """
        if self.start_when_tokens_over is not None and self.shrink_to_tokens is not None:
            return CompressionLimits(self.start_when_tokens_over, self.shrink_to_tokens)
        window = self.model_max_tokens if self.model_max_tokens is not None else model_max_tokens
        if window is None:
            raise ConfigError(
                "the model's max tokens are unknown: set model_max_tokens on the policy, "
                "or use start_when_tokens_over and shrink_to_tokens"
            )
        _positive("model_max_tokens", window)
        start = self.start_when_full_percent or DEFAULT_START_WHEN_FULL_PERCENT
        shrink = self.shrink_to_percent or DEFAULT_SHRINK_TO_PERCENT
        return CompressionLimits(
            start_when_tokens_over=max(1, window * start // 100),
            shrink_to_tokens=max(0, window * shrink // 100),
        )

    def resolved_steps(self) -> tuple[CompressionStep, ...]:
        """The steps to run, in order: the policy's, else the defaults."""
        if self.steps:
            return tuple(self.steps)
        return default_steps(self.summary_model)


def _positive(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be an int >= 1, got {value!r}")


def _percent(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ConfigError(f"{name} must be an int from 1 to 100, got {value!r}")
