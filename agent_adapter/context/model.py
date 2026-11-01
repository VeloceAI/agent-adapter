"""`CompressingModel` and the per-call `LLMRequest.compression` path every built-in port uses."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from agent_adapter.context.policy import CompressionPolicy
from agent_adapter.context.run import (
    after_overflow,
    check_policy,
    compress_for_model,
    context_window,
)
from agent_adapter.context.step import plain
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ModelContextOverflow
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message, StreamEvent
from agent_adapter.core.ports import CapabilityLookup, LLMPort
from agent_adapter.runtime.current import resolve_context

When = Callable[[LLMRequest, ToolContext], bool | Awaitable[bool]]
# Set on a request already compressed, so a nested wrapper neither compresses nor retries again.
COMPRESSED_MARK = "aa.compressed"


def is_compressed(request: LLMRequest) -> bool:
    """True when an outer compression layer already handled `request`."""
    return request.metadata.get(COMPRESSED_MARK) == "1"


def marked(request: LLMRequest) -> LLMRequest:
    """`request` marked as compressed, without a policy of its own."""
    metadata = {**request.metadata, COMPRESSED_MARK: "1"}
    return request.model_copy(update={"metadata": metadata, "compression": None})


class Port(Protocol):
    """What compression needs of a model: its names, `generate` and `stream`."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]: ...


class CompressingModel:
    """An `LLMPort` that compresses every request's messages with `policy` before `model`.

    A request's own `compression` wins over `policy`. `when(request, ctx)` (sync or async)
    can skip compression for some calls. On `ModelContextOverflow` the policy's
    `if_still_too_long` runs once and the call is retried once. Nested layers (a wrapper in
    a wrapper, or `run_tool_loop(compression=)` around one) compress and retry once in all:
    an inner layer passes a request the outer one compressed straight through.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy`, `when` is not callable, or no
            context window is known for `model` (see the policy's `model_max_tokens`).
    """

    def __init__(
        self,
        model: LLMPort,
        policy: CompressionPolicy,
        when: When | None = None,
        *,
        caps: CapabilityLookup | None = None,
    ) -> None:
        self.wrapped = model
        self.policy = check_policy(policy)
        if when is not None and not callable(when):
            raise ConfigError("when must be a callable(request, ctx) -> bool")
        self.when = when
        self.caps = caps
        context_window(policy, model.model, caps)  # fail at setup, not on the first call

    @property
    def provider(self) -> str:
        return self.wrapped.provider

    @property
    def model(self) -> str:
        return self.wrapped.model

    def cache_key_fields(self) -> dict[str, Any]:
        """The wrapped port's key material plus the policy's settings."""
        return {**self.wrapped.cache_key_fields(), "compression": policy_key(self.policy)}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """The wrapped port's reply to the compressed request.

        Raises:
            ConfigError: the policy or a step's output is invalid.
            ModelError: the wrapped port's error (a second overflow included).
        """
        policy = await self._policy_for(request)
        if policy is None:
            return await self.wrapped.generate(request)
        return await generate_compressed(self.wrapped, request, policy, caps=self.caps)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """The wrapped port's stream of the compressed request.

        Raises:
            ConfigError: as `generate`.
            ModelError: as `generate`; an overflow after the first event is not retried.
        """
        policy = await self._policy_for(request)
        if policy is None:
            events = self.wrapped.stream(request)
        else:
            events = stream_compressed(self.wrapped, request, policy, caps=self.caps)
        try:
            async for event in events:
                yield event
        finally:
            await _close(events)

    async def _policy_for(self, request: LLMRequest) -> CompressionPolicy | None:
        if is_compressed(request):
            return None
        if request.compression is not None:
            return check_policy(request.compression)
        if self.when is None:
            return self.policy
        decided = self.when(request, resolve_context(None))
        if inspect.isawaitable(decided):
            decided = await decided
        return self.policy if decided else None


async def generate_compressed(
    port: Port,
    request: LLMRequest,
    policy: Any,
    *,
    caps: CapabilityLookup | None = None,
    ctx: ToolContext | None = None,
) -> LLMResponse:
    """`port.generate` on the compressed request; one overflow retry with `if_still_too_long`.

    The request sent has `compression=None`, so the port does not compress it again.

    Raises:
        ConfigError: `policy` is not a `CompressionPolicy`, or compressing failed.
        ModelError: the port's error (a second overflow included).
    """
    checked = check_policy(policy)
    if is_compressed(request):
        return await port.generate(marked(request))
    target = _Target(port, request, checked, caps, resolve_context(ctx))
    sent = await target.first()
    try:
        return await port.generate(sent)
    except ModelContextOverflow:
        retry = await target.after_overflow(sent)
        if retry is None:
            raise
    return await port.generate(retry)


async def stream_compressed(
    port: Port,
    request: LLMRequest,
    policy: Any,
    *,
    caps: CapabilityLookup | None = None,
    ctx: ToolContext | None = None,
) -> AsyncIterator[StreamEvent]:
    """`port.stream` on the compressed request; an overflow before the first event retries once.

    Raises:
        ConfigError: as `generate_compressed`.
        ModelError: the port's error; an overflow after the first event propagates.
    """
    checked = check_policy(policy)
    if is_compressed(request):
        once = port.stream(marked(request))
        try:
            async for event in once:
                yield event
        finally:
            await _close(once)
        return
    target = _Target(port, request, checked, caps, resolve_context(ctx))
    sent = await target.first()
    started = False
    events = port.stream(sent)
    try:
        async for event in events:
            started = True
            yield event
        return
    except ModelContextOverflow:
        if started:
            raise
        retry = await target.after_overflow(sent)
        if retry is None:
            raise
    finally:
        await _close(events)
    again = port.stream(retry)
    try:
        async for event in again:
            yield event
    finally:
        await _close(again)


class _Target:
    """One compressed call: the port, the model id, the policy and where events go."""

    def __init__(
        self,
        port: Port,
        request: LLMRequest,
        policy: CompressionPolicy,
        caps: CapabilityLookup | None,
        ctx: ToolContext,
    ) -> None:
        self.request = request
        self.policy = policy
        self.provider = port.provider
        self.model = request.model or port.model
        self.caps = caps
        self.ctx = ctx

    async def first(self) -> LLMRequest:
        messages, _ = await compress_for_model(
            self.request.messages,
            self.policy,
            provider=self.provider,
            model=self.model,
            ctx=self.ctx,
            caps=self.caps,
        )
        return _sending(self.request, messages)

    async def after_overflow(self, sent: LLMRequest) -> LLMRequest | None:
        smaller = await after_overflow(
            sent.messages,
            self.policy,
            provider=self.provider,
            model=self.model,
            ctx=self.ctx,
            caps=self.caps,
        )
        return None if smaller is None else _sending(sent, smaller)


def policy_key(policy: CompressionPolicy) -> dict[str, Any]:
    """JSON-safe settings of a policy, for cache keys: limits and each step's config."""
    stronger = policy.if_still_too_long
    return {
        "model_max_tokens": policy.model_max_tokens,
        "start_when_full_percent": policy.start_when_full_percent,
        "shrink_to_percent": policy.shrink_to_percent,
        "start_when_tokens_over": policy.start_when_tokens_over,
        "shrink_to_tokens": policy.shrink_to_tokens,
        "steps": [_step_key(step) for step in policy.resolved_steps()],
        "if_still_too_long": None if stronger is None else policy_key(stronger),
        "count_tokens": None if policy.count_tokens is None else plain(policy.count_tokens),
        "on_step_error": policy.on_step_error,
    }


def _step_key(step: object) -> Any:
    config = getattr(step, "config", None)
    if callable(config):
        return config()
    named = type(step)
    return f"{named.__module__}.{named.__qualname__}"


def _sending(request: LLMRequest, messages: list[Message] | tuple[Message, ...]) -> LLMRequest:
    return marked(request).model_copy(update={"messages": tuple(messages)})


async def _close(events: AsyncIterator[Any]) -> None:
    close = getattr(events, "aclose", None)
    if close is not None:
        await close()
