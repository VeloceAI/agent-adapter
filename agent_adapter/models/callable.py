"""`CallableModel`: a host function as an `LLMPort`, so host routing and stats stay untouched."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

from pydantic import ValidationError

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import ModelOutputInvalid, ToolUsageError, qualified_name
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    Message,
    StreamDone,
    StreamEvent,
)
from agent_adapter.core.types import ToolCall, Usage
from agent_adapter.models._convert import with_parsed
from agent_adapter.models._port import closing, collect_stream, replay

logger = logging.getLogger(__name__)

HostResult = LLMResponse | str | Mapping[str, Any]
_USAGE_FIELDS = frozenset(Usage.model_fields)


class CallableModel:
    """Wraps `fn(request)`; it returns an `LLMResponse`, the reply text, or a response mapping.

    `fn` may be async or sync; a sync `fn` runs in a worker thread (context variables are
    copied), so a blocking local model or HTTP client never stalls the event loop. A mapping
    holds `LLMResponse` fields, including `usage` (a `Usage` or its field names), which the
    ledger prices; `provider`, `model` and `message` are filled in when missing. Exceptions
    raised by `fn` propagate unchanged.
    """

    def __init__(
        self,
        fn: Callable[[LLMRequest], Awaitable[HostResult] | HostResult],
        *,
        provider: str = "callable",
        model: str = "host",
        stream_fn: Callable[[LLMRequest], AsyncIterator[StreamEvent]] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._fn = fn
        self._stream_fn = stream_fn

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the host function (or drain `stream_fn` when `request.stream` is set).

        Raises:
            ModelOutputInvalid: the result does not form a valid response.
            ToolUsageError: `fn` returned an unsupported type.
            ConfigError: `request.compression` is invalid or could not run.
        """
        if request.compression is not None:
            from agent_adapter.context.model import generate_compressed

            return await generate_compressed(self, request, request.compression)
        if request.stream and self._stream_fn is not None:
            return await collect_stream(
                self.stream(request), provider=self.provider, model=self._model(request)
            )
        started = time.perf_counter()
        result = await self._call(request)
        response = self._coerce(result, request)
        if response.latency_ms == 0:
            response = response.model_copy(
                update={"latency_ms": (time.perf_counter() - started) * 1000}
            )
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """`stream_fn`'s events, or the generated reply replayed as deltas.

        Raises:
            ModelOutputInvalid: `stream_fn` ended without `StreamDone`.
            ConfigError: `request.compression` is invalid or could not run.
        """
        if request.compression is not None:
            from agent_adapter.context.model import stream_compressed

            async with closing(stream_compressed(self, request, request.compression)) as events:
                async for event in events:
                    yield event
            return
        if self._stream_fn is not None:
            done = False
            async with closing(self._stream_fn(request)) as events:
                async for event in events:
                    done = done or isinstance(event, StreamDone)
                    yield event
            if not done:
                raise ModelOutputInvalid(
                    f"{self.provider} {self._model(request)}: stream_fn ended without StreamDone",
                    provider=self.provider,
                    model=self._model(request),
                )
            return
        response = await self.generate(request.model_copy(update={"stream": False}))
        for event in replay(response):
            yield event

    def cache_key_fields(self) -> dict[str, Any]:
        """LLM-cache key material: which host function answers."""
        return {"api": "callable", "fn": _name(self._fn)}

    async def _call(self, request: LLMRequest) -> Any:
        if is_async(self._fn):
            return await cast("Awaitable[HostResult]", self._fn(request))
        result: Any = await asyncio.to_thread(self._fn, request)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _model(self, request: LLMRequest) -> str:
        return request.model or self.model

    def _coerce(self, result: Any, request: LLMRequest) -> LLMResponse:
        if isinstance(result, LLMResponse):
            response = result
        elif isinstance(result, str):
            response = LLMResponse(
                provider=self.provider,
                model=self._model(request),
                text=result,
                message=Message(role="assistant", content=result),
            )
        elif isinstance(result, Mapping):
            response = self._from_mapping(result, request)
        else:
            raise ToolUsageError(
                f"CallableModel fn must return LLMResponse, str or a mapping, "
                f"got {type(result).__name__}"
            )
        if response.parsed is not None:
            return response
        return with_parsed(response, request.output_schema)

    def _from_mapping(self, result: Mapping[str, Any], request: LLMRequest) -> LLMResponse:
        data = dict(result)
        self._check_usage_keys(data.get("usage"), request)
        try:
            calls = tuple(
                call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
                for call in data.get("tool_calls", ())
            )
            data["tool_calls"] = calls
            data.setdefault("provider", self.provider)
            data.setdefault("model", self._model(request))
            data.setdefault("finish_reason", "tool_calls" if calls else "stop")
            data.setdefault(
                "message",
                Message(role="assistant", content=data.get("text", ""), tool_calls=calls),
            )
            return LLMResponse.model_validate(data)
        except (ValidationError, TypeError) as exc:
            logger.warning("%s: host function returned an invalid response mapping", self.provider)
            raise ModelOutputInvalid(
                f"{self.provider}: the host function returned an invalid response mapping",
                provider=self.provider,
                model=self._model(request),
            ) from exc

    def _check_usage_keys(self, usage: Any, request: LLMRequest) -> None:
        """Reject provider-shaped usage keys, which would otherwise be dropped and bill $0."""
        if not isinstance(usage, Mapping):
            return
        unknown = sorted(str(key) for key in usage if key not in _USAGE_FIELDS)
        if unknown:
            logger.warning("%s: host usage has unknown keys %s", self.provider, unknown)
            raise ModelOutputInvalid(
                f"{self.provider}: usage keys {unknown} are not Usage fields; use "
                "input_tokens/output_tokens or agent_adapter.models.normalise_usage()",
                provider=self.provider,
                model=self._model(request),
            )


def _name(fn: object) -> str:
    """`module.qualname` of a function, or of the class of a callable object."""
    qualname = getattr(fn, "__qualname__", None)
    if isinstance(qualname, str):
        return f"{getattr(fn, '__module__', '')}.{qualname}"
    return qualified_name(type(fn))
