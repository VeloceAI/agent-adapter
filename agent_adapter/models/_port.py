"""The shared skeleton of the SDK-backed ports: clients, kwargs, plug-in, errors, streaming."""

from __future__ import annotations

import inspect
import logging
import platform
import sys
import sysconfig
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Protocol, TypeVar

from agent_adapter.core.errors import ModelError, ModelOutputInvalid
from agent_adapter.core.messages import (
    LLMRequest,
    LLMResponse,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
)
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.models._convert import dumps_arguments, with_parsed
from agent_adapter.runtime.loop_local import LoopLocal

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Assembler(Protocol):
    """Collects stream chunks into deltas and a final response."""

    def feed(self, chunk: Any) -> list[TextDelta | ToolCallDelta]: ...

    def response(self) -> LLMResponse: ...


class ProviderPort(ABC):
    """Base of `OpenAIModel`, `GeminiModel` and `AnthropicModel` (an `LLMPort`).

    Subclasses supply the SDK pieces: client creation, the call, parsing and the error map.
    """

    provider: ClassVar[str]
    api: str = ""

    def __init__(
        self,
        model: str,
        *,
        client: Any | None,
        client_kwargs: Mapping[str, Any] | None,
        cache_plugin: PromptCachePlugin | None,
        defaults: Mapping[str, Any] | None,
    ) -> None:
        self.model = model
        self.cache_plugin = cache_plugin
        self._client = client
        self._client_kwargs = dict(client_kwargs or {})
        self._defaults = dict(defaults or {})
        self._clients: LoopLocal[Any] | None = None
        if client is None:
            self._require_sdk()
            self._clients = LoopLocal(self._new_client, close=self._close_client)

    # --- subclass hooks -------------------------------------------------------------------

    @abstractmethod
    def _require_sdk(self) -> None:
        """Raise MissingDependency when the SDK is not installed."""

    @abstractmethod
    def _new_client(self) -> Any:
        """A new SDK client from `client_kwargs`."""

    @abstractmethod
    async def _close_client(self, client: Any) -> None:
        """Close a client this port created."""

    @abstractmethod
    def _build(self, request: LLMRequest, model: str) -> dict[str, Any]:
        """SDK kwargs for `request`."""

    @abstractmethod
    async def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        """One non-streamed SDK call."""

    @abstractmethod
    async def _open_stream(self, client: Any, kwargs: dict[str, Any]) -> AsyncIterable[Any]:
        """Start a streamed SDK call."""

    @abstractmethod
    def _parse(self, raw: Any) -> LLMResponse:
        """A normalised response from the SDK reply."""

    @abstractmethod
    def _assembler(self, model: str) -> Assembler:
        """A fresh stream assembler."""

    @abstractmethod
    def _map_error(self, exc: BaseException, model: str) -> ModelError | None:
        """The mapped error, or None to let `exc` propagate unchanged."""

    def _map_client_error(self, exc: BaseException, model: str) -> ModelError | None:
        """The mapped error for a failure creating the SDK client (credentials, config)."""
        return self._map_error(exc, model)

    def _merge(self, kwargs: dict[str, Any], extra: Mapping[str, Any], *, under: bool) -> None:
        """Merge `extra` into SDK kwargs: under the built fields, or over them."""
        for key, value in extra.items():
            if not under or key not in kwargs:
                kwargs[key] = value

    def _finish(
        self, response: LLMResponse, request: LLMRequest, raw: Any, kwargs: Mapping[str, Any]
    ) -> LLMResponse:
        """Post-process a parsed reply (`raw` is None when streamed); fills `parsed`."""
        return with_parsed(response, request.output_schema)

    # --- LLMPort --------------------------------------------------------------------------

    def client(self) -> Any:
        """The host's client, or this event loop's client (created on first use)."""
        if self._clients is None:
            return self._client
        return self._clients.get()

    def cache_key_fields(self) -> dict[str, Any]:
        """LLM-cache key material: the API and the plug-in options."""
        plugin = self.cache_plugin.key_fields() if self.cache_plugin is not None else None
        return {"api": self.api, "plugin": plugin}

    async def kwargs_for(self, request: LLMRequest, client: Any) -> dict[str, Any]:
        """SDK kwargs: defaults under the request fields, then the plug-in, then `extra`."""
        model = request.model or self.model
        kwargs = self._build(request, model)
        self._merge(kwargs, self._defaults, under=True)
        plugin = self.cache_plugin
        if plugin is not None and self.provider in plugin.providers:
            kwargs = await plugin.apply(request, kwargs, model=model, client=client)
        self._merge(kwargs, request.extra, under=False)
        return kwargs

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """One model call; `request.stream=True` streams internally and returns the result.

        Raises:
            ModelError: a mapped provider failure (the SDK exception is the `__cause__`).
            ConfigError: `request.compression` is invalid or could not run.
        """
        if request.compression is not None:
            from agent_adapter.context.model import generate_compressed

            return await generate_compressed(self, request, request.compression)
        if request.stream:
            return await collect_stream(
                self.stream(request), provider=self.provider, model=request.model or self.model
            )
        model = request.model or self.model
        client, kwargs = await self._prepare(request, model)
        started = time.perf_counter()
        try:
            raw = await self._create(client, kwargs)
        except Exception as exc:
            mapped = self._map_error(exc, model)
            if mapped is None:
                raise
            self._log_failure(mapped)
            raise mapped from exc
        response = self._complete(self._parse(raw), request, raw, kwargs, started)
        logger.debug("%s %s finished: %s", self.provider, model, response.finish_reason)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Text and tool-call deltas, then `StreamDone` with the assembled response.

        Raises:
            ModelError: a mapped provider failure, before or during the stream.
            ConfigError: `request.compression` is invalid or could not run.
        """
        if request.compression is not None:
            from agent_adapter.context.model import stream_compressed

            async with closing(stream_compressed(self, request, request.compression)) as events:
                async for event in events:
                    yield event
            return
        model = request.model or self.model
        client, kwargs = await self._prepare(request, model)
        assembler = self._assembler(model)
        started = time.perf_counter()
        try:
            chunks = await self._open_stream(client, kwargs)
            try:
                async for chunk in chunks:
                    for delta in assembler.feed(chunk):
                        yield delta
            finally:
                await _close_stream(chunks)
            response = assembler.response()
        except ModelError as exc:  # raised by the assembler (stream error event, early end)
            self._log_failure(exc)
            raise
        except Exception as exc:
            mapped = self._map_error(exc, model)
            if mapped is None:
                raise
            self._log_failure(mapped)
            raise mapped from exc
        yield StreamDone(response=self._complete(response, request, None, kwargs, started))

    async def _prepare(self, request: LLMRequest, model: str) -> tuple[Any, dict[str, Any]]:
        """The client and SDK kwargs; SDK errors raised on the way are mapped too."""
        try:
            client = self.client()
        except Exception as exc:
            mapped = self._map_client_error(exc, model)
            if mapped is None:
                raise
            self._log_failure(mapped)
            raise mapped from exc
        try:
            return client, await self.kwargs_for(request, client)
        except Exception as exc:
            mapped = self._map_error(exc, model)
            if mapped is None:
                raise
            self._log_failure(mapped)
            raise mapped from exc

    def _complete(
        self,
        response: LLMResponse,
        request: LLMRequest,
        raw: Any,
        kwargs: Mapping[str, Any],
        started: float,
    ) -> LLMResponse:
        response = self._finish(response, request, raw, kwargs)
        update: dict[str, Any] = {"latency_ms": (time.perf_counter() - started) * 1000}
        if not response.model:
            update["model"] = request.model or self.model
        return response.model_copy(update=update)

    def _log_failure(self, error: ModelError) -> None:
        logger.warning(
            "%s %s call failed: %s (status %s, request %s)",
            self.provider,
            error.model,
            error.kind,
            error.status_code,
            error.request_id or "-",
        )


_platform_lock = threading.Lock()
_platform_warmed = False
_PLATFORM_VARIANTS = ((False, False), (True, False), (False, True), (True, True))
# sysconfig.get_platform() of Windows builds -> platform.machine() (what WMI would report).
_BUILD_MACHINES = {"win-amd64": "AMD64", "win32": "x86", "win-arm64": "ARM64", "win-arm32": "ARM"}


def _skip_wmi(*_keys: str) -> Any:
    raise OSError("WMI skipped")


def warm_platform_cache() -> None:
    """On Windows, fill the `platform` module's caches once per process, without WMI."""
    # cpython#125315: on Windows, platform's WMI queries time out under load and the abandoned
    # query thread can later kill the process; the SDKs also first query from a worker thread.
    # With `_wmi_query` failing, CPython uses its registry / getwindowsversion / env fallback.
    global _platform_warmed
    if sys.platform != "win32":
        return
    with _platform_lock:
        original = getattr(platform, "_wmi_query", None)
        if _platform_warmed or original is None:
            return
        setattr(platform, "_wmi_query", _skip_wmi)  # noqa: B010 - private, untyped attribute
        try:
            _fill_machine(platform.uname())
            for aliased, terse in _PLATFORM_VARIANTS:
                platform.platform(aliased=aliased, terse=terse)
            _ = platform.uname().processor
        finally:
            setattr(platform, "_wmi_query", original)  # noqa: B010
        _platform_warmed = True


def _fill_machine(info: platform.uname_result) -> None:
    """Without WMI and PROCESSOR_ARCHITECTURE the machine is blank: take the build's."""
    if info.machine:
        return
    machine = _BUILD_MACHINES.get(sysconfig.get_platform(), "")
    if not machine:
        logger.warning("platform machine unknown (build %s)", sysconfig.get_platform())
        return
    filled = platform.uname_result(info.system, info.node, info.release, info.version, machine)
    setattr(platform, "_uname_cache", filled)  # noqa: B010 - private, untyped attribute
    cache = getattr(platform, "_platform_cache", None)
    if isinstance(cache, dict):
        cache.clear()  # platform() strings built from the blank machine


async def collect_stream(
    events: AsyncIterator[StreamEvent], *, provider: str, model: str
) -> LLMResponse:
    """The `StreamDone` response of a stream.

    Raises:
        ModelOutputInvalid: the stream ended without `StreamDone`.
    """
    final: LLMResponse | None = None
    async for event in events:
        if isinstance(event, StreamDone):
            final = event.response
    if final is None:
        raise ModelOutputInvalid(
            f"{provider} {model}: stream ended without StreamDone", provider=provider, model=model
        )
    return final


def replay(response: LLMResponse) -> list[StreamEvent]:
    """A finished response as stream events: its text, each tool call, then `StreamDone`."""
    events: list[StreamEvent] = []
    if response.text:
        events.append(TextDelta(text=response.text))
    for index, call in enumerate(response.tool_calls):
        events.append(
            ToolCallDelta(
                index=index,
                id=call.id,
                name=call.name,
                arguments_delta=dumps_arguments(call.arguments),
            )
        )
    events.append(StreamDone(response=response))
    return events


async def _close_stream(chunks: Any) -> None:
    """Close an SDK stream (`aclose` or async `close`), releasing its connection."""
    for name in ("aclose", "close"):
        method = getattr(chunks, name, None)
        if method is None:
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


@asynccontextmanager
async def closing(events: AsyncIterator[T]) -> AsyncIterator[AsyncIterator[T]]:
    """Close an inner stream when the block ends, also when the consumer stops early."""
    try:
        yield events
    finally:
        await _close_stream(events)
