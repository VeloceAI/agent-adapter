"""`GeminiModel`: the Gemini API (`google-genai`) as an `LLMPort`."""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from typing import Any, ClassVar

from agent_adapter.core.errors import (
    MissingDependency,
    ModelAuthError,
    ModelError,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import LLMRequest, LLMResponse
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.models._convert import get, with_parsed
from agent_adapter.models._gemini_convert import (
    GeminiStreamAssembler,
    build_gemini_request,
    parse_gemini_response,
)
from agent_adapter.models._port import Assembler, ProviderPort
from agent_adapter.models.errors import class_for_status, model_error, with_overflow

__all__ = ["GeminiModel", "build_gemini_request", "map_gemini_error", "parse_gemini_response"]

PROVIDER = "gemini"


class GeminiModel(ProviderPort):
    """Gemini as an `LLMPort` (`google-genai>=2.30`; `pip install "agent-adapter[gemini]"`).

    `defaults` and `request.extra` are `GenerateContentConfig` fields (merged into `config`).

    Args:
        model: Model id sent when the request names none.
        client: A `genai.Client` the host owns; else one is created per event loop.
        client_kwargs: `genai.Client(...)` kwargs (`api_key`, `vertexai`, `http_options`).
        cache_plugin: Prompt-cache plug-in that edits the SDK kwargs (it gets the client).
        thinking: `ThinkingConfig` fields; `request.reasoning` sets `thinking_level` over them.
        defaults: `GenerateContentConfig` fields applied under the request fields.

    Raises:
        MissingDependency: no `client` and google-genai is not installed.
    """

    provider: ClassVar[str] = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        cache_plugin: PromptCachePlugin | None = None,
        thinking: Mapping[str, Any] | None = None,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.thinking = dict(thinking or {})
        super().__init__(
            model,
            client=client,
            client_kwargs=client_kwargs,
            cache_plugin=cache_plugin,
            defaults=defaults,
        )

    def _require_sdk(self) -> None:
        try:
            import google.genai  # noqa: F401
        except ImportError as exc:
            raise MissingDependency(
                'GeminiModel needs google-genai: pip install "agent-adapter[gemini]"'
            ) from exc

    def _new_client(self) -> Any:
        from google import genai

        return genai.Client(**self._client_kwargs)

    async def _close_client(self, client: Any) -> None:
        await client.aio.aclose()

    def _build(self, request: LLMRequest, model: str) -> dict[str, Any]:
        kwargs = build_gemini_request(request, model=model)
        if self.thinking:
            config = kwargs["config"]
            config["thinking_config"] = {**self.thinking, **config.get("thinking_config", {})}
        return kwargs

    def _merge(self, kwargs: dict[str, Any], extra: Mapping[str, Any], *, under: bool) -> None:
        super()._merge(kwargs.setdefault("config", {}), extra, under=under)

    async def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        return await _aio(client).models.generate_content(**kwargs)

    async def _open_stream(self, client: Any, kwargs: dict[str, Any]) -> AsyncIterable[Any]:
        stream: AsyncIterable[Any] = await _aio(client).models.generate_content_stream(**kwargs)
        return stream

    def _parse(self, raw: Any) -> LLMResponse:
        return parse_gemini_response(raw)

    def _assembler(self, model: str) -> Assembler:
        return GeminiStreamAssembler()

    def _finish(
        self, response: LLMResponse, request: LLMRequest, raw: Any, kwargs: Mapping[str, Any]
    ) -> LLMResponse:
        return with_parsed(response, request.output_schema, sdk_parsed=get(raw, "parsed"))

    def _map_error(self, exc: BaseException, model: str) -> ModelError | None:
        return map_gemini_error(exc, model=model)

    def _map_client_error(self, exc: BaseException, model: str) -> ModelError | None:
        # genai.Client() raises a plain ValueError for a missing API key / project.
        if isinstance(exc, ValueError):
            return model_error(ModelAuthError, exc, provider=PROVIDER, model=model)
        return map_gemini_error(exc, model=model)


def _aio(client: Any) -> Any:
    """The async surface of a `genai.Client` (a host may pass `client.aio` directly)."""
    return getattr(client, "aio", client)


def map_gemini_error(exc: BaseException, *, model: str) -> ModelError | None:
    """The `ModelError` for a google-genai or httpx exception; None when unmapped."""
    cls = _genai_class(exc) or _google_auth_class(exc) or _httpx_class(exc)
    if cls is None:
        return None
    cls = with_overflow(cls, exc)
    response = getattr(exc, "response", None)
    code = getattr(exc, "code", None)
    return model_error(
        cls,
        exc,
        provider=PROVIDER,
        model=model,
        status_code=code if isinstance(code, int) else None,
        headers=getattr(response, "headers", None),
    )


def _genai_class(exc: BaseException) -> type[ModelError] | None:
    try:
        from google.genai import errors
    except ImportError:
        return None
    if isinstance(exc, errors.ServerError):
        return ModelUnavailable
    if isinstance(exc, errors.APIError):  # ClientError (4xx) and any other status
        return class_for_status(exc.code)
    if isinstance(exc, errors.UnknownApiResponseError):
        return ModelError
    return None


def _google_auth_class(exc: BaseException) -> type[ModelError] | None:
    """google-auth credential errors (Vertex AI / ADC): resolved lazily, so they arrive per call."""
    try:
        from google.auth import exceptions
    except ImportError:
        return None
    if isinstance(exc, exceptions.TimeoutError):
        return ModelTimeout
    if isinstance(exc, exceptions.TransportError):
        return ModelUnavailable
    if isinstance(exc, exceptions.GoogleAuthError):
        return ModelAuthError
    return None


def _httpx_class(exc: BaseException) -> type[ModelError] | None:
    try:
        import httpx
    except ImportError:
        return None
    if isinstance(exc, httpx.TimeoutException):
        return ModelTimeout
    if isinstance(exc, httpx.TransportError):
        return ModelUnavailable
    return None
