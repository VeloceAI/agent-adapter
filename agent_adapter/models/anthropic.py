"""`AnthropicModel`: the Anthropic Messages API as an `LLMPort`."""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from typing import Any, ClassVar

from agent_adapter.core.errors import (
    MissingDependency,
    ModelAuthError,
    ModelBadRequest,
    ModelError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import LLMRequest, LLMResponse
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.models._anthropic_convert import (
    AnthropicStreamAssembler,
    build_anthropic_request,
    parse_anthropic_message,
)
from agent_adapter.models._port import Assembler, ProviderPort, warm_platform_cache
from agent_adapter.models.errors import (
    class_for_error_type,
    class_for_status,
    error_type,
    model_error,
    sdk_error,
    with_overflow,
)

__all__ = [
    "AnthropicModel",
    "build_anthropic_request",
    "map_anthropic_error",
    "parse_anthropic_message",
]

PROVIDER = "anthropic"
_AUTH_MISSING = "Could not resolve authentication method"


class AnthropicModel(ProviderPort):
    """Anthropic as an `LLMPort` (`anthropic>=2.3`; `pip install "agent-adapter[anthropic]"`).

    Args:
        model: Model id sent when the request names none.
        client: An `AsyncAnthropic` the host owns; else one is created per event loop.
        client_kwargs: `AsyncAnthropic(...)` kwargs, e.g. `{"max_retries": 0}` for fast failover.
        cache_plugin: Prompt-cache plug-in that edits the SDK kwargs.
        max_tokens: `max_tokens` when the request sets no `max_output_tokens` (required by the API).
        defaults: SDK kwargs applied under the request fields.

    Raises:
        MissingDependency: no `client` and the anthropic SDK is not installed.
    """

    provider: ClassVar[str] = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        cache_plugin: PromptCachePlugin | None = None,
        max_tokens: int = 4096,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        warm_platform_cache()  # before the SDK asks from a worker thread (Windows)
        self.max_tokens = max_tokens
        super().__init__(
            model,
            client=client,
            client_kwargs=client_kwargs,
            cache_plugin=cache_plugin,
            defaults=defaults,
        )

    def _require_sdk(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise MissingDependency(
                'AnthropicModel needs the anthropic SDK: pip install "agent-adapter[anthropic]"'
            ) from exc

    def _new_client(self) -> Any:
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(**self._client_kwargs)

    async def _close_client(self, client: Any) -> None:
        await client.close()

    def _build(self, request: LLMRequest, model: str) -> dict[str, Any]:
        return build_anthropic_request(request, model=model, max_tokens=self.max_tokens)

    async def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        return await client.messages.create(**kwargs)

    async def _open_stream(self, client: Any, kwargs: dict[str, Any]) -> AsyncIterable[Any]:
        stream: AsyncIterable[Any] = await client.messages.create(**{**kwargs, "stream": True})
        return stream

    def _parse(self, raw: Any) -> LLMResponse:
        return parse_anthropic_message(raw)

    def _assembler(self, model: str) -> Assembler:
        return AnthropicStreamAssembler(model=model)

    def _map_error(self, exc: BaseException, model: str) -> ModelError | None:
        return map_anthropic_error(exc, model=model)

    def _map_client_error(self, exc: BaseException, model: str) -> ModelError | None:
        import anthropic

        if isinstance(exc, anthropic.AnthropicError):
            return model_error(ModelAuthError, exc, provider=PROVIDER, model=model)
        return None


def map_anthropic_error(exc: BaseException, *, model: str) -> ModelError | None:
    """The `ModelError` for an anthropic SDK exception; None for other exceptions.

    Every `anthropic.AnthropicError` maps: stream `error` events arrive as an `APIStatusError`
    with the stream's 200 status and are typed by their body.
    """
    try:
        import anthropic
    except ImportError:
        return None
    if isinstance(exc, TypeError) and _AUTH_MISSING in str(exc):
        # The SDK raises a bare TypeError when no api_key / auth_token can be resolved.
        return model_error(ModelAuthError, exc, provider=PROVIDER, model=model)
    if not isinstance(exc, anthropic.AnthropicError):
        return None
    cls = with_overflow(_anthropic_class(exc, anthropic), exc, getattr(exc, "body", None))
    return sdk_error(cls, exc, provider=PROVIDER, model=model)


def _anthropic_class(exc: BaseException, anthropic: Any) -> type[ModelError]:
    table: tuple[tuple[Any, type[ModelError]], ...] = (
        ((anthropic.APITimeoutError, anthropic.DeadlineExceededError), ModelTimeout),
        (anthropic.APIConnectionError, ModelUnavailable),
        (anthropic.RateLimitError, ModelRateLimited),
        (
            (
                anthropic.InternalServerError,
                anthropic.OverloadedError,
                anthropic.ServiceUnavailableError,
            ),
            ModelUnavailable,
        ),
        ((anthropic.AuthenticationError, anthropic.PermissionDeniedError), ModelAuthError),
        (
            (
                anthropic.BadRequestError,
                anthropic.NotFoundError,
                anthropic.ConflictError,
                anthropic.RequestTooLargeError,
                anthropic.UnprocessableEntityError,
            ),
            ModelBadRequest,
        ),
    )
    for sdk_cls, ours in table:
        if isinstance(exc, sdk_cls):
            return ours
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code < 400:
            return class_for_error_type(error_type(exc.body))
        return class_for_status(exc.status_code)
    return ModelError
