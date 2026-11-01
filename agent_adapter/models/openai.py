"""`OpenAIModel`: the OpenAI Chat Completions and Responses APIs as an `LLMPort`."""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from typing import Any, ClassVar, Literal

from agent_adapter.core.errors import (
    ConfigError,
    MissingDependency,
    ModelAuthError,
    ModelBadRequest,
    ModelContentFiltered,
    ModelError,
    ModelOutputInvalid,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.core.messages import LLMRequest, LLMResponse
from agent_adapter.core.ports import PromptCachePlugin
from agent_adapter.models._convert import get
from agent_adapter.models._openai_chat import (
    ChatStreamAssembler,
    build_chat_request,
    parse_chat_completion,
)
from agent_adapter.models._openai_responses import (
    ResponsesStreamAssembler,
    build_responses_request,
    parse_response,
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
    "OpenAIModel",
    "build_chat_request",
    "build_responses_request",
    "map_openai_error",
    "parse_chat_completion",
    "parse_response",
]

PROVIDER = "openai"


class OpenAIModel(ProviderPort):
    """OpenAI as an `LLMPort` (`openai>=3.41`; `pip install "agent-adapter[openai]"`).

    Args:
        model: Model id sent when the request names none.
        api: "responses" (default) or "chat" (Chat Completions).
        client: An `AsyncOpenAI` the host owns; else one is created per event loop.
        client_kwargs: `AsyncOpenAI(...)` kwargs, e.g. `{"max_retries": 0}` for fast failover.
        cache_plugin: Prompt-cache plug-in that edits the SDK kwargs.
        strict_tools: Render tools strict when their `ToolDef.strict` allows it.
        defaults: SDK kwargs applied under the request fields.

    Raises:
        ConfigError: `api` is not "responses" or "chat".
        MissingDependency: no `client` and the openai SDK is not installed.
    """

    provider: ClassVar[str] = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        api: Literal["responses", "chat"] = "responses",
        client: Any | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        cache_plugin: PromptCachePlugin | None = None,
        strict_tools: bool = True,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        warm_platform_cache()  # before the SDK asks from a worker thread (Windows)
        if api not in ("responses", "chat"):
            raise ConfigError(f"OpenAIModel api must be 'responses' or 'chat', got {api!r}")
        self.api = api
        self.strict_tools = strict_tools
        super().__init__(
            model,
            client=client,
            client_kwargs=client_kwargs,
            cache_plugin=cache_plugin,
            defaults=defaults,
        )

    def _require_sdk(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise MissingDependency(
                'OpenAIModel needs the openai SDK: pip install "agent-adapter[openai]"'
            ) from exc

    def _new_client(self) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(**self._client_kwargs)

    async def _close_client(self, client: Any) -> None:
        await client.close()

    def _build(self, request: LLMRequest, model: str) -> dict[str, Any]:
        if self.api == "chat":
            return build_chat_request(request, model=model, strict_tools=self.strict_tools)
        return build_responses_request(request, model=model, strict_tools=self.strict_tools)

    async def _create(self, client: Any, kwargs: dict[str, Any]) -> Any:
        if self.api == "chat":
            return await client.chat.completions.create(**kwargs)
        return await client.responses.create(**kwargs)

    async def _open_stream(self, client: Any, kwargs: dict[str, Any]) -> AsyncIterable[Any]:
        stream: AsyncIterable[Any]
        if self.api == "chat":
            stream_options = {"include_usage": True, **kwargs.get("stream_options", {})}
            stream = await client.chat.completions.create(
                **{**kwargs, "stream": True, "stream_options": stream_options}
            )
        else:
            stream = await client.responses.create(**{**kwargs, "stream": True})
        return stream

    def _parse(self, raw: Any) -> LLMResponse:
        return parse_chat_completion(raw) if self.api == "chat" else parse_response(raw)

    def _assembler(self, model: str) -> Assembler:
        if self.api == "chat":
            return ChatStreamAssembler()
        return ResponsesStreamAssembler(model=model)

    def _finish(
        self, response: LLMResponse, request: LLMRequest, raw: Any, kwargs: Mapping[str, Any]
    ) -> LLMResponse:
        # Older payloads lack the per-TTL split; the requested TTL then covers every write.
        usage = response.usage
        ttl = get(kwargs.get("prompt_cache_options"), "ttl")
        if usage.cache_write_input_tokens and not usage.cache_write_by_ttl and isinstance(ttl, str):
            by_ttl = {ttl: usage.cache_write_input_tokens}
            response = response.model_copy(
                update={"usage": usage.model_copy(update={"cache_write_by_ttl": by_ttl})}
            )
        return super()._finish(response, request, raw, kwargs)

    def _map_error(self, exc: BaseException, model: str) -> ModelError | None:
        return map_openai_error(exc, model=model)

    def _map_client_error(self, exc: BaseException, model: str) -> ModelError | None:
        # AsyncOpenAI() raises a bare OpenAIError only for missing credentials.
        import openai

        if isinstance(exc, openai.OpenAIError):
            return model_error(ModelAuthError, exc, provider=PROVIDER, model=model)
        return None


def map_openai_error(exc: BaseException, *, model: str) -> ModelError | None:
    """The `ModelError` for an openai SDK exception; None for other exceptions.

    Every `openai.OpenAIError` maps: errors sent inside a stream are typed by their body.
    """
    try:
        import openai
    except ImportError:
        return None
    if not isinstance(exc, openai.OpenAIError):
        return None
    cls = with_overflow(_openai_class(exc, openai), exc, getattr(exc, "body", None))
    return sdk_error(cls, exc, provider=PROVIDER, model=model)


def _openai_class(exc: BaseException, openai: Any) -> type[ModelError]:
    table: tuple[tuple[Any, type[ModelError]], ...] = (
        (openai.APITimeoutError, ModelTimeout),
        (openai.APIConnectionError, ModelUnavailable),
        (openai.RateLimitError, ModelRateLimited),
        ((openai.AuthenticationError, openai.PermissionDeniedError), ModelAuthError),
        (
            (
                openai.BadRequestError,
                openai.NotFoundError,
                openai.ConflictError,
                openai.UnprocessableEntityError,
                openai.APIResponseValidationError,
            ),
            ModelBadRequest,
        ),
        (openai.InternalServerError, ModelUnavailable),
        (openai.ContentFilterFinishReasonError, ModelContentFiltered),
        (openai.LengthFinishReasonError, ModelOutputInvalid),
    )
    for sdk_cls, ours in table:
        if isinstance(exc, sdk_cls):
            return ours
    if isinstance(exc, openai.APIStatusError):
        return class_for_status(exc.status_code)
    if isinstance(exc, openai.APIError):  # raised mid-stream; the body is the error object
        return class_for_error_type(error_type(exc.body))
    return ModelError
