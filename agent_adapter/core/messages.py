"""Provider-neutral LLM request, response, message and stream types."""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_adapter.core.errors import ModelContentFiltered, ModelOutputInvalid
from agent_adapter.core.types import ToolCall, Usage

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "other"]
M = TypeVar("M", bound=BaseModel)

_FROZEN = ConfigDict(frozen=True)


class ProviderRaw(BaseModel):
    """Opaque provider items echoed back to the same provider (e.g. thought signatures)."""

    model_config = _FROZEN

    provider: str
    api: str = ""  # "chat" | "responses" | ""
    items: list[dict[str, Any]] = []


class Message(BaseModel):
    """One chat message."""

    model_config = _FROZEN

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant only
    tool_call_id: str = ""  # tool only
    name: str = ""  # tool only: tool name (Gemini needs it)
    raw: ProviderRaw | None = None  # ignored by other providers
    cache_breakpoint: bool = False  # hint for Anthropic cache_control / OpenAI breakpoints
    # Context compression never edits or drops a pinned message; never sent to a provider.
    # Dumped only when True, so unpinned messages serialise (and cache-key) as before.
    pinned: bool = Field(default=False, exclude_if=lambda value: not value)


class ToolDef(BaseModel):
    """A tool as model ports receive it."""

    model_config = _FROZEN

    name: str
    description: str
    parameters: dict[str, Any]  # neutral JSON schema (ToolSpec.json_schema())
    strict: bool = True  # ports render strict only when True


class OutputSchema(BaseModel):
    """Structured-output schema for a request; ports apply provider transforms."""

    model_config = _FROZEN

    name: str
    json_schema: dict[str, Any]
    strict: bool = True


class LLMRequest(BaseModel):
    """One provider-neutral model call."""

    model_config = _FROZEN

    messages: tuple[Message, ...]
    model: str | None = None  # None: the port's configured model
    tools: tuple[ToolDef, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | str = "auto"  # str: one tool name
    parallel_tool_calls: bool | None = None
    output_schema: OutputSchema | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    reasoning: Literal["minimal", "low", "medium", "high"] | None = None
    stream: bool = False
    stage: str = ""  # ledger stage label
    response_cache: bool = True  # allow the LLM response cache
    extra: dict[str, Any] = {}  # provider kwargs merged last; part of the cache key
    metadata: dict[str, str] = {}  # not sent, not in the cache key; "call_id" sets the entry id
    # A context.CompressionPolicy for this call; never sent, dumped or part of the cache key.
    compression: Any = Field(default=None, exclude=True, repr=False)

    @field_validator("compression")
    @classmethod
    def _policy_like(cls, value: Any) -> Any:
        # Duck-typed: core cannot import agent_adapter.context, which checks the type itself.
        if value is None or (hasattr(value, "limits") and hasattr(value, "resolved_steps")):
            return value
        raise ValueError("compression must be an agent_adapter.context.CompressionPolicy")


class LLMResponse(BaseModel):
    """A model reply, normalised across providers."""

    model_config = _FROZEN

    provider: str
    model: str  # response model id
    id: str = ""
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    parsed: Any = None  # JSON value when output_schema was set
    finish_reason: FinishReason = "stop"
    raw_finish_reason: str = ""
    usage: Usage = Usage()
    message: Message  # assistant message to append to history
    latency_ms: float = 0.0
    cached: bool = False  # served from the LLM response cache

    def parse(self, model: type[M]) -> M:
        """Validate the structured output (`parsed`, else `text` as JSON) into `model`.

        Raises:
            ModelContentFiltered: the provider withheld the output.
            ModelOutputInvalid: the output does not validate against `model`.
        """
        if self.finish_reason == "content_filter":
            raise ModelContentFiltered(
                "the model output was filtered",
                provider=self.provider,
                model=self.model,
                request_id=self.id,
            )
        try:
            if self.parsed is not None:
                return model.model_validate(self.parsed)
            return model.model_validate_json(self.text)
        except ValidationError as exc:
            raise ModelOutputInvalid(
                f"output does not match {model.__name__}: {exc.error_count()} error(s)",
                provider=self.provider,
                model=self.model,
                request_id=self.id,
            ) from exc


class TextDelta(BaseModel):
    """A chunk of streamed text."""

    model_config = _FROZEN

    type: Literal["text"] = "text"
    text: str


class ToolCallDelta(BaseModel):
    """A chunk of a streamed tool call."""

    model_config = _FROZEN

    type: Literal["tool_call"] = "tool_call"
    index: int
    id: str = ""
    name: str = ""
    arguments_delta: str = ""


class StreamDone(BaseModel):
    """The last stream item: the assembled response."""

    model_config = _FROZEN

    type: Literal["done"] = "done"
    response: LLMResponse


StreamEvent = TextDelta | ToolCallDelta | StreamDone
