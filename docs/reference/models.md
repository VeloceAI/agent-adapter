# Models

Model ports, provider models and usage helpers. Concepts: [Model ports](../concepts/models.md), [Tool loop](../concepts/tool-loop.md).

## `agent_adapter.models`

::: agent_adapter.models
    options:
      members: [AnthropicModel, CapabilityLookup, DEFAULT_FALL_ON, GeminiModel, InstrumentedModel, LLMPort, ModelCaps, OpenAIModel, OutputSchema, PromptCachePlugin, normalise_usage, usage_anthropic, usage_gemini, usage_openai_chat, usage_openai_responses, usage_voyage]

Also exported here, documented under the canonical path: [`CallableModel`](top-level.md#agent_adapter.CallableModel), [`FallbackModel`](top-level.md#agent_adapter.FallbackModel).

## `agent_adapter.models.openai`

::: agent_adapter.models.openai
    options:
      members: [build_chat_request, build_responses_request, map_openai_error, parse_chat_completion, parse_response]

Also exported here, documented under the canonical path: [`OpenAIModel`](#agent_adapter.models.OpenAIModel).

## `agent_adapter.models.gemini`

::: agent_adapter.models.gemini
    options:
      members: [build_gemini_request, map_gemini_error, parse_gemini_response]

Also exported here, documented under the canonical path: [`GeminiModel`](#agent_adapter.models.GeminiModel).

## `agent_adapter.models.anthropic`

::: agent_adapter.models.anthropic
    options:
      members: [build_anthropic_request, map_anthropic_error, parse_anthropic_message]

Also exported here, documented under the canonical path: [`AnthropicModel`](#agent_adapter.models.AnthropicModel).

## `agent_adapter.models.errors`

::: agent_adapter.models.errors
    options:
      members: [class_for_error_type, class_for_status, error_type, is_context_overflow, model_error, retry_after, with_overflow]

Also exported here, documented under the canonical path: [`ModelAuthError`](errors.md#agent_adapter.errors.ModelAuthError), [`ModelBadRequest`](errors.md#agent_adapter.errors.ModelBadRequest), [`ModelContentFiltered`](errors.md#agent_adapter.errors.ModelContentFiltered), [`ModelContextOverflow`](errors.md#agent_adapter.errors.ModelContextOverflow), [`ModelError`](errors.md#agent_adapter.errors.ModelError), [`ModelOutputInvalid`](errors.md#agent_adapter.errors.ModelOutputInvalid), [`ModelRateLimited`](errors.md#agent_adapter.errors.ModelRateLimited), [`ModelTimeout`](errors.md#agent_adapter.errors.ModelTimeout), [`ModelUnavailable`](errors.md#agent_adapter.errors.ModelUnavailable).
