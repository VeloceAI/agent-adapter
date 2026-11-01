"""Usage normalisers: provider usage payloads to one `Usage`.

Each function takes an SDK usage object, a full response, or a plain dict; missing fields
count as 0. OpenAI's prompt total includes cached reads *and* cache writes, so
`input_tokens` is `total - cached - cache_write`; writes appear only in
`cache_write_input_tokens` (see Usage on the Model ports docs page).
"""

from __future__ import annotations

from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.types import Usage
from agent_adapter.models._convert import count, get

# OpenAI `cache_write_tokens_details` keys -> Usage.cache_write_by_ttl keys.
_OPENAI_TTL_KEYS = {"ttl_30m_tokens": "30m", "ttl_12h_tokens": "12h"}
_ANTHROPIC_TTL_KEYS = {"ephemeral_5m_input_tokens": "5m", "ephemeral_1h_input_tokens": "1h"}


def _inner(raw: Any, *names: str) -> Any:
    """The usage block of a full response, or `raw` itself when it is the usage block."""
    found = get(raw, *names)
    return raw if found is None else found


def _by_ttl(details: Any, keys: dict[str, str]) -> dict[str, int]:
    if details is None:
        return {}
    return {ttl: count(details, key) for key, ttl in keys.items() if get(details, key) is not None}


def _openai(usage: Any, total: str, inputs: str, output: str, outputs: str) -> Usage:
    details = get(usage, inputs)
    cached = count(details, "cached_tokens")
    written = count(details, "cache_write_tokens")
    return Usage(
        input_tokens=max(0, count(usage, total) - cached - written),
        cached_input_tokens=cached,
        cache_write_input_tokens=written,
        cache_write_by_ttl=_by_ttl(get(details, "cache_write_tokens_details"), _OPENAI_TTL_KEYS),
        output_tokens=count(usage, output),
        reasoning_tokens=count(get(usage, outputs), "reasoning_tokens"),
    )


def usage_openai_chat(raw: Any) -> Usage:
    """Usage of a Chat Completions response (`CompletionUsage`)."""
    usage = _inner(raw, "usage")
    return _openai(
        usage,
        "prompt_tokens",
        "prompt_tokens_details",
        "completion_tokens",
        "completion_tokens_details",
    )


def usage_openai_responses(raw: Any) -> Usage:
    """Usage of a Responses API response (`ResponseUsage`)."""
    usage = _inner(raw, "usage")
    return _openai(
        usage, "input_tokens", "input_tokens_details", "output_tokens", "output_tokens_details"
    )


def usage_gemini(raw: Any) -> Usage:
    """Usage of a Gemini `GenerateContentResponse` (snake_case or camelCase)."""
    usage = _inner(raw, "usage_metadata", "usageMetadata")
    cached = count(usage, "cached_content_token_count", "cachedContentTokenCount")
    thoughts = count(usage, "thoughts_token_count", "thoughtsTokenCount")
    prompt = count(usage, "prompt_token_count", "promptTokenCount")
    # Tool-execution results fed back to the model are billed as input but are not part of
    # prompt_token_count (total = prompt + candidates + tool_use_prompt + thoughts).
    tool_use = count(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount")
    return Usage(
        input_tokens=max(0, prompt - cached) + tool_use,
        cached_input_tokens=cached,
        output_tokens=count(usage, "candidates_token_count", "candidatesTokenCount") + thoughts,
        reasoning_tokens=thoughts,
    )


def usage_anthropic(raw: Any) -> Usage:
    """Usage of an Anthropic `Message`; `input_tokens` already excludes cache reads and writes."""
    usage = _inner(raw, "usage")
    return Usage(
        input_tokens=count(usage, "input_tokens"),
        cached_input_tokens=count(usage, "cache_read_input_tokens"),
        cache_write_input_tokens=count(usage, "cache_creation_input_tokens"),
        cache_write_by_ttl=_by_ttl(get(usage, "cache_creation"), _ANTHROPIC_TTL_KEYS),
        output_tokens=count(usage, "output_tokens"),
    )


def usage_voyage(raw: Any) -> Usage:
    """Usage of a Voyage embed/rerank result; `units` is millions of tokens (price unit)."""
    tokens = count(_inner(raw, "usage"), "total_tokens")
    return Usage(input_tokens=tokens, units=tokens / 1e6)


def normalise_usage(provider: str, raw: Any, *, api: str = "") -> Usage:
    """Dispatch to the normaliser of `provider` (and OpenAI `api`: "chat" or "responses").

    Raises:
        ConfigError: no normaliser for this provider or API.
    """
    if provider == "openai":
        api = api or _openai_api(raw)
        if api == "chat":
            return usage_openai_chat(raw)
        if api == "responses":
            return usage_openai_responses(raw)
        raise ConfigError(f"no OpenAI usage normaliser for api {api!r}; use 'chat' or 'responses'")
    normaliser = _NORMALISERS.get(provider)
    if normaliser is None:
        raise ConfigError(f"no usage normaliser for provider {provider!r}")
    return normaliser(raw)


def _openai_api(raw: Any) -> str:
    """Guess the OpenAI API from the payload's shape."""
    if get(raw, "object") == "chat.completion" or get(raw, "choices") is not None:
        return "chat"
    usage = _inner(raw, "usage")
    return "chat" if get(usage, "prompt_tokens") is not None else "responses"


_NORMALISERS = {"gemini": usage_gemini, "anthropic": usage_anthropic, "voyage": usage_voyage}
