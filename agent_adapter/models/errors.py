"""Shared pieces of the provider error maps; each port maps its SDK classes.

The helpers are public on purpose: hosts writing their own `LLMPort` (or a `CallableModel`
around their own client) use them to raise the same typed errors as the built-in ports.
"""

from __future__ import annotations

import email.utils
import logging
import time
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.errors import (
    ModelAuthError,
    ModelBadRequest,
    ModelContentFiltered,
    ModelContextOverflow,
    ModelError,
    ModelOutputInvalid,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ModelAuthError",
    "ModelBadRequest",
    "ModelContentFiltered",
    "ModelContextOverflow",
    "ModelError",
    "ModelOutputInvalid",
    "ModelRateLimited",
    "ModelTimeout",
    "ModelUnavailable",
    "class_for_error_type",
    "class_for_status",
    "error_type",
    "is_context_overflow",
    "model_error",
    "retry_after",
    "with_overflow",
]

MAX_MESSAGE = 500  # provider error bodies can be long; keep messages log-sized


def class_for_status(status: int | None) -> type[ModelError]:
    """The `ModelError` subclass for an HTTP status the SDK has no dedicated class for."""
    if status is None:
        return ModelError
    if status == 429:
        return ModelRateLimited
    if status in (401, 403):
        return ModelAuthError
    if status == 408:
        return ModelTimeout
    if status >= 500:
        return ModelUnavailable
    if status >= 400:
        return ModelBadRequest
    return ModelError


def error_type(body: Any) -> str:
    """The error `type` and `code` of an error body (`{"error": {...}}` or flat), space-joined.

    OpenAI puts the class in either field (`type: "tokens", code: "rate_limit_exceeded"`).
    """
    if not isinstance(body, Mapping):
        return ""
    inner = body.get("error")
    if isinstance(inner, Mapping):
        body = inner
    return " ".join(str(body[key]) for key in ("type", "code") if body.get(key))


# Lower-case markers of a context-window overflow in provider error codes and messages:
# OpenAI `context_length_exceeded`, Anthropic "prompt is too long" and "input length and
# `max_tokens` exceed context limit", Gemini "input token count".
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "exceed context limit",
    "maximum context length",
    "context window",
    "prompt is too long",
    "input token count",
    "exceeds the maximum number of tokens",
    "too many tokens",
)


def is_context_overflow(text: str) -> bool:
    """True when a provider error code or message says the input exceeds the context window."""
    lowered = text.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


def with_overflow(cls: type[ModelError], exc: BaseException, body: Any = None) -> type[ModelError]:
    """`ModelContextOverflow` for a bad request whose code or message names an overflow."""
    if not issubclass(cls, ModelBadRequest) or cls is ModelContextOverflow:
        return cls
    if is_context_overflow(f"{error_type(body)} {exc}"):
        return ModelContextOverflow
    return cls


def class_for_error_type(code: str) -> type[ModelError]:
    """The `ModelError` subclass for a provider error type, e.g. one sent mid-stream."""
    if "context_length_exceeded" in code:
        return ModelContextOverflow
    if "rate_limit" in code:
        return ModelRateLimited
    if "overloaded" in code or "server_error" in code or "api_error" in code.split():
        return ModelUnavailable
    if "timeout" in code:
        return ModelTimeout
    if "authentication" in code or "permission" in code:
        return ModelAuthError
    if "invalid_request" in code or "not_found" in code or "too_large" in code:
        return ModelBadRequest
    return ModelError


def retry_after(headers: Any, *, now: float | None = None) -> float | None:
    """Seconds to wait from `retry-after-ms` or `Retry-After` (seconds or HTTP date)."""
    if not isinstance(headers, Mapping) and not hasattr(headers, "get"):
        return None
    millis = headers.get("retry-after-ms")
    if millis is not None:
        try:
            return max(0.0, float(millis) / 1000)
        except ValueError:
            logger.debug("ignoring malformed retry-after-ms header %r", millis)
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        logger.debug("ignoring malformed retry-after header %r", value)
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def model_error(
    cls: type[ModelError],
    exc: BaseException,
    *,
    provider: str,
    model: str,
    status_code: int | None = None,
    headers: Any = None,
    request_id: str | None = "",
) -> ModelError:
    """Build `cls` from an SDK exception, copying status, `Retry-After` and request id."""
    detail = str(exc) or type(exc).__name__
    if len(detail) > MAX_MESSAGE:
        detail = detail[:MAX_MESSAGE] + "..."
    return cls(
        f"{provider} {model}: {detail}",
        provider=provider,
        model=model,
        status_code=status_code,
        retry_after=retry_after(headers) if headers is not None else None,
        request_id=request_id or "",
    )


def sdk_error(
    cls: type[ModelError], exc: BaseException, *, provider: str, model: str
) -> ModelError:
    """`model_error` with the status, headers and request id an SDK `APIError` carries."""
    response = getattr(exc, "response", None)
    return model_error(
        cls,
        exc,
        provider=provider,
        model=model,
        status_code=getattr(exc, "status_code", None),
        headers=getattr(response, "headers", None),
        request_id=getattr(exc, "request_id", None),
    )
