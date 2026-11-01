"""Errors from a model provider call: rate limits, outages, auth, bad requests, bad output."""

# Each class names `agent_adapter.core.errors`, its public home, so log lines and pickles keep
# the same qualified name.

from __future__ import annotations

from agent_adapter.core.error_base import AdapterError


class ModelError(AdapterError):
    """A model provider call failed."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-error"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str = "",
        status_code: int | None = None,
        retry_after: float | None = None,
        request_id: str = "",
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id
        super().__init__(message)


class ModelRateLimited(ModelError):
    """The provider rate-limited the call."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-rate-limited"
    retryable = True


class ModelUnavailable(ModelError):
    """5xx, overload or a connection error."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-unavailable"
    retryable = True


class ModelTimeout(ModelError):
    """The provider call timed out."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-timeout"
    retryable = True


class ModelAuthError(ModelError):
    """The provider rejected the credentials."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-auth-failed"


class ModelBadRequest(ModelError):
    """The provider rejected the request as invalid."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-bad-request"


class ModelContextOverflow(ModelBadRequest):
    """The request has more tokens than the model's context window."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-context-overflow"


class ModelContentFiltered(ModelError):
    """The provider withheld the output for safety reasons."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-content-filtered"


class ModelOutputInvalid(ModelError):
    """The model output does not match the requested schema."""

    __module__ = "agent_adapter.core.errors"

    kind = "model-output-invalid"
