import email.utils

import pytest

from agent_adapter.core.errors import (
    ModelAuthError,
    ModelBadRequest,
    ModelError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from agent_adapter.models.errors import (
    MAX_MESSAGE,
    class_for_error_type,
    class_for_status,
    error_type,
    model_error,
    retry_after,
)


@pytest.mark.parametrize(
    ("status", "cls"),
    [
        (None, ModelError),
        (200, ModelError),
        (400, ModelBadRequest),
        (401, ModelAuthError),
        (403, ModelAuthError),
        (404, ModelBadRequest),
        (408, ModelTimeout),
        (413, ModelBadRequest),
        (429, ModelRateLimited),
        (500, ModelUnavailable),
        (529, ModelUnavailable),
    ],
)
def test_class_for_status(status, cls):
    assert class_for_status(status) is cls


def test_retry_after_seconds_millis_and_date():
    assert retry_after({"retry-after": "3"}) == 3.0
    assert retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    date = email.utils.formatdate(1_000_010, usegmt=True)
    assert retry_after({"retry-after": date}, now=1_000_000) == pytest.approx(10.0)
    assert retry_after({"retry-after": date}, now=2_000_000) == 0.0


def test_retry_after_missing_or_malformed():
    assert retry_after({}) is None
    assert retry_after(None) is None
    assert retry_after({"retry-after": "soon"}) is None
    assert retry_after({"retry-after-ms": "x"}) is None
    assert retry_after({"retry-after-ms": "x", "retry-after": "2"}) == 2.0


def test_model_error_copies_fields_and_truncates():
    cause = RuntimeError("x" * (MAX_MESSAGE + 10))
    err = model_error(
        ModelRateLimited,
        cause,
        provider="openai",
        model="gpt-4.1",
        status_code=429,
        headers={"retry-after": "2"},
        request_id="req_1",
    )
    assert isinstance(err, ModelRateLimited)
    assert (err.provider, err.model, err.status_code, err.retry_after, err.request_id) == (
        "openai",
        "gpt-4.1",
        429,
        2.0,
        "req_1",
    )
    assert str(err).endswith("...")
    assert len(str(err)) < MAX_MESSAGE + 40


def test_model_error_defaults():
    err = model_error(ModelError, TimeoutError(), provider="gemini", model="m", request_id=None)
    assert str(err) == "gemini m: TimeoutError"
    assert err.retry_after is None and err.request_id == ""


@pytest.mark.parametrize(
    ("code", "cls"),
    [
        ("rate_limit_error", ModelRateLimited),
        ("rate_limit_exceeded", ModelRateLimited),
        ("overloaded_error", ModelUnavailable),
        ("api_error", ModelUnavailable),
        ("server_error", ModelUnavailable),
        ("timeout_error", ModelTimeout),
        ("invalid_request_error", ModelBadRequest),
        ("not_found_error", ModelBadRequest),
        ("request_too_large", ModelBadRequest),
        ("authentication_error", ModelAuthError),
        ("permission_error", ModelAuthError),
        ("mystery", ModelError),
        ("", ModelError),
    ],
)
def test_class_for_error_type(code, cls):
    assert class_for_error_type(code) is cls


def test_error_type_reads_nested_and_flat_bodies():
    assert (
        error_type({"type": "error", "error": {"type": "overloaded_error"}}) == "overloaded_error"
    )
    assert error_type({"type": "server_error", "code": None}) == "server_error"
    assert error_type({"code": "rate_limit_exceeded"}) == "rate_limit_exceeded"
    assert error_type({"error": {"type": "tokens", "code": "rate_limit_exceeded"}}) == (
        "tokens rate_limit_exceeded"
    )
    assert error_type("text body") == ""
    assert error_type(None) == ""
