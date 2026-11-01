"""`is_secret_name` and `redact(match=...)`: secret key names only, not words like max_tokens."""

import pytest

from agent_adapter.hooks.builtin.logging import (
    REDACTED,
    is_secret_field,
    is_secret_name,
    redact,
    redact_secrets,
)


@pytest.mark.parametrize(
    "name",
    [
        "api_key",
        "x-api-key",
        "apiKey",
        "APIKey",
        "apikey",
        "access_token",
        "AUTH_TOKEN",
        "refreshToken",
        "id_token",
        "token",
        "password",
        "password_hash",
        "passwd",
        "client_secret",
        "secret_id",
        "Authorization",
        "auth",
        "Cookie",
        "set-cookie",
        "private_key",
        "aws_secret_access_key",
        "credentials",
    ],
)
def test_secret_names(name):
    assert is_secret_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "max_tokens",
        "tokens",
        "token_count",
        "maxOutputTokens",
        "token_type",
        "keyboard",
        "monkey",
        "key",
        "cache_key",
        "primary_key",
        "author",
        "keywords",
        "",
        3,
    ],
)
def test_ordinary_names(name):
    assert not is_secret_name(name)


def test_redact_with_a_matcher_decides_by_it_and_keeps_the_default_unchanged():
    payload = {"max_tokens": 800, "api_key": "k", "nested": [{"password": "p", "rows": 2}]}
    assert redact(payload, match=is_secret_name) == {
        "max_tokens": 800,
        "api_key": REDACTED,
        "nested": [{"password": REDACTED, "rows": 2}],
    }
    assert redact(payload)["max_tokens"] == REDACTED  # the logging default is unchanged


@pytest.mark.parametrize(
    "name",
    [
        "api_keys",
        "passwords",
        "secrets",
        "cookies",
        "credential",
        "signatures",
        "access_tokens",
        "refreshTokens",
        "private_key_pem",
        "api_key_id",
        "client_secrets",
    ],
)
def test_plural_and_qualified_secret_names(name):
    assert is_secret_name(name)


@pytest.mark.parametrize(
    "name", ["max_tokens", "tokens", "output_tokens", "maxOutputTokens", "keys", "cache_keys"]
)
def test_plural_counts_and_plain_keys_stay_ordinary(name):
    assert not is_secret_name(name)


@pytest.mark.parametrize(
    ("name", "value", "secret"),
    [
        ("key", "plainsecret", True),
        ("key", "abc123secretvalue", True),
        ("keys", ["abc123secretvalue", "other-secret-1"], True),
        ("tokens", "ghx-abcdefgh12345", True),
        ("key", "user_id", False),  # short: a sort or lookup key, not a secret
        ("key", "sort by name", False),
        ("key", 42, False),
        ("tokens", 512, False),
        ("max_tokens", "123456789", False),
        ("password", 1, True),
    ],
)
def test_is_secret_field_reads_bare_names_by_their_value(name, value, secret):
    assert is_secret_field(name, value) is secret


def test_redact_secrets_masks_the_review_repro():
    payload = {"api_keys": ["abc123secretvalue"], "passwords": "hunter2", "key": "plainsecret"}
    payload |= {"max_tokens": 800, "tokens": 12, "rows": [{"secrets": "s3", "id": 1}]}
    assert redact_secrets(payload) == {
        "api_keys": REDACTED,
        "passwords": REDACTED,
        "key": REDACTED,
        "max_tokens": 800,
        "tokens": 12,
        "rows": [{"secrets": REDACTED, "id": 1}],
    }
