"""Secret redaction for text sent to a summariser: secret-named values and secret-shaped ones."""

from __future__ import annotations

import json
import re
from typing import Any

from agent_adapter.core.messages import Message
from agent_adapter.hooks.builtin.logging import REDACTED, is_secret_field, redact_secrets

# Every secret name (`is_secret_field`) holds one of these: a cheap test before the full scan.
_MAYBE_SECRET_NAME = re.compile(r"(?i)key|token|secret|pass|pwd|credential|auth|cookie|bearer|sign")
# `api_key: abc`, `"password": "x y"`, `token=abc`: the value after a secret key name.
# Key names start at a word edge and are bounded, so a long unbroken string is linear time.
_KEY_VALUE = re.compile(
    r"(?<![\w.-])(?P<key>[\w.-]{1,64})(?P<sep>[\"']?[ \t]*[:=][ \t]*)"
    r"(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s,;\"'}\]]+)"
)
# Secrets recognisable by their shape alone, each with a lower-case marker it must contain
# (a cheap substring test skips the pattern on most text).
_SHAPES = (
    ("bearer", re.compile(r"(?i)\b(?P<keep>bearer)\s+[A-Za-z0-9._~+/=-]{6,}")),
    ("basic", re.compile(r"(?i)\b(?P<keep>basic)\s+[A-Za-z0-9._~+/=-]{6,}")),
    ("://", re.compile(r"(?P<keep>://)[^/\s:@]+:[^/\s@]+@")),  # user:password@ in a URL
    (
        "private key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    ("sk-", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),  # OpenAI / Anthropic style API keys
    ("akia", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),  # AWS access key ids
    ("gh", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),  # GitHub tokens
    ("github_pat_", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),  # GitHub fine-grained
    ("glpat-", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}")),  # GitLab tokens
    ("aiza", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),  # Google API keys
    ("k_live_", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,}")),  # Stripe secret keys
    ("hf_", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),  # Hugging Face tokens
    ("xox", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),  # Slack tokens
    ("eyj", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),  # JWTs
)


def redact_text(text: str) -> str:
    """`text` with secret values replaced by `***`; ordinary keys such as max_tokens stay."""
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            value = json.loads(text)
        except ValueError:
            pass  # not JSON after all: the text rules below still apply
        else:
            text = json.dumps(redact_secrets(value), ensure_ascii=False)
    lowered = text.lower()
    for marker, shape in _SHAPES:
        if marker in lowered:
            text = shape.sub(_masked_shape, text)
    if not _MAYBE_SECRET_NAME.search(text):
        return text
    return _KEY_VALUE.sub(_masked_value, text)


def message_view(message: Message) -> dict[str, Any]:
    """A redacted, provider-neutral view of `message` for a summariser; no ids, no raw items."""
    view: dict[str, Any] = {"role": message.role}
    if message.name:
        view["name"] = message.name
    if message.content:
        view["content"] = redact_text(message.content)
    if message.tool_calls:
        view["tool_calls"] = [
            {"name": call.name, "arguments": _redact_value(call.arguments)}
            for call in message.tool_calls
        ]
    return view


def _redact_value(value: Any) -> Any:
    """Secret-named mapping values, then secret-shaped strings."""
    return _strings(redact_secrets(value))


def _strings(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: _strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strings(item) for item in value]
    return value


def _masked_shape(match: re.Match[str]) -> str:
    keep = match.groupdict().get("keep")
    if keep is None:
        return REDACTED
    return f"{keep}{REDACTED}@" if keep == "://" else f"{keep} {REDACTED}"


def _masked_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if not is_secret_field(match.group("key"), value.strip("\"'")):
        return match.group(0)
    if value == REDACTED or value.strip("\"'") == REDACTED:
        return match.group(0)
    quote = value[0] if value[:1] in ("'", '"') else ""
    return f"{match.group('key')}{match.group('sep')}{quote}{REDACTED}{quote}"
