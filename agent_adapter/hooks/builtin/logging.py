"""LoggingHook: one structured log line per tool, LLM and node start, finish and error."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_adapter.core.messages import LLMResponse
from agent_adapter.core.types import ToolResult
from agent_adapter.hooks.base import Hook, LLMInvocation, NodeInvocation, ToolInvocation

DEFAULT_REDACT: tuple[str, ...] = ("key", "token", "secret", "password", "authorization", "cookie")
REDACTED = "***"


_NAME_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# A name holding any of these words names a secret.
_SECRET_WORDS = frozenset(
    {"password", "passwd", "pwd", "passphrase", "secret", "credential", "credentials", "apikey"}
)
# A name ending in one of these nouns names a secret (`access_token`, not `max_tokens`).
_SECRET_NOUNS = frozenset({"token", "cookie", "authorization", "auth", "bearer", "signature"})
# `<qualifier>_key` names a secret (`api_key`, `private_key_pem`, not `cache_key`).
_KEY_QUALIFIERS = frozenset(
    {"api", "access", "secret", "private", "client", "auth", "signing", "encryption", "master"}
)
# Plural `tokens` is a count (`max_tokens`) unless one of these says which token it is.
_TOKEN_QUALIFIERS = frozenset(
    {"access", "refresh", "id", "auth", "api", "bearer", "session", "oauth", "csrf", "xsrf"}
    | {"jwt", "client", "device", "service", "personal", "github", "slack", "secret"}
)
_VOCABULARY = _SECRET_WORDS | _SECRET_NOUNS | {"key"}
# Bare names that hold a secret only when their value looks like one: `{"key": "abc..."}`.
_BARE_NAMES = (("key",), ("token",))
_SECRET_VALUE_CHARS = 8  # shortest unbroken string a bare `key` value is masked at


def _singular(word: str) -> str:
    for suffix in ("es", "s"):
        if word.endswith(suffix) and word[: -len(suffix)] in _VOCABULARY:
            return word[: -len(suffix)]
    return word


def is_secret_name(name: object) -> bool:
    """True for a key name that holds a secret, by its words: `api_keys`, not `max_tokens`."""
    words = [word.lower() for word in _NAME_WORDS.findall(str(name))]
    if not words:
        return False
    singular = [_singular(word) for word in words]
    if any(word in _SECRET_WORDS for word in singular):
        return True
    last = singular[-1]
    if last == "token" and words[-1] != "token":
        return len(words) > 1 and words[-2] in _TOKEN_QUALIFIERS  # `access_tokens`
    if last in _SECRET_NOUNS:
        return True
    return any(
        word == "key" and singular[i - 1] in _KEY_QUALIFIERS
        for i, word in enumerate(singular)
        if i > 0
    )


def is_secret_field(name: object, value: Any) -> bool:
    """True for a secret-named entry, or a bare `key`/`tokens` entry holding a secret-like str."""
    if is_secret_name(name):
        return True
    words = tuple(_singular(word.lower()) for word in _NAME_WORDS.findall(str(name)))
    return words in _BARE_NAMES and _secret_like(value)


def _secret_like(value: Any) -> bool:
    if isinstance(value, str):
        return len(value) >= _SECRET_VALUE_CHARS and not any(c.isspace() for c in value)
    if isinstance(value, list | tuple) and value:
        return all(_secret_like(item) for item in value)
    return False


def redact_secrets(value: Any) -> Any:
    """A copy of `value` whose secret entries (`is_secret_field`) are `***`."""
    return _redact(value, (), _never, is_secret_field)


def _never(_: str) -> bool:
    return False


def redact(
    value: Any,
    patterns: Iterable[str] = DEFAULT_REDACT,
    *,
    match: Callable[[str], bool] | None = None,
) -> Any:
    """A copy of `value` whose secret mapping keys are `***`.

    By default a key is secret when it contains any pattern (case-insensitive); `match`,
    e.g. `is_secret_name`, decides instead.
    """
    lowered = tuple(p.lower() for p in patterns)
    return _redact(value, lowered, match)


def _redact(
    value: Any,
    patterns: tuple[str, ...],
    match: Callable[[str], bool] | None = None,
    also: Callable[[str, Any], bool] | None = None,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: REDACTED
            if _hidden(key, item, patterns, match, also)
            else _redact(item, patterns, match, also)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact(item, patterns, match, also) for item in value]
    return value


def _hidden(
    key: Any,
    item: Any,
    patterns: tuple[str, ...],
    match: Callable[[str], bool] | None,
    also: Callable[[str, Any], bool] | None,
) -> bool:
    if also is not None and also(str(key), item):
        return True
    return match(str(key)) if match is not None else _matches(key, patterns)


def _matches(key: Any, patterns: tuple[str, ...]) -> bool:
    name = str(key).lower()
    return any(pattern in name for pattern in patterns)


class LoggingHook(Hook):
    """Logs `<subject>.start|finish|error` lines; errors log at WARNING or above.

    Each record carries `aa_event` and `aa_fields` attributes for structured handlers, and
    a `"<event> <json>"` message. Arguments are redacted; those of `confirm` tools, message
    content and result values are never logged.
    """

    name = "logging"

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        redact: Iterable[str] = DEFAULT_REDACT,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.level = level
        self.redact = tuple(pattern.lower() for pattern in redact)

    async def before_tool(self, inv: ToolInvocation) -> None:
        if not self.logger.isEnabledFor(self.level):
            return
        fields = self._tool_fields(inv)
        fields["kind"] = inv.spec.kind
        if inv.spec.confirm is False:
            # Your patterns, plus every name `is_secret_field` knows (`passwd`, `api_keys`).
            fields["args"] = _redact(inv.call.arguments, self.redact, also=is_secret_field)
        else:
            fields["args_withheld"] = True
        self._log(self.level, "tool.start", fields)

    async def after_tool(self, inv: ToolInvocation, result: ToolResult) -> None:
        fields = self._tool_fields(inv)
        fields.update(
            status=result.status,
            attempts=result.attempts,
            cached=result.cached,
            latency_ms=result.latency_ms,
        )
        if result.error is not None:
            fields["error_kind"] = result.error.kind
        self._log(self.level, "tool.finish", fields)

    async def on_tool_error(self, inv: ToolInvocation, error: BaseException) -> None:
        self._log(self._error_level, "tool.error", {**self._tool_fields(inv), **_error(error)})

    async def before_llm(self, inv: LLMInvocation) -> None:
        fields = self._llm_fields(inv)
        fields.update(
            messages=len(inv.request.messages),
            tools=len(inv.request.tools),
            stream=inv.request.stream,
        )
        self._log(self.level, "llm.start", fields)

    async def after_llm(self, inv: LLMInvocation, response: LLMResponse) -> None:
        fields = self._llm_fields(inv)
        usage = response.usage
        fields.update(
            response_model=response.model,
            finish_reason=response.finish_reason,
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            tool_calls=len(response.tool_calls),
            latency_ms=response.latency_ms,
            cached=response.cached,
        )
        self._log(self.level, "llm.finish", fields)

    async def on_llm_error(self, inv: LLMInvocation, error: BaseException) -> None:
        self._log(self._error_level, "llm.error", {**self._llm_fields(inv), **_error(error)})

    async def before_node(self, inv: NodeInvocation) -> None:
        self._log(self.level, "node.start", self._node_fields(inv))

    async def after_node(self, inv: NodeInvocation, update: Mapping[str, Any] | None) -> None:
        fields = {**self._node_fields(inv), "updated": sorted(update or {})}
        self._log(self.level, "node.finish", fields)

    async def on_node_error(self, inv: NodeInvocation, error: BaseException) -> None:
        self._log(self._error_level, "node.error", {**self._node_fields(inv), **_error(error)})

    @property
    def _error_level(self) -> int:
        return max(self.level, logging.WARNING)

    @staticmethod
    def _tool_fields(inv: ToolInvocation) -> dict[str, Any]:
        return {
            "tool": inv.spec.name,
            "run_id": inv.ctx.run_id,
            "call_id": inv.call.id or inv.ctx.call_id,
        }

    @staticmethod
    def _llm_fields(inv: LLMInvocation) -> dict[str, Any]:
        return {
            "provider": inv.provider,
            "model": inv.model,
            "stage": inv.request.stage,
            "run_id": inv.ctx.run_id,
        }

    @staticmethod
    def _node_fields(inv: NodeInvocation) -> dict[str, Any]:
        return {"flow": inv.flow, "node": inv.node, "run_id": inv.ctx.run_id}

    def _log(self, level: int, event: str, fields: dict[str, Any]) -> None:
        if not self.logger.isEnabledFor(level):
            return
        body = json.dumps(fields, default=str)
        self.logger.log(level, "%s %s", event, body, extra={"aa_event": event, "aa_fields": fields})


def _error(error: BaseException) -> dict[str, Any]:
    # The message is left out: exception text can echo argument values.
    cls = type(error)
    return {
        "error_kind": getattr(error, "kind", "") or cls.__name__,
        "error_type": f"{cls.__module__}.{cls.__qualname__}",
    }
