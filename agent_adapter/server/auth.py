"""Who is calling: principals, the framework-neutral request view and bearer-token auth."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_adapter.approval.store import ApprovalRecord
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.tool import ToolSpec

logger = logging.getLogger(__name__)

APPROVER_SCOPE = "approvals:decide"
_BEARER = "bearer"
_log_key = secrets.token_bytes(32)  # per process unless `set_log_key` shares one


@dataclass(frozen=True)
class Principal:
    """An authenticated caller; `id` scopes every run, call and approval it touches."""

    id: str
    scopes: frozenset[str] = frozenset()
    claims: Mapping[str, Any] = field(default_factory=dict, repr=False)  # may hold PII


@dataclass(frozen=True)
class RequestInfo:
    """A framework-neutral view of one HTTP request (header names in any case)."""

    method: str
    path: str
    headers: Mapping[str, str]
    query: Mapping[str, str]
    client: str | None

    def header(self, name: str) -> str | None:
        """The value of header `name`, matched case-insensitively."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


Authenticator = Callable[[RequestInfo], Awaitable[Principal | None]]
Authorizer = Callable[[Principal, ToolSpec, Mapping[str, Any] | None], Awaitable[bool]]
DecisionAuthorizer = Callable[[Principal, ApprovalRecord], Awaitable[bool]]
TokenVerifier = Callable[[str], Awaitable[Principal | None]]


def bearer_auth(tokens: Mapping[str, Principal] | TokenVerifier) -> Authenticator:
    """An authenticator for `Authorization: Bearer <token>`.

    Args:
        tokens: A fixed `{token: Principal}` table (looked up by SHA-256, never by the raw
            token), or an async `verify(token) -> Principal | None`.

    Raises:
        ToolUsageError: `tokens` is neither, or the table holds an empty token or a value
            that is not a `Principal`.
    """
    if isinstance(tokens, Mapping):
        table = _token_table(tokens)

        async def lookup(token: str) -> Principal | None:
            return table.get(_sha256(token))

        verify: TokenVerifier = lookup
    elif callable(tokens):
        verify = tokens
    else:
        raise ToolUsageError(f"bearer_auth needs a mapping or a callable, got {tokens!r}")

    async def authenticate(request: RequestInfo) -> Principal | None:
        token = bearer_token(request)
        if not token:
            return None
        principal = await verify(token)
        if principal is not None and not isinstance(principal, Principal):
            raise ToolUsageError(
                f"token verifier returned {type(principal).__name__}, not Principal or None"
            )
        return principal

    return authenticate


def bearer_token(request: RequestInfo) -> str:
    """The token of an `Authorization: Bearer <token>` header; "" when absent or malformed."""
    value = request.header("authorization") or ""
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() != _BEARER:
        return ""
    return token.strip()


async def allow_authenticated(
    principal: Principal, spec: ToolSpec, arguments: Mapping[str, Any] | None
) -> bool:
    """Default `authorize`: every authenticated principal may list and call every tool."""
    return True


async def owner_or_approver(principal: Principal, record: ApprovalRecord) -> bool:
    """Default `authorize_decision`: the invoker of the call, or a holder of `APPROVER_SCOPE`."""
    return APPROVER_SCOPE in principal.scopes or is_invoker(principal, record)


def is_invoker(principal: Principal, record: ApprovalRecord) -> bool:
    """True when `principal` made the call `record` waits on (from the record's metadata).

    Anonymous invokers are marked `metadata["anonymous"]`; a record naming nobody has no
    invoker, so it matches no one (approvers only).
    """
    metadata = record.request.metadata
    if metadata.get("anonymous") is True:
        return principal.id == ""
    invoker = metadata.get("principal")
    return isinstance(invoker, str) and invoker != "" and invoker == principal.id


def invoker_of(record: ApprovalRecord) -> Principal | None:
    """The invoker rebuilt from metadata (id and scopes; claims are never stored), or None."""
    metadata = record.request.metadata
    if metadata.get("anonymous") is True:
        return Principal("")
    invoker = metadata.get("principal")
    if not isinstance(invoker, str) or not invoker:
        return None
    scopes = metadata.get("principal_scopes", [])
    if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
        logger.warning("approval %s has malformed invoker scopes", record.approval_id)
        return None
    return Principal(invoker, scopes=frozenset(scopes))


def principal_ref(principal_id: str | None) -> str:
    """A short keyed hash (HMAC-SHA256) of a principal id for logs (ids may be e-mails)."""
    if not principal_id:
        return "anonymous"
    mac = hmac.new(_log_key, principal_id.encode("utf-8", "surrogatepass"), hashlib.sha256)
    return mac.hexdigest()[:12]


def set_log_key(key: bytes) -> None:
    """Share the `principal_ref` key between workers so their logs correlate (>= 16 bytes).

    Raises:
        ConfigError: a key shorter than 16 bytes.
    """
    global _log_key
    if len(key) < 16:
        raise ConfigError("the log key needs at least 16 bytes")
    _log_key = bytes(key)


def _token_table(tokens: Mapping[str, Principal]) -> dict[str, Principal]:
    table: dict[str, Principal] = {}
    for token, principal in tokens.items():
        if not isinstance(token, str) or not token.strip():
            raise ToolUsageError("bearer_auth tokens must be non-empty strings")
        if not isinstance(principal, Principal):
            raise ToolUsageError(
                f"bearer_auth maps tokens to Principal, got {type(principal).__name__}"
            )
        table[_sha256(token.strip())] = principal
    return table


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
