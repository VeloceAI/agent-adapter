"""Client-side OAuth for `mcp_client`: the SDK provider with Client ID Metadata Documents."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.mcp._handlers import require_mcp

if TYPE_CHECKING:
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthToken

__all__ = ["MemoryTokenStorage", "client_metadata_document", "oauth_client"]

logger = logging.getLogger(__name__)

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class MemoryTokenStorage:
    """Tokens and client registration for one server, in this process only; never logged."""

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client = client_info

    def __repr__(self) -> str:
        return f"MemoryTokenStorage(tokens={'set' if self._tokens else 'none'})"


def client_metadata_document(
    client_metadata_url: str,
    *,
    redirect_uris: Sequence[str],
    client_name: str = "agent-adapter",
    scope: str | None = None,
) -> dict[str, Any]:
    """The JSON to publish at `client_metadata_url`; its `client_id` is that URL.

    Raises:
        ConfigError: the URL is not https with a path, or a redirect URI is missing or unsafe.
    """
    _check_metadata_url(client_metadata_url)
    if not redirect_uris or isinstance(redirect_uris, str):
        raise ConfigError("client_metadata_document needs at least one redirect URI")
    for uri in redirect_uris:
        _check_redirect_uri(uri)
    document: dict[str, Any] = {
        "client_id": client_metadata_url,
        "client_name": client_name,
        "redirect_uris": list(redirect_uris),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if scope:
        document["scope"] = scope
    return document


def oauth_client(
    server_url: str,
    *,
    client_metadata_url: str,
    redirect_uri: str,
    redirect_handler: Callable[[str], Awaitable[None]],
    callback_handler: Callable[[], Awaitable[AuthorizationCodeResult]],
    storage: Any = None,
    client_name: str = "agent-adapter",
    scope: str | None = None,
) -> OAuthClientProvider:
    """An `httpx2.Auth` for `mcp_client(server_url, auth=...)`: authorization code + PKCE.

    The SDK discovers the protected-resource and authorization-server metadata, uses the
    Client ID Metadata Document at `client_metadata_url` when the server supports it (else
    dynamic registration), refreshes tokens, and binds stored credentials to their issuer.

    Args:
        redirect_handler: Sends the person to the authorization URL.
        callback_handler: Waits for the redirect and returns its code and state.
        storage: A `TokenStorage`; default `MemoryTokenStorage()` (per process).

    Raises:
        ConfigError: `server_url` not https (http only on a loopback host), or a bad
            `client_metadata_url` / `redirect_uri`.
        MissingDependency: `mcp` is not installed.
    """
    require_mcp()
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    parts = urlsplit(server_url)
    if not (parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOOPBACK)):
        raise ConfigError("oauth_client needs an https server_url (http only on a loopback host)")
    document = client_metadata_document(
        client_metadata_url, redirect_uris=[redirect_uri], client_name=client_name, scope=scope
    )
    try:
        metadata = OAuthClientMetadata.model_validate(
            {key: value for key, value in document.items() if key != "client_id"}
        )
        provider = OAuthClientProvider(
            server_url,
            client_metadata=metadata,
            storage=storage if storage is not None else MemoryTokenStorage(),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            client_metadata_url=client_metadata_url,
        )
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        logger.error("oauth_client: unusable client settings: %s", type(exc).__name__)
        raise ConfigError(f"oauth_client: {exc}") from exc
    logger.info("OAuth client for host %s uses its client id document", parts.hostname)
    return provider


def _check_metadata_url(url: object) -> None:
    if not isinstance(url, str):
        raise ConfigError("client_metadata_url must be a str")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.path in ("", "/"):
        raise ConfigError("client_metadata_url must be an https URL with a path")


def _check_redirect_uri(uri: object) -> None:
    """RFC 8252: https, http on a loopback host, or a reverse-domain private-use scheme."""
    if not isinstance(uri, str) or not uri:
        raise ConfigError("client_metadata_document needs at least one redirect URI")
    try:
        parts = urlsplit(uri)
        host = parts.hostname
    except ValueError as exc:
        raise ConfigError("redirect_uri is not a valid URL") from exc
    if parts.fragment or "#" in uri or parts.username is not None or "@" in parts.netloc:
        raise ConfigError("redirect_uri must not carry a fragment or user info")
    if parts.scheme == "https" and host:
        return
    if parts.scheme == "http" and host in _LOOPBACK:
        return
    if "." in parts.scheme and not parts.netloc and parts.path.startswith("/"):
        return  # a private-use scheme such as com.example.app:/callback
    raise ConfigError(
        "redirect_uri must be https, http on a loopback host, or a reverse-domain app scheme"
    )
