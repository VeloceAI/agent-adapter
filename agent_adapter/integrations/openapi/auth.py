"""Credentials for imported OpenAPI tools, applied to every request (values never logged)."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.integrations.openapi._redact import protect_query_secret

TokenSource = str | Callable[[], str | Awaitable[str]]


class Auth(ABC):
    """Adds credentials to one request's headers or query parameters."""

    @abstractmethod
    async def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        """Add the credentials in place."""


class BearerAuth(Auth):
    """`Authorization: Bearer <token>`; a sync or async `token` callable is asked per call."""

    def __init__(self, token: TokenSource) -> None:
        if not (callable(token) or (isinstance(token, str) and token)):
            raise ConfigError("BearerAuth needs a non-empty token or a callable returning one")
        self._token = token

    async def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        """Set the Authorization header.

        Raises:
            ConfigError: the token callable returned no non-empty string.
        """
        token: Any = self._token
        if callable(token):
            token = token()
            if inspect.isawaitable(token):
                token = await token
        if not isinstance(token, str) or not token:
            raise ConfigError("the BearerAuth token callable returned no token")
        headers["Authorization"] = f"Bearer {token}"

    def __repr__(self) -> str:
        return "BearerAuth(token=***)"


class _NamedValueAuth(Auth):
    def __init__(self, name: str, value: str) -> None:
        if not (isinstance(name, str) and name and isinstance(value, str) and value):
            raise ConfigError(f"{type(self).__name__} needs a non-empty name and value")
        self.name = name
        self._value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, value=***)"


class HeaderAuth(_NamedValueAuth):
    """A fixed header, e.g. `HeaderAuth("X-API-Key", key)`."""

    async def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        """Set the header."""
        headers[self.name] = self._value


class QueryAuth(_NamedValueAuth):
    """A fixed query parameter, e.g. `QueryAuth("api_key", key)`; prefer `HeaderAuth`.

    The value is masked in this process's httpx log lines, but servers, proxies and their
    access logs still record query strings. Use it only when the API accepts nothing else.
    """

    def __init__(self, name: str, value: str) -> None:
        super().__init__(name, value)
        protect_query_secret(name, value)

    async def apply(self, headers: dict[str, str], params: dict[str, Any]) -> None:
        """Set the query parameter."""
        params[self.name] = self._value


__all__ = ["Auth", "BearerAuth", "HeaderAuth", "QueryAuth"]
