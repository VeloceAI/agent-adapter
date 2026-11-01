"""Host allowlist for the REST routes (DNS rebinding): `host`, `host:port`, `host:*`, `*`."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]")
LOCAL_BINDS = frozenset({"127.0.0.1", "localhost", "::1"})


class HostRule:
    """Which `Host` headers a request may carry; an entry without a port allows any port."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self.any = False
        self._allowed: set[tuple[str, str | None]] = set()
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                raise ConfigError(f"allowed_hosts entries must be host names, got {pattern!r}")
            if pattern.strip() == "*":
                self.any = True
                continue
            host, port = split_host(pattern.strip().lower())
            self._allowed.add((host, None if port in (None, "*") else port))

    def allows(self, header: str | None) -> bool:
        """True when the `Host` header names an allowed host (and port)."""
        if self.any:
            return True
        if not header:
            return False
        host, port = split_host(header.strip().lower())
        return (host, None) in self._allowed or (port is not None and (host, port) in self._allowed)


def split_host(value: str) -> tuple[str, str | None]:
    """`"[::1]:80"` -> `("[::1]", "80")`; `"a.b"` -> `("a.b", None)`."""
    if value.startswith("["):
        end = value.find("]")
        host, rest = value[: end + 1], value[end + 1 :]
        return host, rest[1:] if rest.startswith(":") else None
    host, sep, port = value.rpartition(":")
    return (host, port) if sep and host else (value, None)


def default_hosts(
    given: Sequence[str] | None,
    mcp_options: Mapping[str, Any],
    *,
    anonymous: bool,
    bind: str | None = None,
) -> list[str]:
    """`given`; else MCP's `transport_security` hosts; else local names when the bind host
    (`bind`, else `mcp_options["host"]`, else 127.0.0.1) is local or the app is anonymous;
    else any host (with a WARNING: set `allowed_hosts`)."""
    if given is not None:
        if isinstance(given, str):
            raise ConfigError("allowed_hosts must be a list of host names, not a string")
        return list(given)
    security = mcp_options.get("transport_security")
    hosts = getattr(security, "allowed_hosts", None)
    if hosts:
        return list(hosts)
    host = bind if bind is not None else mcp_options.get("host", "127.0.0.1")
    if host in LOCAL_BINDS or anonymous:
        return list(LOCAL_NAMES)
    logger.warning(
        "bound to %r with no allowed_hosts: the routes accept any Host header; set "
        "allowed_hosts= when the authenticator trusts client IPs or networks",
        host,
    )
    return ["*"]


class HostPolicy:
    """The app's `HostRule`, re-derived when the real bind host is known (`serve`).

    Raises:
        ConfigError: an anonymous app on a non-local bind host without `anonymous_public`.
    """

    def __init__(
        self,
        given: Sequence[str] | None,
        mcp_options: Mapping[str, Any],
        *,
        anonymous: bool,
        bind: str | None = None,
        anonymous_public: bool = False,
    ) -> None:
        self._given = given
        self._mcp_options = dict(mcp_options)
        self._anonymous = anonymous
        self._anonymous_public = anonymous_public
        self._check(bind if bind is not None else self._mcp_options.get("host", "127.0.0.1"))
        self.rule = HostRule(default_hosts(given, mcp_options, anonymous=anonymous, bind=bind))

    def bind(self, host: str) -> None:
        """Use `host` as the bind host (explicit `allowed_hosts` are kept as given).

        Raises:
            ConfigError: an anonymous app on a non-local `host` without `anonymous_public`.
        """
        self._check(host)
        if self._given is None:
            hosts = default_hosts(None, self._mcp_options, anonymous=self._anonymous, bind=host)
            self.rule = HostRule(hosts)

    def allows(self, header: str | None) -> bool:
        return self.rule.allows(header)

    @property
    def local_only(self) -> bool:
        """Anonymous without `anonymous_public`: only loopback peers may be served."""
        return self._anonymous and not self._anonymous_public

    def _check(self, host: Any) -> None:
        # The Host allowlist only stops browsers (DNS rebinding): any remote client can send
        # `Host: localhost`, so an anonymous app must not listen beyond this machine by default.
        if self._anonymous and not self._anonymous_public and host not in LOCAL_BINDS:
            logger.error("refused: anonymous access on non-local bind host %r", host)
            raise ConfigError(
                f"anonymous access on non-local host {host!r} needs allow_anonymous_public=True "
                "(anyone who reaches the port can call every tool)"
            )
