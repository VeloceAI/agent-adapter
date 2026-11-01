"""`python -m agent_adapter.server --registry mypkg.tools:registry`: serve a registry over HTTP.

Bearer tokens come from the `AGENT_ADAPTER_TOKENS` environment variable, a JSON object
`{"<token>": {"id": "<principal>", "scopes": ["approvals:decide"]}}`; without it the
server refuses to start unless `--allow-anonymous` is given (on a non-local `--host` only
with `--allow-anonymous-public`). `--allowed-host` (repeatable) lists the `Host` names the
routes accept.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import AdapterError, ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.server._hosts import LOCAL_BINDS
from agent_adapter.server.asgi import create_app, serve
from agent_adapter.server.auth import Principal, bearer_auth

TOKENS_ENV = "AGENT_ADAPTER_TOKENS"


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Parse arguments, build the app and serve it; returns the process exit code."""
    args = _parser().parse_args(argv)
    env = os.environ if environ is None else environ
    anonymous = args.allow_anonymous or args.allow_anonymous_public
    if anonymous and args.host not in LOCAL_BINDS and not args.allow_anonymous_public:
        sys.stderr.write(
            "agent_adapter.server: anonymous access on a non-local host needs "
            "--allow-anonymous-public (anyone who reaches the port can call every tool)\n"
        )
        return 2
    try:
        registry = load_registry(args.registry)
        options: dict[str, Any] = {
            "allow_anonymous": anonymous,
            "allow_anonymous_public": args.allow_anonymous_public,
        }
        tokens = tokens_from(env.get(TOKENS_ENV, ""))
        if tokens:
            options["authenticate"] = bearer_auth(tokens)
        options["host"] = args.host  # the bind host decides the default Host allowlist
        if not args.no_mcp:
            options["mcp_options"] = {"host": args.host}
        if args.allowed_host:
            options["allowed_hosts"] = args.allowed_host
        app = create_app(registry, mcp=not args.no_mcp, **options)
    except AdapterError as exc:
        sys.stderr.write(f"agent_adapter.server: {exc}\n")
        return 2
    serve(app, host=args.host, port=args.port)
    return 0


def load_registry(target: str) -> Registry:
    """The `Registry` at `module:attribute` (dots allowed in the attribute path).

    Raises:
        ConfigError: a malformed target, an import failure, or not a Registry.
    """
    module_name, sep, attribute = target.partition(":")
    if not sep or not module_name or not attribute:
        raise ConfigError(f"--registry must look like 'package.module:attribute', got {target!r}")
    try:
        value: Any = importlib.import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load {target!r}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, Registry):
        raise ConfigError(f"{target!r} is a {type(value).__name__}, not a Registry")
    return value


def tokens_from(raw: str) -> dict[str, Principal]:
    """Principals by bearer token from the `AGENT_ADAPTER_TOKENS` JSON ("" -> none).

    Raises:
        ConfigError: not a JSON object of `{"id": str, "scopes": [str]}` entries (the
            message never contains a token).
    """
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"{TOKENS_ENV} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{TOKENS_ENV} must be a JSON object of token -> principal")
    tokens: dict[str, Principal] = {}
    for index, (token, entry) in enumerate(data.items()):
        principal_id = entry.get("id") if isinstance(entry, dict) else None
        scopes = entry.get("scopes", []) if isinstance(entry, dict) else None
        valid = (
            isinstance(principal_id, str)
            and principal_id
            and isinstance(scopes, list)
            and all(isinstance(scope, str) for scope in scopes)
        )
        if not token or not valid:
            raise ConfigError(f"{TOKENS_ENV} entry #{index + 1} needs a token, an id and scopes")
        tokens[token] = Principal(str(principal_id), scopes=frozenset(scopes or ()))
    return tokens


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_adapter.server", description="Serve a tool registry over HTTP."
    )
    parser.add_argument("--registry", required=True, help="package.module:attribute")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-anonymous",
        action="store_true",
        help="accept requests without a token (all anonymous callers share one identity)",
    )
    parser.add_argument(
        "--allow-anonymous-public",
        action="store_true",
        help="also allow anonymous access on a non-local --host",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="a Host header the routes accept (repeatable; default: local names when local)",
    )
    parser.add_argument("--no-mcp", action="store_true", help="do not mount MCP at /mcp")
    return parser


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
