"""Load an OpenAPI 3.x document (size-checked) and pick the server URL."""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent_adapter.core.errors import ConfigError, MissingDependency
from agent_adapter.integrations.openapi._refs import check_document

_YAML_HINT = 'YAML specs need pyyaml: pip install "agent-adapter[openapi]"'


def load_document(source: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    """The document as a dict: a mapping (copied), document text, or a file path.

    A `str` holding a newline or starting with `{` is document text; any other `str` is a
    path. URLs are refused: fetch the document yourself.

    Raises:
        ConfigError: unreadable, unparsable or not OpenAPI 3.x.
        MissingDependency: YAML without pyyaml.
    """
    if isinstance(source, Mapping):
        check_document(dict(source))  # before copying: deepcopy recurses
        document: Any = copy.deepcopy(dict(source))
    elif isinstance(source, str) and ("\n" in source or source.lstrip().startswith("{")):
        document = _parse(source, yaml_file=False)
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        raise ConfigError(f"import_openapi does not fetch URLs: download {source} and pass it")
    elif isinstance(source, str | os.PathLike):
        document = _read(Path(source))
    else:
        raise ConfigError(f"an OpenAPI document must be a mapping, text or a path, got {source!r}")
    if not isinstance(document, dict):
        raise ConfigError("an OpenAPI document must be a mapping at the root")
    check_document(document)
    version = document.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        found = version or document.get("swagger") or "none"
        raise ConfigError(f"only OpenAPI 3.x documents are supported (version: {found})")
    if not isinstance(document.get("paths", {}), Mapping):
        raise ConfigError("the OpenAPI document's 'paths' must be a mapping")
    return document


def _read(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read OpenAPI document {path}: {exc.strerror}") from exc
    return _parse(text, yaml_file=path.suffix.lower() in (".yaml", ".yml"))


def _parse(text: str, *, yaml_file: bool) -> Any:
    try:
        return _parse_text(text, yaml_file=yaml_file)
    except RecursionError as exc:
        raise ConfigError("the OpenAPI document is nested too deeply to parse") from exc


def _parse_text(text: str, *, yaml_file: bool) -> Any:
    if not yaml_file and text.lstrip().startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"the OpenAPI document is not valid JSON: {exc}") from exc
    try:
        import yaml
    except ImportError as exc:
        raise MissingDependency(_YAML_HINT) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"the OpenAPI document is not valid YAML: {exc}") from exc


def server_url(document: Mapping[str, Any], base_url: str | None) -> str:
    """`base_url`, else the first server URL with its variables' defaults filled in.

    Raises:
        ConfigError: no absolute URL is available.
    """
    if base_url is not None:
        url = base_url
    else:
        servers = document.get("servers") or [{}]
        server = servers[0] if isinstance(servers, list) and servers else {}
        url = str(server.get("url", "")) if isinstance(server, Mapping) else ""
        variables = server.get("variables", {}) if isinstance(server, Mapping) else {}
        url = re.sub(r"\{([^}]+)\}", lambda m: _variable(variables, m.group(1)), url)
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ConfigError(
            f"no absolute server URL ({url or 'none'} in the document): pass base_url="
        )
    _check_authority(url)
    return url


def _check_authority(url: str) -> None:
    """No credentials in the URL, and no ambiguous numeric host (octal, hex, short forms)."""
    parts = urlsplit(url)
    if "@" in parts.netloc:
        raise ConfigError("the server URL contains credentials: put credentials in auth=")
    host = parts.hostname or ""
    labels = host.split(".")
    if host and all(re.fullmatch(r"0x[0-9a-f]*|[0-9]+", label, re.IGNORECASE) for label in labels):
        try:
            ipaddress.IPv4Address(host)  # four plain decimal octets only
        except ValueError:
            raise ConfigError(
                f"ambiguous numeric host {host!r}: write the IPv4 address as a.b.c.d"
            ) from None


def _variable(variables: Any, name: str) -> str:
    variable = variables.get(name) if isinstance(variables, Mapping) else None
    if not isinstance(variable, Mapping) or "default" not in variable:
        raise ConfigError(f"server variable {name!r} has no default: pass base_url=")
    value = str(variable["default"])
    if any(char in value for char in "@/\\?#"):
        raise ConfigError(f"server variable {name!r} may not contain @, /, backslash, ? or #")
    return value
