"""Read pip's index settings (config files and environment) without running pip."""

from __future__ import annotations

import configparser
import logging
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_adapter.cli.findings import Finding

logger = logging.getLogger(__name__)

_CREDENTIALS = re.compile(r"//[^/@]*@")
_TOKEN = re.compile(r"(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9_-]{16,}")  # e.g. a Gemfury push token
INSTALL_DOC = "docs/get-started/install.md"
_ENV_KEYS = (("PIP_INDEX_URL", "index-url"), ("PIP_EXTRA_INDEX_URL", "extra-index-url"))


@dataclass(frozen=True)
class IndexSetting:
    """One `index-url` or `extra-index-url` value and where it was set."""

    key: str
    url: str
    source: str


def redact(url: str) -> str:
    """`url` without user, password, query, fragment or token-like path segments."""
    bare = _CREDENTIALS.sub("//***@", url).split("?", 1)[0].split("#", 1)[0]
    head, sep, path = bare.partition("://")
    if not sep:
        return bare
    host, slash, rest = path.partition("/")
    segments = ["***" if _TOKEN.fullmatch(part) else part for part in rest.split("/")]
    return f"{head}://{host}{slash}{'/'.join(segments)}"


def config_files(
    environ: Mapping[str, str], *, windows: bool | None = None, prefix: str | None = None
) -> list[Path]:
    """pip's config files, lowest priority first: global, user, site, then PIP_CONFIG_FILE."""
    windows = sys.platform == "win32" if windows is None else windows
    home = Path(environ.get("USERPROFILE" if windows else "HOME", "~")).expanduser()
    name = "pip.ini" if windows else "pip.conf"
    if windows:
        program_data = Path(environ.get("PROGRAMDATA", r"C:\ProgramData"))
        appdata = Path(environ.get("APPDATA", str(home / "AppData" / "Roaming")))
        files = [program_data / "pip" / name, home / "pip" / name, appdata / "pip" / name]
    else:
        xdg_dirs = environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(os.pathsep)
        xdg_home = Path(environ.get("XDG_CONFIG_HOME", str(home / ".config")))
        files = [
            *(Path(d) / "pip" / name for d in xdg_dirs if d),
            Path("/etc") / name,
            home / ".pip" / name,
            xdg_home / "pip" / name,
        ]
    files.append(Path(prefix if prefix is not None else sys.prefix) / name)
    if environ.get("PIP_CONFIG_FILE"):
        files.append(Path(environ["PIP_CONFIG_FILE"]))
    return files


def read_settings(
    files: list[Path], environ: Mapping[str, str]
) -> tuple[list[IndexSetting], list[Finding]]:
    """Every index setting in `files` and `environ`, lowest priority first, plus read errors."""
    settings: list[IndexSetting] = []
    problems: list[Finding] = []
    for path in files:
        if not path.is_file():
            continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeDecodeError) as exc:
            # Not the message: a parse error quotes the line, which may hold a token.
            logger.warning("cannot read pip config %s: %s", path, type(exc).__name__)
            problems.append(
                Finding("warn", "pip index", f"cannot read {path} ({type(exc).__name__})")
            )
            continue
        for section in ("global", "install"):
            for key in ("index-url", "extra-index-url"):
                value = parser.get(section, key, fallback="")
                settings.extend(IndexSetting(key, url, str(path)) for url in value.split() if url)
    for variable, key in (
        ("PIP_INDEX_URL", "index-url"),
        ("PIP_EXTRA_INDEX_URL", "extra-index-url"),
    ):
        settings.extend(
            IndexSetting(key, url, f"${variable}") for url in environ.get(variable, "").split()
        )
    return settings, problems


def index_findings(environ: Mapping[str, str], files: list[Path] | None = None) -> list[Finding]:
    """Which index does pip use, and does any `extra-index-url` mix in a second one?"""
    settings, findings = read_settings(config_files(environ) if files is None else files, environ)
    indexes = [s for s in settings if s.key == "index-url"]
    extras = [s for s in settings if s.key == "extra-index-url"]
    if not indexes:
        findings.append(Finding("ok", "pip index", "public PyPI (no index-url set)"))
    else:
        chosen = indexes[-1]
        findings.append(Finding("ok", "pip index", f"{redact(chosen.url)} (from {chosen.source})"))
    for extra in extras:
        findings.append(
            Finding(
                "warn",
                "pip index",
                f"extra-index-url {redact(extra.url)} (from {extra.source}): pip takes "
                "the highest version from any index, which allows dependency confusion "
                f"({INSTALL_DOC})",
            )
        )
    uv = sorted(v for v in ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX") if environ.get(v))
    if uv:
        findings.append(Finding("ok", "uv index", f"set by {', '.join(uv)}"))
    if environ.get("UV_EXTRA_INDEX_URL"):
        findings.append(
            Finding("warn", "uv index", "UV_EXTRA_INDEX_URL is set: use UV_DEFAULT_INDEX")
        )
    return findings
