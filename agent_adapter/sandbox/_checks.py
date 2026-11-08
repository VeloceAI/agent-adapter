"""Input checks every sandbox shares, and the checked `SandboxRequest` backends receive."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.sandbox.port import CODE_FILES, InputFiles

MAX_CODE_BYTES = 256_000
MAX_TIMEOUT_S = 600.0
MAX_OUTPUT_BYTES = 10_000_000
MAX_INPUT_FILES = 64
MAX_INPUT_BYTES = 10_000_000
MAX_ENV = 64
MAX_NAME_CHARS = 255

_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_PART = re.compile(r"[A-Za-z0-9._-]+")
# Windows device names, unsafe as a file name on a Windows host whatever the extension.
_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in "123456789"}
)


def positive_int(name: str, value: object, minimum: int = 1, maximum: int | None = None) -> int:
    """`value` as an int in [minimum, maximum].

    Raises:
        ConfigError: not an int (a bool is not), or out of range.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an int >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value!r}")
    return value


@dataclass(frozen=True)
class Limits:
    """Bounds on what a backend returns: produced files and their bytes."""

    max_files: int = 16
    max_file_bytes: int = 1_000_000  # all produced files together

    def __post_init__(self) -> None:
        positive_int("max_files", self.max_files, 0)
        positive_int("max_file_bytes", self.max_file_bytes, 0)


@dataclass(frozen=True)
class SandboxRequest:
    """A checked run: what a backend's `_run` receives."""

    code: str
    language: str
    timeout_s: float
    max_output_bytes: int
    network: bool
    files: Mapping[str, bytes] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    limits: Limits = Limits()

    @property
    def code_file(self) -> str:
        """The file name the code is written to."""
        return CODE_FILES[self.language]


def check_timeout(value: object, name: str = "timeout_s") -> float:
    """A finite number of seconds in (0, MAX_TIMEOUT_S].

    Raises:
        ConfigError: not a number, not finite, or out of range.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number, got {type(value).__name__}")
    seconds = float(value)
    if not (math.isfinite(seconds) and 0 < seconds <= MAX_TIMEOUT_S):
        raise ConfigError(f"{name} must be > 0 and <= {MAX_TIMEOUT_S}, got {value!r}")
    return seconds


def check_language(language: object, supported: tuple[str, ...]) -> str:
    """`language` when this sandbox runs it.

    Raises:
        ToolUsageError: a language this sandbox does not run.
    """
    if not isinstance(language, str) or language not in supported:
        raise ToolUsageError(f"language must be one of {list(supported)}, got {language!r}")
    return language


def check_code(code: object) -> str:
    """Non-blank code within MAX_CODE_BYTES.

    Raises:
        ToolUsageError: not a str, blank, too long, or holding a NUL byte.
    """
    if not isinstance(code, str) or not code.strip():
        raise ToolUsageError("code must be a non-blank str")
    if "\x00" in code:
        raise ToolUsageError("code must not hold a NUL byte")
    if len(code.encode("utf-8", errors="surrogatepass")) > MAX_CODE_BYTES:
        raise ToolUsageError(f"code is longer than {MAX_CODE_BYTES} bytes")
    return code


def check_name(name: object) -> str:
    """A relative POSIX path of plain parts, never absolute, hidden-parent or a drive.

    Raises:
        ToolUsageError: an unsafe or over-long file name.
    """
    if not isinstance(name, str) or not name or len(name) > MAX_NAME_CHARS:
        raise ToolUsageError(f"file names must be 1 to {MAX_NAME_CHARS} characters")
    parts = name.split("/")
    if any(part in ("", ".", "..") or not _PART.fullmatch(part) for part in parts):
        raise ToolUsageError(f"unsafe file name {name!r}: use relative names of [A-Za-z0-9._-]")
    if any(part.split(".")[0].upper() in _DEVICES or part.endswith(".") for part in parts):
        raise ToolUsageError(f"unsafe file name {name!r}: a device name or a trailing dot")
    return name


def is_safe_name(name: object) -> bool:
    """True when `check_name` accepts `name`."""
    try:
        check_name(name)
    except ToolUsageError:
        return False
    return True


def check_files(files: InputFiles | None, code_file: str) -> dict[str, bytes]:
    """Input files as bytes under safe names, bounded in count and total size.

    Raises:
        ToolUsageError: too many, too large, an unsafe name, a name clash, or a bad value.
    """
    if files is None:
        return {}
    if not isinstance(files, Mapping):
        raise ToolUsageError("files must be a mapping of name to bytes or str")
    if len(files) > MAX_INPUT_FILES:
        raise ToolUsageError(f"at most {MAX_INPUT_FILES} input files")
    checked: dict[str, bytes] = {}
    folders: set[str] = set()
    for name, value in files.items():
        name = check_name(name)
        if name == code_file:
            raise ToolUsageError(f"input file {name!r} would overwrite the code")
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes):
            raise ToolUsageError(f"input file {name!r} must be bytes or str")
        checked[name] = value
        parts = name.split("/")
        folders.update("/".join(parts[:i]) for i in range(1, len(parts)))
    if folders & checked.keys():
        raise ToolUsageError("an input file name is also used as a folder")
    if sum(len(data) for data in checked.values()) > MAX_INPUT_BYTES:
        raise ToolUsageError(f"input files are larger than {MAX_INPUT_BYTES} bytes")
    return checked


def check_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Environment variables with plain names and NUL-free str values.

    Raises:
        ToolUsageError: too many, a bad name or a bad value.
    """
    if env is None:
        return {}
    if not isinstance(env, Mapping):
        raise ToolUsageError("env must be a mapping of str to str")
    if len(env) > MAX_ENV:
        raise ToolUsageError(f"at most {MAX_ENV} environment variables")
    checked: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_KEY.fullmatch(key):
            raise ToolUsageError(f"bad environment variable name {key!r}")
        if not isinstance(value, str) or "\x00" in value:
            raise ToolUsageError(f"environment variable {key} must be a NUL-free str")
        checked[key] = value
    return checked
