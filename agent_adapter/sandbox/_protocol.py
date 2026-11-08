"""The host side of the runner protocol: the JSON payload in and the JSON envelope out."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

from agent_adapter.core.errors import IntegrationError
from agent_adapter.sandbox import _bootstrap
from agent_adapter.sandbox._bootstrap import PROTOCOL
from agent_adapter.sandbox._checks import SandboxRequest
from agent_adapter.sandbox.port import COMMANDS, SandboxFile, SandboxResult

ENVELOPE_SLACK = 64_000  # JSON keys and padding around the base64 data


@cache
def bootstrap_source() -> str:
    """The runner's source, as Docker passes it to the image's `python3 -c`."""
    return Path(_bootstrap.__file__).read_text(encoding="utf-8")


def payload(request: SandboxRequest, *, workdir: str | None = None) -> bytes:
    """The request as the runner reads it from stdin."""
    command: Sequence[str] | None = COMMANDS.get(request.language)
    body = {
        "v": PROTOCOL,
        "workdir": workdir,
        "code_file": request.code_file,
        "code": request.code,
        "command": list(command) if command else None,
        "files": {
            name: base64.b64encode(data).decode("ascii") for name, data in request.files.items()
        },
        "env": dict(request.env),
        "timeout_s": request.timeout_s,
        "max_output": request.max_output_bytes,
        "max_files": request.limits.max_files,
        "max_file_bytes": request.limits.max_file_bytes,
    }
    return json.dumps(body).encode("utf-8")


def envelope_cap(request: SandboxRequest) -> int:
    """The most stdout bytes a well-behaved runner can print for `request`."""
    data = 2 * request.max_output_bytes + request.limits.max_file_bytes
    return data * 4 // 3 + 512 * (request.limits.max_files + 2) + ENVELOPE_SLACK


def parse(stdout: bytes, *, backend: str, duration_ms: float) -> SandboxResult:
    """The result in the runner's last stdout line.

    Raises:
        IntegrationError: no readable envelope (the runner did not finish or was tampered with).
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        raw = json.loads(lines[-1]) if lines else None
        if not isinstance(raw, dict) or raw.get("v") != PROTOCOL or "error" in raw:
            raise ValueError("not a runner envelope")
        return _result(raw, backend, duration_ms)
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise IntegrationError(
            f"{backend} sandbox returned no readable result", integration=backend
        ) from exc


def _text(encoded: Any) -> str:
    return base64.b64decode(encoded, validate=True).decode("utf-8", errors="replace")


def _result(raw: dict[str, Any], backend: str, duration_ms: float) -> SandboxResult:
    files = tuple(
        SandboxFile(
            name=str(item["name"]),
            data=base64.b64decode(item["data"], validate=True),
            size=int(item["size"]),
        )
        for item in raw["files"]
    )
    return SandboxResult(
        exit_code=int(raw["exit_code"]),
        stdout=_text(raw["stdout"]),
        stderr=_text(raw["stderr"]),
        stdout_truncated=bool(raw["stdout_truncated"]),
        stderr_truncated=bool(raw["stderr_truncated"]),
        timed_out=bool(raw["timed_out"]),
        duration_ms=duration_ms,
        files=files,
        files_truncated=bool(raw["files_truncated"]),
        backend=backend,
    )
