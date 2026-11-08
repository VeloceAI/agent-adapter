"""`CallableSandbox`: any code-execution service (E2B, Modal, Daytona, your own) as a sandbox."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_adapter.core.callables import is_async
from agent_adapter.core.errors import IntegrationError, ToolUsageError
from agent_adapter.sandbox._base import GuardedSandbox
from agent_adapter.sandbox._checks import Limits, SandboxRequest
from agent_adapter.sandbox.port import SandboxFile, SandboxResult

# fn(request) -> SandboxResult, or a mapping of its fields; sync or async.
RunFn = Callable[[SandboxRequest], Any]


class CallableSandbox(GuardedSandbox):
    """Wraps `fn(request)`; the service does the isolating, this class checks and bounds.

    `network=True` lets requests ask for network access; your service must enforce
    `request.network` and `request.timeout_s` itself.

    Raises:
        ToolUsageError: `fn` is not callable.
        ConfigError: bad languages, limits or `grace_s`.
    """

    backend = "callable"

    def __init__(
        self,
        fn: RunFn,
        *,
        name: str = "",
        languages: Iterable[str] = ("python",),
        network: bool = False,
        limits: Limits | None = None,
        grace_s: float = 30.0,
        capture_content: bool = False,
    ) -> None:
        if not callable(fn):
            raise ToolUsageError(f"CallableSandbox fn must be callable, got {type(fn).__name__}")
        super().__init__(
            name=name,
            languages=languages,
            network=network,
            limits=limits,
            grace_s=grace_s,
            capture_content=capture_content,
        )
        self._fn = fn
        self._is_async = is_async(fn)

    async def _run(self, request: SandboxRequest) -> SandboxResult:
        if self._is_async:
            raw = await self._fn(request)
        else:
            raw = await asyncio.to_thread(self._fn, request)
        if inspect.isawaitable(raw):
            raw = await raw
        return as_result(raw, self.name)


def as_result(raw: Any, name: str) -> SandboxResult:
    """`raw` (a `SandboxResult` or a mapping of its fields) as a `SandboxResult`.

    Raises:
        IntegrationError: `raw` is neither, or its fields are invalid.
    """
    if isinstance(raw, SandboxResult):
        return raw
    if not isinstance(raw, Mapping):
        raise IntegrationError(
            f"sandbox {name!r} must return a SandboxResult or a mapping, got {type(raw).__name__}",
            integration="callable",
        )
    fields = dict(raw)
    try:
        files = fields.get("files", ())
        if isinstance(files, Mapping):
            fields["files"] = tuple(_file(key, value) for key, value in files.items())
        return SandboxResult.model_validate(fields)
    except ValueError as exc:
        raise IntegrationError(
            f"sandbox {name!r} returned an invalid result", integration="callable"
        ) from exc


def _file(name: Any, value: Any) -> SandboxFile | Any:
    """A `{name: bytes | str}` entry as a file; anything else is left to validation."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    if isinstance(value, bytes):
        return SandboxFile(name=str(name), data=value, size=len(value))
    return value
