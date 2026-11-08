"""The shared front of every sandbox: input checks, hard limits on what comes back, events."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import AdapterError, ConfigError, IntegrationError, log_failure
from agent_adapter.runtime.current import current_context, use_context
from agent_adapter.runtime.sync import run_sync
from agent_adapter.sandbox._checks import (
    MAX_OUTPUT_BYTES,
    Limits,
    SandboxRequest,
    check_code,
    check_env,
    check_files,
    check_language,
    check_timeout,
    is_safe_name,
    positive_int,
)
from agent_adapter.sandbox.port import CODE_FILES, InputFiles, SandboxFile, SandboxResult
from agent_adapter.tracing.events import EventType, emit, make_event, require_recorded

logger = logging.getLogger(__name__)


class GuardedSandbox(ABC):
    """A `SandboxPort` that checks every request and re-applies its limits to every result.

    Backends implement `_run`. Output and files over the limits are cut here too, so a
    backend that ignores them fails closed (less output), never open.
    """

    backend = "sandbox"
    supported: tuple[str, ...] = tuple(CODE_FILES)
    isolated = True  # False: the code shares the host (development only)

    def __init__(
        self,
        *,
        name: str = "",
        languages: Iterable[str] = ("python",),
        network: bool = False,
        limits: Limits | None = None,
        grace_s: float = 30.0,
        capture_content: bool = False,
    ) -> None:
        self.name = name or self.backend
        self.languages = tuple(languages)
        if not self.languages:
            raise ConfigError(f"{self.name}: languages must not be empty")
        for language in self.languages:
            if language not in self.supported:
                raise ConfigError(f"{self.name} cannot run {language!r}; it runs {self.supported}")
        self.network = bool(network)
        self.limits = limits if limits is not None else Limits()
        self.grace_s = check_timeout(grace_s, "grace_s")
        self.capture_content = bool(capture_content)

    def request(
        self,
        code: str,
        *,
        language: str = "python",
        files: InputFiles | None = None,
        timeout_s: float = 30.0,
        max_output_bytes: int = 64_000,
        network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> SandboxRequest:
        """The checked request `run` would send to the backend.

        Raises:
            ToolUsageError: bad code, language, files or env.
            ConfigError: bad limits, or network asked of a sandbox that does not allow it.
        """
        language = check_language(language, self.languages)
        if network and not self.network:
            raise ConfigError(f"{self.name} does not allow network access (network=False)")
        return SandboxRequest(
            code=check_code(code),
            language=language,
            timeout_s=check_timeout(timeout_s),
            max_output_bytes=positive_int(
                "max_output_bytes", max_output_bytes, 1, MAX_OUTPUT_BYTES
            ),
            network=bool(network),
            files=check_files(files, CODE_FILES[language]),
            env=check_env(env),
            limits=self.limits,
        )

    async def run(
        self,
        code: str,
        *,
        language: str = "python",
        files: InputFiles | None = None,
        timeout_s: float = 30.0,
        max_output_bytes: int = 64_000,
        network: bool = False,
        env: Mapping[str, str] | None = None,
        ctx: ToolContext | None = None,
    ) -> SandboxResult:
        """Run `code` in the sandbox and return its bounded result; a timeout is a result.

        Raises:
            ToolUsageError: bad code, language, files or env.
            ConfigError: bad limits, or network asked of a sandbox that does not allow it.
            IntegrationError: the backend failed (the cause is chained).
            AdapterError: a fail-closed audit sink could not record the start; nothing ran.
        """
        request = self.request(
            code,
            language=language,
            files=files,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            network=network,
            env=env,
        )
        ctx = ctx if ctx is not None else current_context()
        started = time.perf_counter()
        self._event(
            EventType.SANDBOX_STARTED,
            ctx,
            request,
            {"arguments": {"code": request.code}},
            code_bytes=len(request.code.encode("utf-8", errors="replace")),
            files=len(request.files),
            network=request.network,
            timeout_s=request.timeout_s,
        )
        await require_recorded(ctx.sink if ctx is not None else None)  # fail-closed audit: no run
        try:
            raw = await self._call(request, ctx)
        except Exception as exc:
            kind = exc.kind if isinstance(exc, AdapterError) else "integration-error"
            duration = _ms(started)
            self._event(
                EventType.SANDBOX_FAILED, ctx, request, error_kind=kind, duration_ms=duration
            )
            raise
        result = bounded(raw, request, self.backend, _ms(started))
        self._event(
            EventType.SANDBOX_FINISHED,
            ctx,
            request,
            {"result": {"stdout": result.stdout, "stderr": result.stderr}},
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
            stdout_bytes=len(result.stdout.encode("utf-8", errors="replace")),
            stderr_bytes=len(result.stderr.encode("utf-8", errors="replace")),
            truncated=result.truncated,
            files=len(result.files),
        )
        return result

    def run_sync(self, code: str, *, timeout: float | None = None, **kw: Any) -> SandboxResult:
        """`run` from sync code (inside a running loop, on a worker thread)."""
        return run_sync(lambda: self.run(code, **kw), timeout=timeout)

    @abstractmethod
    async def _run(self, request: SandboxRequest) -> SandboxResult: ...

    async def _call(self, request: SandboxRequest, ctx: ToolContext | None) -> SandboxResult:
        """`_run` under a backstop of the run's time limit plus `grace_s`."""
        limit = request.timeout_s + self.grace_s
        try:
            async with asyncio.timeout(limit) as scope:
                if ctx is None:
                    return await self._run(request)
                with use_context(ctx):
                    return await self._run(request)
        except AdapterError:
            raise
        except TimeoutError as exc:
            if not scope.expired():
                raise self._failed(exc) from exc
            logger.warning(
                "%s sandbox gave no result within %ss; reported as timed out", self.name, limit
            )
            return SandboxResult(exit_code=-1, timed_out=True, backend=self.backend)
        except Exception as exc:
            raise self._failed(exc) from exc

    def _failed(self, exc: BaseException) -> IntegrationError:
        log_failure(logger, logging.ERROR, exc, "%s sandbox run failed", self.name)
        return IntegrationError(f"{self.name} sandbox run failed", integration=self.backend)

    def _event(
        self,
        type: str,
        ctx: ToolContext | None,
        request: SandboxRequest,
        content: dict[str, Any] | None = None,
        **data: Any,
    ) -> None:
        if ctx is None or ctx.sink is None:
            return
        extra = content if self.capture_content and content else {}
        event = make_event(
            type,
            ctx,
            sandbox=self.name,
            backend=self.backend,
            language=request.language,
            **data,
            **extra,
        )
        emit(ctx.sink, event)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def cap_text(text: str, limit: int) -> tuple[str, bool]:
    """`text` cut to at most `limit` UTF-8 bytes, and whether it was cut."""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", errors="ignore"), True


def bounded(
    result: SandboxResult, request: SandboxRequest, backend: str, duration_ms: float
) -> SandboxResult:
    """`result` within the request's output and file limits, timed on the host.

    Raises:
        IntegrationError: the backend returned something other than a `SandboxResult`.
    """
    if not isinstance(result, SandboxResult):
        raise IntegrationError(
            f"{backend} sandbox must return a SandboxResult", integration=backend
        )
    stdout, stdout_cut = cap_text(result.stdout, request.max_output_bytes)
    stderr, stderr_cut = cap_text(result.stderr, request.max_output_bytes)
    files, files_cut = _bounded_files(result.files, request.limits)
    return result.model_copy(
        update={
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": result.stdout_truncated or stdout_cut,
            "stderr_truncated": result.stderr_truncated or stderr_cut,
            "files": files,
            "files_truncated": result.files_truncated or files_cut,
            "duration_ms": duration_ms,
            "backend": backend,
        }
    )


def _bounded_files(
    files: tuple[SandboxFile, ...], limits: Limits
) -> tuple[tuple[SandboxFile, ...], bool]:
    """Files with safe, unique names, within the count and byte limits; and whether any were cut.

    Produced names come from the code (or a service), so an unsafe one is dropped, never used.
    """
    seen: set[str] = set()
    unique: list[SandboxFile] = []
    for item in files:
        if item.name not in seen and is_safe_name(item.name):
            seen.add(item.name)
            unique.append(item)
    if len(unique) < len(files):
        logger.warning(
            "dropped %d produced files with unsafe or repeated names", len(files) - len(unique)
        )
    kept: list[SandboxFile] = []
    budget = limits.max_file_bytes
    cut = len(unique) < len(files) or len(unique) > limits.max_files
    for item in unique[: limits.max_files]:
        data = item.data[:budget]
        budget -= len(data)
        cut = cut or len(data) < len(item.data)
        size = max(item.size, len(item.data))
        kept.append(
            item
            if len(data) == len(item.data) and size == item.size
            else SandboxFile(name=item.name, data=data, size=size)
        )
    return tuple(kept), cut
