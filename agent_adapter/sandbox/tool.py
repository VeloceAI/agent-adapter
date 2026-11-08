"""`code_tool`: a sandbox as a registry tool that needs a person's approval for every run."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, create_model

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.runtime.current import current_context
from agent_adapter.sandbox._base import GuardedSandbox
from agent_adapter.sandbox._checks import (
    MAX_OUTPUT_BYTES,
    SandboxRequest,
    check_env,
    check_files,
    check_timeout,
    positive_int,
)
from agent_adapter.sandbox.port import CODE_FILES, InputFiles, SandboxPort, SandboxResult

CODE_KIND = "code"
CODE_HELP = "The complete program to run. Print what you need to see; nothing else is returned."

# on_result(result, ctx): the host's hook, e.g. to keep the files a run produced.
OnResult = Callable[[SandboxResult, ToolContext], Any]


def code_tool(
    sandbox: SandboxPort,
    *,
    name: str = "run_code",
    description: str | None = None,
    languages: Iterable[str] = ("python",),
    timeout_s: float = 30.0,
    max_output_bytes: int = 64_000,
    network: bool = False,
    env: Mapping[str, str] | None = None,
    files: InputFiles | None = None,
    on_result: OnResult | None = None,
    tags: Iterable[str] = (),
) -> ToolSpec:
    """A tool (kind "code") that runs the model's code in `sandbox`; every run needs approval.

    `confirm=True` is fixed: there is no option to run model-written code unapproved. Limits,
    network, env and input files are the host's; the model chooses only the code (and the
    language, when more than one is offered).

    Raises:
        ConfigError: bad limits, env, files or languages, or `network=True` on a sandbox
            that does not allow it.
        ToolUsageError: `sandbox` has no async `run`.
    """
    if not isinstance(sandbox, SandboxPort):
        raise ToolUsageError("code_tool needs a SandboxPort (async run(code, ...))")
    langs = _languages(languages, sandbox, name)
    guarded = (
        sandbox if isinstance(sandbox, GuardedSandbox) else PortSandbox(sandbox, langs, network)
    )
    if network and not guarded.network:
        raise ConfigError(f"code tool {name!r}: {guarded.name} does not allow network access")
    seconds = check_timeout(timeout_s)
    cap = positive_int("max_output_bytes", max_output_bytes, 1, MAX_OUTPUT_BYTES)
    try:
        host_env = check_env(env)
        host_files = files if files is None else dict(files)
        for language in langs:
            check_files(host_files, CODE_FILES[language])
    except ToolUsageError as exc:
        raise ConfigError(f"code tool {name!r}: {exc}") from exc

    async def execute(code: str, language: str, ctx: ToolContext) -> dict[str, Any]:
        result = await guarded.run(
            code,
            language=language,
            files=host_files,
            timeout_s=seconds,
            max_output_bytes=cap,
            network=network,
            env=host_env,
            ctx=ctx,
        )
        if on_result is not None:
            done = on_result(result, ctx)
            if inspect.isawaitable(done):
                await done
        return shown(result)

    text = description or _description(langs, seconds, network, guarded.isolated)
    policy: dict[str, Any] = {
        "name": name,
        "description": text,
        "kind": CODE_KIND,
        "tags": frozenset(tags),
        "confirm": True,
        "effects": "destructive",
        "idempotent": False,
    }
    if len(langs) == 1:
        only = langs[0]

        async def run_code(code: str, ctx: ToolContext) -> dict[str, Any]:
            """Run code in the sandbox.

            Args:
                code: The complete program to run. Print what you need to see.
            """
            return await execute(code, only, ctx)

        return build_spec(run_code, **policy)

    async def run_code_in(code: str, language: str, ctx: ToolContext) -> dict[str, Any]:
        """Run code in the sandbox."""
        return await execute(code, language, ctx)

    choice = cast(Any, Literal)[langs]
    model = create_model(
        "RunCodeArgs",
        __config__=ConfigDict(extra="forbid"),
        code=(str, Field(description=CODE_HELP)),
        language=(choice, Field(description="The language the code is written in.")),
    )
    return build_spec(run_code_in, args_model=model, **policy)


def shown(result: SandboxResult) -> dict[str, Any]:
    """The JSON a model sees: exit code, output, flags and the produced files' names and sizes."""
    return {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "files": [{"name": item.name, "size": item.size} for item in result.files],
        "files_truncated": result.files_truncated,
        "duration_ms": result.duration_ms,
    }


def _languages(languages: Iterable[str], sandbox: SandboxPort, name: str) -> tuple[str, ...]:
    """The offered languages, each one the sandbox runs.

    Raises:
        ConfigError: none, repeated, unknown, or not run by `sandbox`.
    """
    langs = tuple(languages) if not isinstance(languages, str) else (languages,)
    if not langs or len(set(langs)) != len(langs):
        raise ConfigError(f"code tool {name!r}: languages must be a non-empty, unique list")
    runs = sandbox.languages if isinstance(sandbox, GuardedSandbox) else tuple(CODE_FILES)
    for language in langs:
        if language not in runs:
            raise ConfigError(f"code tool {name!r}: the sandbox does not run {language!r}")
    return langs


def _description(langs: tuple[str, ...], timeout_s: float, network: bool, isolated: bool) -> str:
    if not isolated:
        where = "on the host with NO isolation (a development sandbox: network, files and "
        where += "processes are the host's)"
    elif network:
        where = "in an isolated sandbox with network access"
    else:
        where = "in an isolated sandbox with no network access"
    return (
        f"Run {' or '.join(langs)} code {where}, for up to {timeout_s:g} seconds. Returns the "
        "exit code, stdout, stderr and the names of files it wrote. A person approves each run."
    )


class PortSandbox(GuardedSandbox):
    """Any other `SandboxPort` behind the shared checks, limits and events."""

    backend = "port"

    def __init__(self, port: SandboxPort, languages: Iterable[str], network: bool) -> None:
        super().__init__(name=type(port).__name__, languages=languages, network=network)
        self.port = port

    async def _run(self, request: SandboxRequest) -> SandboxResult:
        result = await self.port.run(
            request.code,
            language=request.language,
            files=dict(request.files),
            timeout_s=request.timeout_s,
            max_output_bytes=request.max_output_bytes,
            network=request.network,
            env=dict(request.env),
            ctx=current_context(),
        )
        if not isinstance(result, SandboxResult):
            raise ToolUsageError(f"sandbox {self.name!r} must return a SandboxResult")
        return result
