"""`LocalSubprocessSandbox`: DEVELOPMENT ONLY. Runs code as a plain local process, not isolated."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from agent_adapter.core.errors import ConfigError, IntegrationError, log_failure
from agent_adapter.sandbox import _bootstrap, _process
from agent_adapter.sandbox._base import GuardedSandbox
from agent_adapter.sandbox._bootstrap import kill_tree
from agent_adapter.sandbox._checks import Limits, SandboxRequest
from agent_adapter.sandbox._protocol import envelope_cap, parse, payload
from agent_adapter.sandbox.port import SandboxResult

logger = logging.getLogger(__name__)

UNSAFE_WARNING = (
    "LocalSubprocessSandbox is NOT a sandbox: the code runs as this user, with this machine's "
    "files, network and processes. Use it for development only; use DockerSandbox or a hosted "
    "sandbox for untrusted code."
)


class LocalSubprocessSandbox(GuardedSandbox):
    """DEVELOPMENT ONLY: Python in a temporary folder with a time limit and output caps.

    It isolates nothing (files, network, processes and memory are the host's) and refuses to
    start unless `allow_unsafe_local=True`. Every run logs a WARNING. On Windows, processes
    the code leaves running in the background are not killed when it exits.

    Raises:
        ConfigError: `allow_unsafe_local` is not True, or bad limits.
    """

    backend = "local"
    supported = ("python",)
    isolated = False

    def __init__(
        self,
        *,
        allow_unsafe_local: bool = False,
        python: str = sys.executable,
        name: str = "",
        limits: Limits | None = None,
        grace_s: float = 10.0,
        capture_content: bool = False,
    ) -> None:
        if allow_unsafe_local is not True:
            raise ConfigError(
                "LocalSubprocessSandbox runs code without isolation; pass "
                "allow_unsafe_local=True to use it for development, or use DockerSandbox"
            )
        super().__init__(
            name=name,
            languages=("python",),
            network=True,  # it cannot turn the network off; requests for it are honoured as-is
            limits=limits,
            grace_s=grace_s,
            capture_content=capture_content,
        )
        if not python or not os.path.isabs(python):
            raise ConfigError("python must be an absolute path to an interpreter")
        self.python = python

    async def _run(self, request: SandboxRequest) -> SandboxResult:
        logger.warning(UNSAFE_WARNING)
        workdir = tempfile.mkdtemp(prefix="aa-sandbox-")
        try:
            argv = [self.python, "-I", "-S", _bootstrap_path()]
            env = _host_env(workdir)
            proc = _process.start(argv, env=env, cwd=workdir, **_group())
            finished = await _process.run_process(
                proc,
                payload(request, workdir=workdir),
                timeout_s=request.timeout_s + self.grace_s / 2,
                max_stdout=envelope_cap(request),
                kill=kill_tree,
            )
            if finished.timed_out:
                logger.warning("%s: the runner did not stop in time; killed it", self.name)
                return SandboxResult(exit_code=-1, timed_out=True, backend=self.backend)
            if finished.stdout_over:
                raise IntegrationError(
                    f"{self.name}: the runner printed more than its limit", integration=self.backend
                )
            return parse(finished.stdout, backend=self.backend, duration_ms=finished.duration_ms)
        finally:
            _remove(workdir)


def _bootstrap_path() -> str:
    return os.path.abspath(_bootstrap.__file__)


def _host_env(workdir: str) -> dict[str, str]:
    """PATH and the Windows basics only: the host's secrets stay out of the code's reach."""
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": workdir}
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


def _group() -> dict[str, Any]:
    """Start the runner in its own process group, so a kill reaches its children too."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _remove(workdir: str) -> None:
    try:
        shutil.rmtree(workdir)
    except OSError as exc:
        # Leaving a temp folder behind must not hide the run's result.
        log_failure(logger, logging.WARNING, exc, "could not remove a sandbox work folder")
