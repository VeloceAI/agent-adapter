"""`DockerSandbox`: each run in a fresh, locked-down Linux container, driven by the `docker` CLI."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from agent_adapter.core.errors import ConfigError, IntegrationError, log_failure
from agent_adapter.sandbox import _process
from agent_adapter.sandbox._base import GuardedSandbox
from agent_adapter.sandbox._checks import Limits, SandboxRequest, positive_int
from agent_adapter.sandbox._protocol import bootstrap_source, envelope_cap, parse, payload
from agent_adapter.sandbox.port import SandboxResult

logger = logging.getLogger(__name__)

WORKDIR = "/work"
LABEL = "agent-adapter.sandbox=1"
# Exit codes of `docker run` itself: the daemon failed, or the command could not run.
DOCKER_FAILED = {
    125: "docker could not start the container",
    126: "the runner cannot run",
    127: "the image has no such command",
}

_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}")
_USER = re.compile(r"(\d{1,10}):(\d{1,10})")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class DockerStatus:
    """Whether Docker can run Linux containers here, with the versions found."""

    available: bool
    client_version: str = ""
    server_version: str = ""
    reason: str = ""  # why not, when not available


def docker_status(docker: str = "docker", *, timeout_s: float = 10.0) -> DockerStatus:
    """Probe the `docker` CLI and its daemon; never raises."""
    path = shutil.which(docker)
    if path is None:
        return DockerStatus(False, reason=f"{docker!r} is not on PATH")
    probe = "{{.Client.Version}}|{{.Server.Version}}|{{.Server.Os}}"
    try:
        done = subprocess.run(
            [path, "version", "--format", probe],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log_failure(logger, logging.WARNING, exc, "docker version failed")
        return DockerStatus(False, reason=f"docker version failed ({type(exc).__name__})")
    client, _, rest = done.stdout.decode("utf-8", errors="replace").strip().partition("|")
    server, _, os_name = rest.partition("|")
    if done.returncode != 0 or not server:
        return DockerStatus(False, client, reason="the Docker daemon is not reachable")
    if os_name.strip() != "linux":
        reason = f"the daemon runs {os_name.strip() or 'unknown'} containers; Linux is needed"
        return DockerStatus(False, client, server, reason)
    return DockerStatus(True, client, server)


class DockerSandbox(GuardedSandbox):
    """Runs code in a throwaway container: no network, read-only root, no capabilities.

    The image needs `python3` (the runner), plus `node` or `bash` for those languages. It is
    never pulled at run time: `docker pull` it first. `network=True` lets a request use the
    default `bridge` network: unrestricted egress, including cloud metadata addresses and
    the host's own services.

    Raises:
        ConfigError: a bad image, user (root uid or gid is refused), runtime or limit.
    """

    backend = "docker"

    def __init__(
        self,
        image: str = "python:3.12-slim",
        *,
        cpus: float = 1.0,
        memory_mb: int = 512,
        pids: int = 128,
        network: bool = False,
        user: str = "65534:65534",
        runtime: str | None = None,
        workdir_mb: int = 64,
        tmp_mb: int = 16,
        python: str = "python3",
        docker: str = "docker",
        languages: Iterable[str] = ("python",),
        name: str = "",
        limits: Limits | None = None,
        grace_s: float = 30.0,
        capture_content: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            languages=languages,
            network=network,
            limits=limits,
            grace_s=grace_s,
            capture_content=capture_content,
        )
        if not isinstance(image, str) or not _IMAGE.fullmatch(image):
            raise ConfigError(f"bad image name {image!r}")
        if isinstance(cpus, bool) or not isinstance(cpus, int | float) or not 0 < cpus <= 64:
            raise ConfigError(f"cpus must be > 0 and <= 64, got {cpus!r}")
        self.image = image
        self.cpus = float(cpus)
        self.memory_mb = positive_int("memory_mb", memory_mb, 32)
        self.pids = positive_int("pids", pids, 16)
        self.user = _check_user(user)
        self.runtime = _word("runtime", runtime) if runtime is not None else None
        self.workdir_mb = positive_int("workdir_mb", workdir_mb, 1)
        self.tmp_mb = positive_int("tmp_mb", tmp_mb, 1)
        self.python = _word("python", python)
        self.docker = docker

    def command_line(self, request: SandboxRequest, *, container: str) -> list[str]:
        """The exact `docker run` argv for `request` (the code goes on stdin, not here)."""
        argv = [self.docker, "run", "--rm", "-i", "--name", container, "--label", LABEL]
        argv += ["--pull", "never", "--network", "bridge" if request.network else "none"]
        argv += ["--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
        argv += ["--ipc", "none", "--user", self.user, "--pids-limit", str(self.pids)]
        argv += ["--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m"]
        argv += ["--cpus", f"{self.cpus:g}", "--ulimit", "nofile=1024:1024"]
        argv += ["--tmpfs", f"{WORKDIR}:rw,nosuid,nodev,noexec,size={self.workdir_mb}m,mode=1777"]
        argv += ["--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={self.tmp_mb}m,mode=1777"]
        argv += ["--workdir", WORKDIR]
        if self.runtime is not None:
            argv += ["--runtime", self.runtime]
        argv += ["--entrypoint", self.python, self.image, "-I", "-S", "-c", bootstrap_source()]
        return argv

    def status(self) -> DockerStatus:
        """`docker_status` for this sandbox's CLI."""
        return docker_status(self.docker)

    async def _run(self, request: SandboxRequest) -> SandboxResult:
        path = shutil.which(self.docker)
        if path is None:
            raise IntegrationError(f"{self.docker!r} is not on PATH", integration=self.backend)
        container = f"aa-sandbox-{uuid.uuid4().hex[:16]}"
        argv = [path, *self.command_line(request, container=container)[1:]]

        def kill(proc: subprocess.Popen[bytes]) -> None:
            self._remove(path, container)
            proc.kill()

        proc = _process.start(argv)
        finished = await _process.run_process(
            proc,
            payload(request, workdir=WORKDIR),
            timeout_s=request.timeout_s + self.grace_s / 2,
            max_stdout=envelope_cap(request),
            kill=kill,
        )
        return self._result(finished)

    def _result(self, finished: _process.Finished) -> SandboxResult:
        """The run's result from the finished `docker run`.

        Raises:
            IntegrationError: Docker failed, or the runner printed no readable result.
        """
        if finished.timed_out:
            logger.warning("%s: the container did not stop in time; removed it", self.name)
            return SandboxResult(exit_code=-1, timed_out=True, backend=self.backend)
        if finished.stdout_over:
            raise IntegrationError(
                f"{self.name}: the runner printed more than its limit", integration=self.backend
            )
        if finished.exit_code in DOCKER_FAILED:
            logger.warning("%s: docker run exited %d", self.name, finished.exit_code)
            message = DOCKER_FAILED[finished.exit_code]
            raise IntegrationError(
                f"{self.name}: {message} (image {self.image!r}; is it pulled?)",
                integration=self.backend,
            )
        try:
            return parse(finished.stdout, backend=self.backend, duration_ms=finished.duration_ms)
        except IntegrationError:
            if finished.exit_code == 0:
                raise
        logger.warning("%s: the container was killed (exit %d)", self.name, finished.exit_code)
        note = f"the sandbox was killed (exit {finished.exit_code}); it may have run out of memory"
        return SandboxResult(exit_code=finished.exit_code, stderr=note, backend=self.backend)

    def _remove(self, docker: str, container: str) -> None:
        """Force-remove `container`; a failure is logged, the caller's kill still proceeds."""
        try:
            subprocess.run(
                [docker, "rm", "-f", container],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log_failure(logger, logging.WARNING, exc, "could not remove container %s", container)


def _check_user(user: object) -> str:
    """A numeric `uid:gid`, neither of them root (without a gid, Docker may pick group 0).

    Raises:
        ConfigError: not numeric `uid:gid`, or uid or gid 0.
    """
    match = _USER.fullmatch(user) if isinstance(user, str) else None
    if match is None or int(match.group(1)) == 0 or int(match.group(2)) == 0:
        raise ConfigError(f"user must be a numeric non-root 'uid:gid', got {user!r}")
    return user  # type: ignore[return-value]


def _word(field: str, value: object) -> str:
    """A plain command or runtime name, never an option.

    Raises:
        ConfigError: anything else.
    """
    if not isinstance(value, str) or not _WORD.fullmatch(value):
        raise ConfigError(f"bad {field} {value!r}")
    return value
