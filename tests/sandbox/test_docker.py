import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from agent_adapter.core.errors import ConfigError, IntegrationError
from agent_adapter.sandbox import DockerSandbox, DockerStatus, SandboxPort, docker_status
from agent_adapter.sandbox._bootstrap import PROTOCOL
from agent_adapter.sandbox._process import Finished
from agent_adapter.sandbox._protocol import bootstrap_source


def test_command_line_locks_the_container_down() -> None:
    box = DockerSandbox("python:3.12-slim", cpus=0.5, memory_mb=256, pids=64)
    request = box.request("print(1)")
    assert box.command_line(request, container="aa-sandbox-1") == [
        "docker", "run", "--rm", "-i", "--name", "aa-sandbox-1",
        "--label", "agent-adapter.sandbox=1",
        "--pull", "never", "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--ipc", "none", "--user", "65534:65534", "--pids-limit", "64",
        "--memory", "256m", "--memory-swap", "256m",
        "--cpus", "0.5", "--ulimit", "nofile=1024:1024",
        "--tmpfs", "/work:rw,nosuid,nodev,noexec,size=64m,mode=1777",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777",
        "--workdir", "/work",
        "--entrypoint", "python3", "python:3.12-slim", "-I", "-S", "-c", bootstrap_source(),
    ]  # fmt: skip


def test_command_line_never_holds_the_code_env_or_files() -> None:
    box = DockerSandbox()
    request = box.request("print('code-here')", env={"TOKEN": "env-here"}, files={"f": "file-here"})
    line = " ".join(box.command_line(request, container="c"))
    assert "code-here" not in line and "env-here" not in line and "file-here" not in line


def test_network_and_runtime_only_when_asked() -> None:
    box = DockerSandbox(network=True, runtime="runsc")
    argv = box.command_line(box.request("x", network=True), container="c")
    assert argv[argv.index("--network") + 1] == "bridge"
    assert argv[argv.index("--runtime") + 1] == "runsc"
    closed = box.command_line(box.request("x"), container="c")
    assert closed[closed.index("--network") + 1] == "none"
    with pytest.raises(ConfigError, match="network"):
        DockerSandbox().request("x", network=True)


@pytest.mark.parametrize(
    "kw",
    [
        {"image": "-v"},
        {"image": "a b"},
        {"image": ""},
        {"user": "0"},
        {"user": "0:0"},
        {"user": "root"},
        {"user": "1000:x"},
        {"user": "1000:0"},
        {"user": "1000"},
        {"user": 1000},
        {"runtime": "--privileged"},
        {"python": "/bin/sh"},
        {"cpus": 0},
        {"cpus": True},
        {"cpus": 65},
        {"memory_mb": 16},
        {"pids": 4},
        {"workdir_mb": 0},
        {"languages": ("ruby",)},
    ],
)
def test_bad_settings_are_refused(kw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        DockerSandbox(**kw)


def test_more_languages_need_their_own_commands() -> None:
    box = DockerSandbox("node-and-python:1", languages=("python", "javascript", "bash"))
    assert isinstance(box, SandboxPort) and box.languages == ("python", "javascript", "bash")


def _which(path: str | None) -> Any:
    return lambda name: path


def test_docker_status_without_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_adapter.sandbox.docker.shutil.which", _which(None))
    assert docker_status() == DockerStatus(False, reason="'docker' is not on PATH")


@pytest.mark.parametrize(
    ("code", "stdout", "expected"),
    [
        (0, b"27.1.1|27.1.1|linux\n", DockerStatus(True, "27.1.1", "27.1.1")),
        (
            1,
            b"27.1.1||\n",
            DockerStatus(False, "27.1.1", reason="the Docker daemon is not reachable"),
        ),
        (
            0,
            b"27.1.1|27.1.1|windows",
            DockerStatus(
                False, "27.1.1", "27.1.1", "the daemon runs windows containers; Linux is needed"
            ),
        ),
    ],
)
def test_docker_status_reads_versions_and_os(
    monkeypatch: pytest.MonkeyPatch, code: int, stdout: bytes, expected: DockerStatus
) -> None:
    seen: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> SimpleNamespace:
        seen.append(argv)
        return SimpleNamespace(returncode=code, stdout=stdout)

    monkeypatch.setattr("agent_adapter.sandbox.docker.shutil.which", _which("/usr/bin/docker"))
    monkeypatch.setattr("agent_adapter.sandbox.docker.subprocess.run", run)
    assert DockerSandbox().status() == expected
    assert seen[0][:3] == ["/usr/bin/docker", "version", "--format"]


def test_docker_status_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(argv: list[str], **kw: Any) -> None:
        raise subprocess.TimeoutExpired(argv, 10)

    monkeypatch.setattr("agent_adapter.sandbox.docker.shutil.which", _which("/usr/bin/docker"))
    monkeypatch.setattr("agent_adapter.sandbox.docker.subprocess.run", run)
    status = docker_status()
    assert not status.available and status.reason == "docker version failed (TimeoutExpired)"


def _envelope() -> bytes:
    raw = {
        "v": PROTOCOL,
        "exit_code": 0,
        "timed_out": False,
        "stdout": "NAo=",
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": False,
        "files": [],
        "files_truncated": False,
    }
    return json.dumps(raw).encode()


class FakeDocker:
    """Stands in for the docker CLI: records argv, stdin and kills; answers with `finished`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, finished: Finished) -> None:
        self.finished = finished
        self.argv: list[str] = []
        self.stdin = b""
        self.removed: list[list[str]] = []
        self.killed = 0
        monkeypatch.setattr("agent_adapter.sandbox.docker.shutil.which", _which("/usr/bin/docker"))
        monkeypatch.setattr("agent_adapter.sandbox._process.start", self.start)
        monkeypatch.setattr("agent_adapter.sandbox._process.run_process", self.run_process)
        monkeypatch.setattr("agent_adapter.sandbox.docker.subprocess.run", self.run)

    def start(self, argv: list[str]) -> SimpleNamespace:
        self.argv = argv
        return SimpleNamespace(kill=self.kill)

    def kill(self) -> None:
        self.killed += 1

    async def run_process(self, proc: Any, data: bytes, **kw: Any) -> Finished:
        self.stdin = data
        if self.finished.timed_out:
            kw["kill"](proc)
        return self.finished

    def run(self, argv: list[str], **kw: Any) -> SimpleNamespace:
        self.removed.append(argv)
        return SimpleNamespace(returncode=0)


async def test_a_run_sends_the_payload_on_stdin_and_parses_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeDocker(monkeypatch, Finished(0, _envelope(), False, b"", False, 5.0))
    result = await DockerSandbox().run("print(4)", env={"A": "1"})
    assert result.ok and result.stdout == "4\n" and result.backend == "docker"
    assert fake.argv[0] == "/usr/bin/docker" and fake.argv[1] == "run"
    body = json.loads(fake.stdin)
    assert body["code"] == "print(4)" and body["env"] == {"A": "1"} and body["workdir"] == "/work"
    assert fake.removed == []


async def test_a_timeout_removes_the_container(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeDocker(monkeypatch, Finished(-1, b"", False, b"", True, 5.0))
    result = await DockerSandbox().run("while True: pass", timeout_s=1)
    assert result.timed_out and result.exit_code == -1
    name = fake.argv[fake.argv.index("--name") + 1]
    assert name.startswith("aa-sandbox-")
    assert fake.removed == [["/usr/bin/docker", "rm", "-f", name]] and fake.killed == 1
    assert "removed it" in caplog.text


async def test_a_failed_removal_is_logged_and_the_client_still_killed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeDocker(monkeypatch, Finished(-1, b"", False, b"", True, 5.0))

    def broken(argv: list[str], **kw: Any) -> None:
        raise OSError("gone")

    monkeypatch.setattr("agent_adapter.sandbox.docker.subprocess.run", broken)
    await DockerSandbox().run("x")
    assert fake.killed == 1 and "could not remove container" in caplog.text


@pytest.mark.parametrize(
    ("finished", "message"),
    [
        (Finished(125, b"", False, b"daemon text", False, 1.0), "could not start the container"),
        (Finished(127, b"", False, b"", False, 1.0), "no such command"),
        (Finished(0, b"x", True, b"", False, 1.0), "more than its limit"),
        (Finished(0, b"garbage", False, b"", False, 1.0), "no readable result"),
    ],
)
async def test_docker_failures_raise_integration_error(
    monkeypatch: pytest.MonkeyPatch, finished: Finished, message: str
) -> None:
    FakeDocker(monkeypatch, finished)
    with pytest.raises(IntegrationError, match=message):
        await DockerSandbox().run("x")


async def test_a_killed_container_is_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeDocker(monkeypatch, Finished(137, b"", False, b"", False, 1.0))
    result = await DockerSandbox().run("x")
    assert result.exit_code == 137 and "out of memory" in result.stderr and not result.ok


async def test_no_cli_is_an_integration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_adapter.sandbox.docker.shutil.which", _which(None))
    with pytest.raises(IntegrationError, match="not on PATH"):
        await DockerSandbox().run("x")


REAL_IMAGE = "python:3.12-slim"


def _real_docker() -> bool:
    if not docker_status().available:
        return False
    probe = subprocess.run(["docker", "image", "inspect", REAL_IMAGE], capture_output=True)
    return probe.returncode == 0


@pytest.mark.contract
async def test_real_docker_runs_isolated() -> None:  # pragma: no cover - needs Docker
    if not _real_docker():
        pytest.skip(f"needs Docker with Linux containers and {REAL_IMAGE} pulled")
    box = DockerSandbox(REAL_IMAGE)
    code = (
        "import os, socket\n"
        "print(os.getuid())\n"
        "open('out.txt', 'w').write('ok')\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    print('net')\n"
        "except OSError:\n"
        "    print('no-net')\n"
        "try:\n"
        "    open('/etc/x', 'w')\n"
        "except OSError:\n"
        "    print('read-only')\n"
    )
    result = await box.run(code, timeout_s=60)
    assert result.ok and result.stdout.split() == ["65534", "no-net", "read-only"]
    assert [(f.name, f.data) for f in result.files] == [("out.txt", b"ok")]
    slow = await box.run("import time\ntime.sleep(60)", timeout_s=2)
    assert slow.timed_out


async def test_a_forged_envelope_cannot_plant_unsafe_file_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.loads(_envelope())
    raw["files"] = [
        {"name": "../../etc/cron.d/x", "size": 1, "data": "eA=="},
        {"name": "/abs", "size": 1, "data": "eA=="},
        {"name": "chart.png", "size": 1, "data": "eA=="},
    ]
    forged = b"real output\n" + json.dumps(raw).encode()
    FakeDocker(monkeypatch, Finished(0, forged, False, b"", False, 1.0))
    result = await DockerSandbox().run("x")
    assert [item.name for item in result.files] == ["chart.png"] and result.files_truncated


async def test_a_runner_error_envelope_at_exit_zero_is_an_integration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = json.dumps({"v": PROTOCOL, "error": "unsafe file name"}).encode()
    FakeDocker(monkeypatch, Finished(0, refused, False, b"", False, 1.0))
    with pytest.raises(IntegrationError, match="no readable result"):
        await DockerSandbox().run("x")
