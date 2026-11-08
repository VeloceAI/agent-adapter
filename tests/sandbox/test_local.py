import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

from agent_adapter.core.errors import ConfigError, IntegrationError
from agent_adapter.sandbox import LocalSubprocessSandbox, SandboxPort, _process, local
from agent_adapter.sandbox.local import UNSAFE_WARNING


def sandbox(**kw: object) -> LocalSubprocessSandbox:
    return LocalSubprocessSandbox(allow_unsafe_local=True, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("flag", [False, None, 1, "yes"])
def test_refuses_to_start_without_the_explicit_opt_in(flag: object) -> None:
    with pytest.raises(ConfigError, match="allow_unsafe_local=True"):
        LocalSubprocessSandbox(allow_unsafe_local=flag)  # type: ignore[arg-type]


def test_runs_python_only_and_needs_an_absolute_interpreter() -> None:
    box = sandbox()
    assert isinstance(box, SandboxPort) and box.languages == ("python",)
    with pytest.raises(ConfigError):
        sandbox(python="python")
    with pytest.raises(ConfigError):
        sandbox(python="")


async def test_runs_code_with_inputs_env_and_produced_files(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_SECRET", "abc")
    caplog.set_level(logging.WARNING)
    code = (
        "import os\n"
        "print(open('data/in.txt').read(), os.environ.get('HOST_SECRET'), os.environ['A'])\n"
        "os.makedirs('out')\n"
        "open('out/r.txt', 'w').write('done')\n"
    )
    result = await sandbox().run(code, files={"data/in.txt": "hello"}, env={"A": "1"})
    assert result.ok and result.backend == "local"
    assert result.stdout.split() == ["hello", "None", "1"]
    assert [(f.name, f.data) for f in result.files] == [("out/r.txt", b"done")]
    assert caplog.messages.count(UNSAFE_WARNING) == 1


async def test_warns_on_every_run(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    box = sandbox()
    await box.run("pass")
    await box.run("pass")
    assert caplog.messages.count(UNSAFE_WARNING) == 2


async def test_times_out_caps_output_and_reports_exit_codes() -> None:
    box = sandbox()
    slow = await box.run("import time\nprint('a', flush=True)\ntime.sleep(30)", timeout_s=1)
    assert slow.timed_out and slow.exit_code == -1 and slow.stdout.strip() == "a"
    loud = await box.run("print('x' * 100000)", max_output_bytes=10)
    assert loud.stdout == "x" * 10 and loud.stdout_truncated
    failed = await box.run("raise SystemExit(3)")
    assert failed.exit_code == 3 and not failed.ok


async def test_the_work_folder_is_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    made: list[str] = []
    real = tempfile.mkdtemp

    def mkdtemp(**kw: str) -> str:
        made.append(real(dir=tmp_path, **kw))
        return made[-1]

    monkeypatch.setattr("agent_adapter.sandbox.local.tempfile.mkdtemp", mkdtemp)
    await sandbox().run("open('x.txt', 'w').write('1')")
    assert made and not os.path.exists(made[0])


async def test_a_stuck_runner_is_killed_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stuck(proc: object, data: bytes, **kw: object) -> _process.Finished:
        proc.kill()  # type: ignore[attr-defined]
        proc.communicate()  # type: ignore[attr-defined]
        return _process.Finished(-1, b"", False, b"", True, 1.0)

    monkeypatch.setattr("agent_adapter.sandbox._process.run_process", stuck)
    result = await sandbox().run("pass")
    assert result.timed_out and result.exit_code == -1


async def test_a_flooding_runner_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def flood(proc: object, data: bytes, **kw: object) -> _process.Finished:
        proc.kill()  # type: ignore[attr-defined]
        proc.communicate()  # type: ignore[attr-defined]
        return _process.Finished(0, b"x", True, b"", False, 1.0)

    monkeypatch.setattr("agent_adapter.sandbox._process.run_process", flood)
    with pytest.raises(IntegrationError, match="more than its limit"):
        await sandbox().run("pass")


async def test_a_folder_that_cannot_be_removed_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(path: str) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr("agent_adapter.sandbox.local.shutil.rmtree", fail)
    result = await sandbox().run("print(1)")
    assert result.ok
    assert "could not remove a sandbox work folder (PermissionError)" in caplog.text
    assert "locked" not in caplog.text


def test_host_env_keeps_only_path_and_basics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOST_SECRET", "abc")
    made = local._host_env("/w")
    assert "HOST_SECRET" not in made and made["HOME"] == "/w" and "PATH" in made
    assert ("SYSTEMROOT" in made) is (sys.platform == "win32" and "SYSTEMROOT" in os.environ)
