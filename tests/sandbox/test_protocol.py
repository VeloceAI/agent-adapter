import asyncio
import base64
import json
import sys

import pytest

from agent_adapter.core.errors import IntegrationError
from agent_adapter.sandbox import Limits, SandboxRequest, _process
from agent_adapter.sandbox._bootstrap import PROTOCOL, kill_tree
from agent_adapter.sandbox._protocol import bootstrap_source, envelope_cap, parse, payload


def request(**changes: object) -> SandboxRequest:
    base: dict[str, object] = {
        "code": "print(1)",
        "language": "python",
        "timeout_s": 5.0,
        "max_output_bytes": 100,
        "network": False,
        "files": {"a.txt": b"hi"},
        "env": {"A": "1"},
        "limits": Limits(max_files=2, max_file_bytes=10),
    }
    base.update(changes)
    return SandboxRequest(**base)  # type: ignore[arg-type]


def test_payload_carries_the_request_and_no_command_for_python() -> None:
    body = json.loads(payload(request(), workdir="/work"))
    assert body == {
        "v": PROTOCOL,
        "workdir": "/work",
        "code_file": "main.py",
        "code": "print(1)",
        "command": None,
        "files": {"a.txt": base64.b64encode(b"hi").decode()},
        "env": {"A": "1"},
        "timeout_s": 5.0,
        "max_output": 100,
        "max_files": 2,
        "max_file_bytes": 10,
    }


def test_payload_names_the_command_for_other_languages() -> None:
    body = json.loads(payload(request(language="bash")))
    assert body["command"] == ["bash", "main.sh"] and body["code_file"] == "main.sh"


def test_envelope_cap_grows_with_the_limits() -> None:
    small = envelope_cap(request())
    assert envelope_cap(request(max_output_bytes=10_000)) > small > 0


def test_bootstrap_source_is_the_runner() -> None:
    source = bootstrap_source()
    assert "def main()" in source and "agent_adapter" not in source.split('"""', 2)[2]


def _envelope(**changes: object) -> bytes:
    raw: dict[str, object] = {
        "v": PROTOCOL,
        "exit_code": 0,
        "timed_out": False,
        "stdout": base64.b64encode("é".encode()).decode(),
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": True,
        "files": [{"name": "o.txt", "size": 5, "data": base64.b64encode(b"ab").decode()}],
        "files_truncated": True,
    }
    raw.update(changes)
    return b"noise\n" + json.dumps(raw).encode() + b"\n"


def test_parse_reads_the_last_line() -> None:
    result = parse(_envelope(), backend="local", duration_ms=3.0)
    assert result.stdout == "é" and result.stderr_truncated and result.files_truncated
    assert result.files[0].truncated and result.duration_ms == 3.0 and result.backend == "local"


@pytest.mark.parametrize(
    "stdout",
    [
        b"",
        b"not json",
        b"[1]",
        json.dumps({"v": 99}).encode(),
        json.dumps({"v": PROTOCOL, "error": "unsafe file name"}).encode(),
        _envelope(stdout="!!not base64!!"),
        _envelope(files=[{"name": "x"}]),
        _envelope(exit_code="x"),
    ],
)
def test_parse_refuses_anything_but_an_envelope(stdout: bytes) -> None:
    with pytest.raises(IntegrationError, match="no readable result"):
        parse(stdout, backend="local", duration_ms=0)


def _python(code: str) -> object:
    return _process.start([sys.executable, "-I", "-c", code])


async def test_run_process_feeds_stdin_and_caps_stdout() -> None:
    code = "import sys; d = sys.stdin.read(); sys.stdout.write(d * 100); sys.stderr.write('e')"
    proc = _process.start([sys.executable, "-I", "-c", code])
    done = await _process.run_process(proc, b"ab", timeout_s=20, max_stdout=10, kill=kill_tree)
    assert done.stdout == b"ab" * 5 and done.stdout_over
    assert done.stderr == b"e" and done.exit_code == 0 and not done.timed_out


async def test_run_process_kills_on_timeout() -> None:
    killed: list[int] = []

    def kill(proc: object) -> None:
        killed.append(1)
        kill_tree(proc)  # type: ignore[arg-type]

    proc = _process.start([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
    done = await _process.run_process(proc, b"", timeout_s=0.5, max_stdout=10, kill=kill)
    assert done.timed_out and done.exit_code == -1 and killed == [1]


async def test_run_process_survives_a_process_that_ignores_stdin() -> None:
    proc = _process.start([sys.executable, "-I", "-c", "pass"])
    big = b"x" * 5_000_000
    done = await _process.run_process(proc, big, timeout_s=20, max_stdout=10, kill=kill_tree)
    assert done.exit_code == 0


async def test_a_cancelled_run_kills_the_process() -> None:
    killed: list[int] = []

    def kill(proc: object) -> None:
        killed.append(1)
        kill_tree(proc)  # type: ignore[arg-type]

    proc = _process.start([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
    task = asyncio.create_task(
        _process.run_process(proc, b"", timeout_s=30, max_stdout=10, kill=kill)
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed == [1]
    await asyncio.to_thread(proc.wait, 10)
