import asyncio
import logging

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ConfigError, IntegrationError, ToolUsageError
from agent_adapter.runtime.current import current_context
from agent_adapter.sandbox import (
    CallableSandbox,
    Limits,
    SandboxFile,
    SandboxRequest,
    SandboxResult,
)
from agent_adapter.sandbox._base import cap_text
from agent_adapter.tracing import EventLog
from tests.sandbox.support import recording


async def test_run_passes_a_checked_request_to_the_backend() -> None:
    sandbox, service = recording()
    result = await sandbox.run("print(2+2)", files={"in.txt": "x"}, env={"A": "1"}, timeout_s=5)
    assert result.stdout == "4\n" and result.ok and result.backend == "callable"
    [request] = service.requests
    assert request == SandboxRequest(
        code="print(2+2)",
        language="python",
        timeout_s=5.0,
        max_output_bytes=64_000,
        network=False,
        files={"in.txt": b"x"},
        env={"A": "1"},
        limits=Limits(),
    )
    assert request.code_file == "main.py"


async def test_bad_inputs_fail_before_the_backend_runs() -> None:
    sandbox, service = recording()
    with pytest.raises(ToolUsageError):
        await sandbox.run("  ")
    with pytest.raises(ToolUsageError):
        await sandbox.run("x", language="bash")
    with pytest.raises(ConfigError):
        await sandbox.run("x", timeout_s=0)
    with pytest.raises(ConfigError):
        await sandbox.run("x", max_output_bytes=0)
    with pytest.raises(ToolUsageError):
        await sandbox.run("x", files={"../a": "x"})
    assert service.requests == []


async def test_network_is_refused_unless_the_sandbox_allows_it() -> None:
    closed, service = recording()
    with pytest.raises(ConfigError, match="network"):
        await closed.run("x", network=True)
    assert service.requests == []
    opened, service = recording(network=True)
    await opened.run("x", network=True)
    assert service.requests[0].network is True


def test_languages_must_be_known_and_non_empty() -> None:
    with pytest.raises(ConfigError, match="empty"):
        CallableSandbox(lambda r: None, languages=())
    with pytest.raises(ConfigError, match="cannot run"):
        CallableSandbox(lambda r: None, languages=("ruby",))
    with pytest.raises(ConfigError):
        CallableSandbox(lambda r: None, grace_s=0)


async def test_output_over_the_limit_is_cut_even_when_the_backend_ignores_it() -> None:
    big = SandboxResult(exit_code=0, stdout="é" * 10, stderr="e" * 50)
    sandbox, _ = recording(big)
    result = await sandbox.run("x", max_output_bytes=5)
    assert result.stdout == "éé" and result.stdout_truncated
    assert result.stderr == "eeeee" and result.stderr_truncated
    assert result.truncated


async def test_files_over_the_limits_are_cut() -> None:
    files = tuple(SandboxFile(name=f"f{i}", data=b"abcd", size=4) for i in range(3))
    sandbox, _ = recording(
        SandboxResult(exit_code=0, files=files), limits=Limits(max_files=2, max_file_bytes=6)
    )
    result = await sandbox.run("x")
    assert [(f.name, f.data, f.size, f.truncated) for f in result.files] == [
        ("f0", b"abcd", 4, False),
        ("f1", b"ab", 4, True),
    ]
    assert result.files_truncated


async def test_a_file_whose_size_is_understated_is_corrected() -> None:
    lying = SandboxFile(name="a", data=b"abcdef", size=1)
    sandbox, _ = recording(SandboxResult(exit_code=0, files=(lying,)))
    [kept] = (await sandbox.run("x")).files
    assert kept.size == 6 and not kept.truncated


async def test_a_backend_returning_no_result_fails_closed() -> None:
    class Odd(CallableSandbox):
        async def _run(self, request: SandboxRequest) -> SandboxResult:
            return "nope"  # type: ignore[return-value]

    with pytest.raises(IntegrationError, match="SandboxResult"):
        await Odd(lambda r: None).run("x")


async def test_events_carry_sizes_and_exit_codes_but_no_code_or_output() -> None:
    sink = EventLog()
    ctx = ToolContext(run_id="run_1", call_id="call_1", sink=sink)
    sandbox, _ = recording(SandboxResult(exit_code=3, stdout="secret", stderr="err"), name="box")
    await sandbox.run("print('secret')", files={"a": "b"}, ctx=ctx)
    started, finished = sink.events
    assert (started.type, finished.type) == ("sandbox.started", "sandbox.finished")
    assert (finished.run_id, finished.call_id) == ("run_1", "call_1")
    assert dict(started.data) == {
        "sandbox": "box",
        "backend": "callable",
        "language": "python",
        "code_bytes": 15,
        "files": 1,
        "network": False,
        "timeout_s": 30.0,
    }
    data = dict(finished.data)
    assert data.pop("duration_ms") >= 0
    assert data == {
        "sandbox": "box",
        "backend": "callable",
        "language": "python",
        "exit_code": 3,
        "timed_out": False,
        "stdout_bytes": 6,
        "stderr_bytes": 3,
        "truncated": False,
        "files": 0,
    }


async def test_capture_content_adds_the_code_and_output() -> None:
    sink = EventLog()
    ctx = ToolContext(run_id="run_1", sink=sink)
    sandbox, _ = recording(SandboxResult(exit_code=0, stdout="4"), capture_content=True)
    await sandbox.run("print(4)", ctx=ctx)
    started, finished = sink.events
    assert started.data["arguments"] == {"code": "print(4)"}
    assert finished.data["result"] == {"stdout": "4", "stderr": ""}


async def test_a_failing_backend_emits_failed_and_chains_the_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(request: SandboxRequest) -> None:
        raise RuntimeError("token=abc")

    sink = EventLog()
    with pytest.raises(IntegrationError, match="sandbox run failed") as info:
        await CallableSandbox(broken).run("x", ctx=ToolContext(run_id="r", sink=sink))
    assert isinstance(info.value.__cause__, RuntimeError)
    failed = sink.events[-1]
    assert failed.type == "sandbox.failed" and failed.data["error_kind"] == "integration-error"
    assert "token=abc" not in caplog.text


async def test_a_typed_backend_error_keeps_its_kind() -> None:
    def denied(request: SandboxRequest) -> None:
        raise ConfigError("no")

    sink = EventLog()
    with pytest.raises(ConfigError):
        await CallableSandbox(denied).run("x", ctx=ToolContext(run_id="r", sink=sink))
    assert sink.events[-1].data["error_kind"] == "config-invalid"


async def test_a_hung_backend_is_reported_as_timed_out(caplog: pytest.LogCaptureFixture) -> None:
    async def hang(request: SandboxRequest) -> SandboxResult:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    caplog.set_level(logging.WARNING)
    result = await CallableSandbox(hang, grace_s=0.05).run("x", timeout_s=0.01)
    assert result.timed_out and result.exit_code == -1 and not result.ok
    assert "reported as timed out" in caplog.text


async def test_a_timeout_error_raised_by_the_backend_is_a_failure() -> None:
    def raises(request: SandboxRequest) -> None:
        raise TimeoutError

    with pytest.raises(IntegrationError):
        await CallableSandbox(raises).run("x")


async def test_the_backend_sees_the_callers_context() -> None:
    seen: list[ToolContext | None] = []

    def look(request: SandboxRequest) -> SandboxResult:
        seen.append(current_context())
        return SandboxResult(exit_code=0)

    await CallableSandbox(look).run("x", ctx=ToolContext(run_id="run_7"))
    assert seen[0] is not None and seen[0].run_id == "run_7"


async def test_without_any_context_the_backend_still_runs() -> None:
    sandbox, service = recording()
    assert (await sandbox.run("x")).ok and len(service.requests) == 1


def test_run_sync_runs_on_the_shared_loop() -> None:
    sandbox, _ = recording()
    assert sandbox.run_sync("print(4)", timeout=10, timeout_s=5).stdout == "4\n"


def test_cap_text_never_splits_a_character() -> None:
    assert cap_text("aé", 2) == ("a", True)
    assert cap_text("ab", 2) == ("ab", False)


async def test_unsafe_and_repeated_produced_names_are_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    names = ["../../outside.txt", "/abs/path", "C:\\win", "CON.txt", "a/../b", "ok.txt", "ok.txt"]
    raw = {"exit_code": 0, "files": [{"name": n, "data": b"x", "size": 1} for n in names]}
    sandbox, _ = recording(raw)
    result = await sandbox.run("x")
    assert [item.name for item in result.files] == ["ok.txt"]
    assert result.files_truncated
    assert "dropped 6 produced files with unsafe or repeated names" in caplog.text


async def test_run_sync_works_inside_a_running_loop() -> None:
    sandbox, _ = recording()
    assert sandbox.run_sync("print(4)", timeout=10).stdout == "4\n"
