from typing import Any

import pytest

from agent_adapter import CallbackApproval, Executor, Registry
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ApprovalDenied, ConfigError, ToolUsageError
from agent_adapter.core.types import ApprovalRequest
from agent_adapter.sandbox import (
    CODE_KIND,
    InputFiles,
    LocalSubprocessSandbox,
    SandboxFile,
    SandboxResult,
    code_tool,
)
from agent_adapter.sandbox.tool import PortSandbox, shown
from agent_adapter.tracing import EventLog
from tests.sandbox.support import recording


def run_with(spec: Any, approve: bool | None = True, sink: EventLog | None = None) -> Executor:
    registry = Registry()
    registry.add(spec)
    approval = None if approve is None else CallbackApproval(lambda request: approve)
    return Executor(registry, approval=approval, sink=sink)


def test_the_tool_always_needs_confirmation() -> None:
    sandbox, _ = recording()
    spec = code_tool(sandbox)
    assert spec.confirm is True and spec.kind == CODE_KIND == "code"
    assert spec.effects == "destructive" and spec.idempotent is False
    assert spec.name == "run_code" and "no network access" in spec.description
    assert "A person approves each run" in spec.description
    schema = spec.json_schema()
    assert list(schema["properties"]) == ["code"] and schema["required"] == ["code"]


def test_code_tool_has_no_way_to_turn_approval_off() -> None:
    sandbox, _ = recording()
    with pytest.raises(TypeError):
        code_tool(sandbox, confirm=False)  # type: ignore[call-arg]


async def test_without_an_approval_port_nothing_runs() -> None:
    sandbox, service = recording()
    executor = run_with(code_tool(sandbox), approve=None)
    with pytest.raises(ApprovalDenied):
        await executor.run("run_code", {"code": "print(4)"})
    assert service.requests == []


async def test_a_denied_run_never_reaches_the_sandbox() -> None:
    sandbox, service = recording()
    with pytest.raises(ApprovalDenied):
        await run_with(code_tool(sandbox), approve=False).run("run_code", {"code": "print(4)"})
    assert service.requests == []


async def test_an_approved_run_uses_the_hosts_limits_and_shows_the_result() -> None:
    files = (SandboxFile(name="chart.png", data=b"png", size=3),)
    sandbox, service = recording(SandboxResult(exit_code=0, stdout="4\n", files=files))
    seen: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    registry = Registry()
    spec = code_tool(
        sandbox, timeout_s=5, max_output_bytes=100, env={"MODE": "test"}, files={"in.csv": "a,b"}
    )
    registry.add(spec)
    sink = EventLog()
    executor = Executor(registry, approval=CallbackApproval(approve), sink=sink)
    value = await executor.run("run_code", {"code": "print(2+2)"})
    assert seen[0].arguments == {"code": "print(2+2)"}
    [request] = service.requests
    assert (request.timeout_s, request.max_output_bytes, request.network) == (5.0, 100, False)
    assert request.env == {"MODE": "test"} and request.files == {"in.csv": b"a,b"}
    assert value["stdout"] == "4\n" and value["ok"] is True
    assert value["files"] == [{"name": "chart.png", "size": 3}]
    types = [event.type for event in sink.events]
    assert "sandbox.started" in types and "sandbox.finished" in types
    started = next(event for event in sink.events if event.type == "sandbox.started")
    assert "arguments" not in started.data


async def test_more_languages_let_the_model_choose_one() -> None:
    sandbox, service = recording(languages=("python", "bash"))
    spec = code_tool(sandbox, languages=("python", "bash"))
    schema = spec.json_schema()
    assert schema["properties"]["language"]["enum"] == ["python", "bash"]
    executor = run_with(spec)
    await executor.run("run_code", {"code": "echo hi", "language": "bash"})
    assert service.requests[0].language == "bash"
    with pytest.raises(Exception, match="language"):
        await executor.run("run_code", {"code": "x", "language": "ruby"})


@pytest.mark.parametrize("hook_async", [False, True])
async def test_on_result_gets_the_full_result(hook_async: bool) -> None:
    sandbox, _ = recording(SandboxResult(exit_code=0, stdout="x"))
    kept: list[tuple[SandboxResult, ToolContext]] = []

    def keep(result: SandboxResult, ctx: ToolContext) -> None:
        kept.append((result, ctx))

    async def keep_async(result: SandboxResult, ctx: ToolContext) -> None:
        keep(result, ctx)

    executor = run_with(code_tool(sandbox, on_result=keep_async if hook_async else keep))
    await executor.run("run_code", {"code": "x"})
    assert kept[0][0].stdout == "x" and isinstance(kept[0][1], ToolContext)


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"languages": ()}, "non-empty"),
        ({"languages": ("python", "python")}, "unique"),
        ({"languages": ("bash",)}, "does not run"),
        ({"network": True}, "network"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"max_output_bytes": 0}, "max_output_bytes"),
        ({"env": {"1": "x"}}, "environment"),
        ({"files": {"main.py": "x"}}, "overwrite"),
    ],
)
def test_bad_settings_fail_when_the_tool_is_built(kw: dict[str, Any], message: str) -> None:
    sandbox, _ = recording()
    with pytest.raises(ConfigError, match=message):
        code_tool(sandbox, **kw)


def test_a_single_language_may_be_a_str_and_needs_a_port() -> None:
    sandbox, _ = recording()
    assert code_tool(sandbox, languages="python").json_schema()["required"] == ["code"]
    with pytest.raises(ToolUsageError, match="SandboxPort"):
        code_tool(object())  # type: ignore[arg-type]


def test_network_can_be_offered_by_a_sandbox_that_allows_it() -> None:
    sandbox, _ = recording(network=True)
    spec = code_tool(sandbox, network=True, name="py", description="Run Python.", tags=("x",))
    assert spec.name == "py" and spec.description == "Run Python." and "x" in spec.tags
    assert "with network access" in code_tool(sandbox, network=True).description


class PlainPort:
    """A `SandboxPort` that is not a `GuardedSandbox`."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run(self, code: str, *, files: InputFiles | None = None, **kw: Any) -> SandboxResult:
        self.calls.append({"code": code, "files": files, **kw})
        return self.result  # type: ignore[no-any-return]


async def test_any_port_is_wrapped_with_the_shared_checks_and_limits() -> None:
    port = PlainPort(SandboxResult(exit_code=0, stdout="y" * 50))
    executor = run_with(code_tool(port, max_output_bytes=10))
    value = await executor.run("run_code", {"code": "x"})
    assert value["stdout"] == "y" * 10 and value["stdout_truncated"]
    assert port.calls[0]["timeout_s"] == 30.0 and port.calls[0]["network"] is False


async def test_a_port_returning_no_result_fails_closed() -> None:
    with pytest.raises(ToolUsageError, match="SandboxResult"):
        await PortSandbox(PlainPort("nope"), ("python",), False).run("x")


def test_a_port_may_not_offer_network_it_was_not_given() -> None:
    port = PlainPort(None)
    assert code_tool(port, network=True).name == "run_code"


def test_shown_is_what_the_model_sees() -> None:
    result = SandboxResult(exit_code=-1, timed_out=True, stderr="e", duration_ms=2.0)
    assert shown(result) == {
        "ok": False,
        "exit_code": -1,
        "timed_out": True,
        "stdout": "",
        "stderr": "e",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "files": [],
        "files_truncated": False,
        "duration_ms": 2.0,
    }


async def test_the_dev_only_local_sandbox_runs_behind_the_tool() -> None:
    box = LocalSubprocessSandbox(allow_unsafe_local=True)
    value = await run_with(code_tool(box)).run("run_code", {"code": "print(6*7)"})
    assert value["stdout"].strip() == "42"


def test_the_local_sandbox_tool_says_it_is_not_isolated() -> None:
    box = LocalSubprocessSandbox(allow_unsafe_local=True)
    text = code_tool(box).description
    assert "NO isolation" in text and "no network access" not in text
