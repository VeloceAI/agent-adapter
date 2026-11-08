import pytest

from agent_adapter.core.errors import IntegrationError, ToolUsageError
from agent_adapter.sandbox import CallableSandbox, SandboxPort, SandboxRequest, SandboxResult
from agent_adapter.sandbox.callable import as_result


def test_callable_sandbox_is_a_port_and_needs_a_callable() -> None:
    assert isinstance(CallableSandbox(lambda r: None), SandboxPort)
    with pytest.raises(ToolUsageError):
        CallableSandbox("nope")  # type: ignore[arg-type]


async def test_async_fn_and_mapping_results_with_named_files() -> None:
    async def service(request: SandboxRequest) -> dict[str, object]:
        return {"exit_code": 0, "stdout": "ok", "files": {"out.txt": "hi", "raw.bin": b"\x01"}}

    result = await CallableSandbox(service).run("x")
    assert result.stdout == "ok"
    assert [(f.name, f.data, f.size) for f in result.files] == [
        ("out.txt", b"hi", 2),
        ("raw.bin", b"\x01", 1),
    ]


async def test_sync_fn_returning_an_awaitable_is_awaited() -> None:
    async def later() -> SandboxResult:
        return SandboxResult(exit_code=1)

    result = await CallableSandbox(lambda r: later()).run("x")
    assert result.exit_code == 1 and not result.ok


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "text",
        {"stdout": "no exit code"},
        {"exit_code": 0, "files": {"": "x"}},
        {"exit_code": 0, "files": {"a": 3}},
        {"exit_code": 0, "extra": 1},
    ],
)
def test_invalid_results_raise_integration_error(raw: object) -> None:
    with pytest.raises(IntegrationError):
        as_result(raw, "svc")


def test_result_helpers() -> None:
    result = as_result({"exit_code": 0, "files": [{"name": "a", "data": b"x", "size": 3}]}, "s")
    [item] = result.files
    assert item.truncated and item.text() == "x"
    assert result.truncated is False
