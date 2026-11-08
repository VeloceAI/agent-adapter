"""Shared fakes for the sandbox tests: a scripted callable sandbox and its requests."""

from agent_adapter.sandbox import CallableSandbox, SandboxRequest, SandboxResult


class Recorder:
    """A sandbox service that records each request and answers with a fixed result."""

    def __init__(self, result: SandboxResult | dict[str, object] | None = None) -> None:
        self.requests: list[SandboxRequest] = []
        self.result = result if result is not None else SandboxResult(exit_code=0, stdout="4\n")

    def __call__(self, request: SandboxRequest) -> SandboxResult | dict[str, object]:
        self.requests.append(request)
        return self.result


def recording(
    result: SandboxResult | dict[str, object] | None = None, **kw: object
) -> tuple[CallableSandbox, Recorder]:
    """A `CallableSandbox` over a `Recorder`, and the recorder."""
    service = Recorder(result)
    return CallableSandbox(service, **kw), service  # type: ignore[arg-type]
